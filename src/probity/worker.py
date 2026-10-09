"""Single-consumer leased job worker (``python -m probity.worker``)."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import signal
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from probity.config import Settings, get_settings
from probity.domain.enums import ErrorCode, JobStage, JobState
from probity.domain.errors import JobCancelled, ProbityError
from probity.domain.ids import format_utc, new_uuid7
from probity.domain.models import JobError, PartialCoverage
from probity.jobs import SqlJobStore
from probity.persistence.db import create_engine_for, database_path, run_migrations
from probity.ports import ClaimedJob, JobHandler, JobOutcome, JobStore

HandlerFactory = Callable[[Settings], Sequence[JobHandler]]

_HANDLER_FACTORY: HandlerFactory | None = None
INTERNAL_MESSAGE = "Internal error"


def register_handler_factory(factory: HandlerFactory) -> None:
    """Install the factory used by ``python -m probity.worker`` and ``load_handlers``."""
    global _HANDLER_FACTORY
    _HANDLER_FACTORY = factory


def load_handlers(settings: Settings) -> Sequence[JobHandler]:
    if _HANDLER_FACTORY is not None:
        return list(_HANDLER_FACTORY(settings))
    spec = os.environ.get("PROBITY_HANDLER_FACTORY", "").strip()
    if not spec:
        return []
    module_name, _, attr = spec.partition(":")
    attr = attr or "handler_factory"
    module = importlib.import_module(module_name)
    factory: HandlerFactory = getattr(module, attr)
    return list(factory(settings))


def redact_db_path(url: str) -> str:
    path = database_path(url)
    if path == ":memory:":
        return ":memory:"
    return Path(path).name


def json_log(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    job_id: str | None = None,
    correlation_id: str | None = None,
    latency_ms: int | None = None,
    error_code: str | None = None,
    error_type: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "ts": format_utc(datetime.now(UTC)),
        "level": logging.getLevelName(level),
        "component": "worker",
        "event": event,
    }
    if job_id is not None:
        payload["job_id"] = job_id
    if correlation_id is not None:
        payload["correlation_id"] = correlation_id
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms
    if error_code is not None:
        payload["error_code"] = error_code
    if error_type is not None:
        payload["error_type"] = error_type
    logger.log(level, json.dumps(payload, sort_keys=True, separators=(",", ":")))


class _Cancel:
    def __init__(self, store: JobStore, job_id: str, stopping: Callable[[], bool]) -> None:
        self._store = store
        self._job_id = job_id
        self._stopping = stopping

    def is_cancelled(self) -> bool:
        return self._stopping() or self._store.is_cancel_requested(self._job_id)

    def checkpoint(self) -> None:
        if self.is_cancelled():
            raise JobCancelled("job cancellation requested")


class WorkerJobContext:
    """Concrete ``JobContext`` with ``record_partial`` for cooperative cancel."""

    def __init__(
        self,
        store: JobStore,
        job_id: str,
        owner: str,
        stopping: Callable[[], bool],
    ) -> None:
        self._store = store
        self._job_id = job_id
        self._owner = owner
        self._partial: PartialCoverage | None = None
        self.cancel = _Cancel(store, job_id, stopping)

    def progress(self, stage: JobStage, completed: int, total: int) -> None:
        self._store.report_progress(self._job_id, self._owner, stage, completed, total)

    def record_partial(self, partial: PartialCoverage) -> None:
        self._partial = partial

    @property
    def partial(self) -> PartialCoverage | None:
        return self._partial


class Worker:
    def __init__(
        self,
        job_store: JobStore,
        handlers: Sequence[JobHandler],
        owner_id: str,
        lease_seconds: int = 30,
        renew_seconds: float | None = None,
        poll_seconds: float = 0.5,
        logger: logging.Logger | None = None,
    ) -> None:
        self._store = job_store
        self._handlers = {handler.kind: handler for handler in handlers}
        self.owner_id = owner_id
        self.lease_seconds = lease_seconds
        self.renew_seconds = lease_seconds / 3 if renew_seconds is None else renew_seconds
        self.poll_seconds = poll_seconds
        self._logger = logger or logging.getLogger("probity.worker")
        self._stop = asyncio.Event()
        self._current_job_id: str | None = None
        self.recovered_count = 0

    def request_stop(self) -> None:
        self._stop.set()
        job_id = self._current_job_id
        if job_id is not None:
            with suppress(Exception):
                self._store.request_cancel(job_id)

    def _stopping(self) -> bool:
        return self._stop.is_set()

    def recover(self) -> int:
        recovered = list(self._store.recover_expired())
        self.recovered_count = len(recovered)
        json_log(
            self._logger,
            "worker.recovered",
            job_id=None,
            latency_ms=None,
            error_code=None,
        )
        return self.recovered_count

    async def run(self, *, recover: bool = True) -> None:
        if recover:
            self.recover()
        while not self._stop.is_set():
            claimed = self._store.claim_next(self.owner_id, self.lease_seconds)
            if claimed is None:
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
                continue
            await self._execute(claimed)

    async def _execute(self, claimed: ClaimedJob) -> None:
        job = claimed.job
        self._current_job_id = job.job_id
        started = datetime.now(UTC)
        json_log(
            self._logger,
            "job.claimed",
            job_id=job.job_id,
            correlation_id=job.correlation_id,
        )
        ctx = WorkerJobContext(self._store, job.job_id, self.owner_id, self._stopping)
        handler = self._handlers.get(job.kind)
        renew_task = asyncio.create_task(self._renew_loop(job.job_id))
        try:
            if handler is None:
                json_log(
                    self._logger,
                    "job.missing_handler",
                    level=logging.ERROR,
                    job_id=job.job_id,
                    correlation_id=job.correlation_id,
                    error_code=ErrorCode.INTERNAL_ERROR.value,
                    error_type="MissingHandler",
                )
                self._fail(job.job_id, ErrorCode.INTERNAL_ERROR, INTERNAL_MESSAGE, retryable=False)
                return
            try:
                outcome = await handler.run(claimed, ctx)
            except JobCancelled:
                self._store.mark_cancelled(job.job_id, self.owner_id, ctx.partial)
                json_log(
                    self._logger,
                    "job.cancelled",
                    job_id=job.job_id,
                    correlation_id=job.correlation_id,
                    error_code=ErrorCode.CANCELLED.value,
                    latency_ms=_latency_ms(started),
                )
                return
            except ProbityError as exc:
                json_log(
                    self._logger,
                    "job.failed",
                    level=logging.ERROR,
                    job_id=job.job_id,
                    correlation_id=job.correlation_id,
                    error_code=exc.code.value,
                    error_type=type(exc).__name__,
                    latency_ms=_latency_ms(started),
                )
                self._fail(job.job_id, exc.code, exc.message[:500], retryable=exc.retryable)
                return
            except Exception as exc:
                json_log(
                    self._logger,
                    "job.failed",
                    level=logging.ERROR,
                    job_id=job.job_id,
                    correlation_id=job.correlation_id,
                    error_code=ErrorCode.INTERNAL_ERROR.value,
                    error_type=type(exc).__name__,
                    latency_ms=_latency_ms(started),
                )
                self._fail(job.job_id, ErrorCode.INTERNAL_ERROR, INTERNAL_MESSAGE, retryable=False)
                return
            try:
                self._store.complete(job.job_id, self.owner_id, outcome)
            except ProbityError as exc:
                json_log(
                    self._logger,
                    "job.complete_rejected",
                    level=logging.ERROR,
                    job_id=job.job_id,
                    correlation_id=job.correlation_id,
                    error_code=exc.code.value,
                    error_type=type(exc).__name__,
                )
                self._fail(job.job_id, ErrorCode.INTERNAL_ERROR, INTERNAL_MESSAGE, retryable=False)
                return
            json_log(
                self._logger,
                "job.completed",
                job_id=job.job_id,
                correlation_id=job.correlation_id,
                latency_ms=_latency_ms(started),
            )
        finally:
            renew_task.cancel()
            with suppress(asyncio.CancelledError):
                await renew_task
            self._current_job_id = None

    async def _renew_loop(self, job_id: str) -> None:
        while True:
            await asyncio.sleep(self.renew_seconds)
            if not self._store.renew_lease(job_id, self.owner_id, self.lease_seconds):
                return

    def _fail(self, job_id: str, code: ErrorCode, message: str, *, retryable: bool) -> None:
        outcome = JobOutcome(
            state=JobState.FAILED,
            error=JobError(code=code, message=message, retryable=retryable),
        )
        with suppress(Exception):
            self._store.complete(job_id, self.owner_id, outcome)


def _latency_ms(started: datetime) -> int:
    return max(0, int((datetime.now(UTC) - started).total_seconds() * 1000))


def _configure_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("probity.worker")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def health_line(worker_id: str, database_url: str, recovered: int) -> str:
    return json.dumps(
        {
            "process": "worker",
            "worker_id": worker_id,
            "database": redact_db_path(database_url),
            "recovered": recovered,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _install_default_handlers() -> None:
    """Register ingest and fixture mocks when the process did not install its own."""
    if _HANDLER_FACTORY is not None:
        return
    if os.environ.get("PROBITY_HANDLER_FACTORY", "").strip():
        return
    from probity.wiring import handler_factory

    register_handler_factory(handler_factory)


def build_worker(settings: Settings | None = None) -> tuple[Worker, str]:
    settings = settings or get_settings()
    _install_default_handlers()
    url = settings.resolved_database_url
    run_migrations(url)
    engine = create_engine_for(url)
    owner = settings.worker_id or f"worker-{new_uuid7()}"
    store = SqlJobStore(engine, lease_seconds=settings.lease_seconds)
    logger = _configure_logging(settings.log_level)
    worker = Worker(
        store,
        load_handlers(settings),
        owner,
        lease_seconds=settings.lease_seconds,
        poll_seconds=settings.worker_poll_seconds,
        logger=logger,
    )
    return worker, url


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    del argv
    settings = get_settings()
    worker, url = build_worker(settings)
    recovered = worker.recover()
    print(health_line(worker.owner_id, url, recovered), file=stdout or sys.stdout, flush=True)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _stop(*_args: object) -> None:
        worker.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _stop)
    try:
        loop.run_until_complete(worker.run(recover=False))
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
