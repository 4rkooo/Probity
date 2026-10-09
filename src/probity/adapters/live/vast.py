"""Live VAST adapter shell.

No official VAST AI OS client, collection/index names, or SDK is bound here. An optional
``VastTransport`` may be injected so the official client can be wired once event docs exist.
Sponsor-specific types never leave this module.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol

import anyio
from pydantic import SecretStr, ValidationError

from probity.domain.enums import AdapterMode, HealthStatus
from probity.domain.errors import SponsorMalformedResponse, SponsorTimeout, SponsorUnavailable
from probity.domain.ids import utc_now
from probity.domain.models import (
    AdapterHealth,
    Detection,
    Embedding,
    EvidenceReport,
    LineageRecord,
    RetrievedSegment,
    SourceVideo,
    VideoSegment,
)
from probity.ports import SearchRequest

SleepFn = Callable[[float], Awaitable[None]]
READ_RETRY_DELAYS_S: tuple[float, ...] = (2.0, 5.0)
VAST_TIMEOUT_S = 30.0
HEALTH_BUDGET_S = 2.0


class UncommittedWrite(Exception):
    """Transport proved a write was not committed. Adapter may retry the write once."""


class VastTransport(Protocol):
    """Probity-owned injection point. Bind the official client once docs exist."""

    async def execute(self, operation: str, payload: Mapping[str, Any]) -> Any:
        """Return JSON-compatible data for a Probity protocol operation.

        Operation names are Probity protocol methods, not vendor APIs. Raise
        ``UncommittedWrite`` when a write is proven not to have been committed.
        """
        ...


class LiveVastEvidenceStore:
    """EvidenceStore shell. Disabled unless ``enabled`` and a transport is bound."""

    adapter_name = "vast-live"
    schema_version = "1.0"

    def __init__(
        self,
        *,
        enabled: bool,
        endpoint: str | None = None,
        token: SecretStr | None = None,
        transport: VastTransport | None = None,
        model_id: str | None = None,
        timeout_s: float = VAST_TIMEOUT_S,
        read_retry_delays_s: tuple[float, ...] = READ_RETRY_DELAYS_S,
        sleep: SleepFn | None = None,
    ) -> None:
        self._enabled = enabled
        self._endpoint = endpoint
        self._token = token
        self._transport = transport
        self._model_id = model_id
        self._timeout_s = timeout_s
        self._read_retry_delays_s = read_retry_delays_s
        self._sleep = sleep

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def mode(self) -> AdapterMode:
        if not self._enabled or self._transport is None:
            return AdapterMode.DISABLED
        return AdapterMode.LIVE

    def _bound(self) -> bool:
        return self._enabled and self._transport is not None

    async def _sleep_for(self, seconds: float) -> None:
        if self._sleep is not None:
            await self._sleep(seconds)
            return
        await anyio.sleep(seconds)

    async def _call(self, operation: str, payload: Mapping[str, Any]) -> Any:
        if not self._bound() or self._transport is None:
            raise SponsorUnavailable("vast live adapter is disabled or has no transport bound")
        with anyio.fail_after(self._timeout_s):
            return await self._transport.execute(operation, payload)

    async def _read(self, operation: str, payload: Mapping[str, Any]) -> Any:
        delays = self._read_retry_delays_s
        attempts = len(delays) + 1
        last_timeout = False
        last_unavailable: SponsorUnavailable | None = None
        for attempt in range(attempts):
            try:
                return await self._call(operation, payload)
            except SponsorMalformedResponse:
                raise
            except TimeoutError:
                last_timeout = True
            except SponsorTimeout:
                last_timeout = True
            except SponsorUnavailable as exc:
                last_unavailable = exc
            if attempt < len(delays):
                await self._sleep_for(delays[attempt])
        if last_unavailable is not None and not last_timeout:
            raise last_unavailable
        raise SponsorTimeout(f"vast {operation} exhausted read retry budget")

    async def _write(self, operation: str, payload: Mapping[str, Any]) -> Any:
        try:
            return await self._call(operation, payload)
        except UncommittedWrite:
            try:
                return await self._call(operation, payload)
            except UncommittedWrite as exc:
                raise SponsorTimeout(f"vast {operation} write remained uncommitted") from exc
        except SponsorMalformedResponse:
            raise
        except (TimeoutError, SponsorTimeout) as exc:
            raise SponsorTimeout(f"vast {operation} timed out") from exc

    async def health(self) -> AdapterHealth:
        started = time.perf_counter()
        if not self._enabled:
            return AdapterHealth(
                adapter_name=self.adapter_name,
                model_id=self.model_id,
                mode=AdapterMode.DISABLED,
                status=HealthStatus.DISABLED,
                checked_at=utc_now(),
                latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
                detail="PROBITY_VAST_ENABLED is false",
            )
        if self._transport is None:
            return AdapterHealth(
                adapter_name=self.adapter_name,
                model_id=self.model_id,
                mode=AdapterMode.DISABLED,
                status=HealthStatus.UNAVAILABLE,
                checked_at=utc_now(),
                latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
                detail="no transport bound; official VAST client is not configured",
            )
        try:
            with anyio.fail_after(HEALTH_BUDGET_S):
                await self._transport.execute("health", {})
            status = HealthStatus.OK
            detail = None
            mode = AdapterMode.LIVE
        except (TimeoutError, SponsorTimeout, SponsorUnavailable, SponsorMalformedResponse) as exc:
            status = HealthStatus.UNAVAILABLE
            detail = type(exc).__name__
            mode = AdapterMode.LIVE
        return AdapterHealth(
            adapter_name=self.adapter_name,
            model_id=self.model_id,
            mode=mode,
            status=status,
            checked_at=utc_now(),
            latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
            detail=detail,
        )

    async def put_source(self, video: SourceVideo, local_path: str) -> str:
        raw = await self._write(
            "put_source",
            {"video": video.model_dump(mode="json"), "local_path": local_path},
        )
        if isinstance(raw, str) and raw:
            return raw
        if isinstance(raw, Mapping) and isinstance(raw.get("storage_uri"), str):
            return str(raw["storage_uri"])
        raise SponsorMalformedResponse("vast put_source returned no storage_uri")

    async def upsert_segments(
        self, segments: Sequence[VideoSegment], embeddings: Sequence[Embedding]
    ) -> None:
        await self._write(
            "upsert_segments",
            {
                "segments": [item.model_dump(mode="json") for item in segments],
                "embeddings": [item.model_dump(mode="json") for item in embeddings],
            },
        )

    async def upsert_detections(self, detections: Sequence[Detection]) -> None:
        await self._write(
            "upsert_detections",
            {"detections": [item.model_dump(mode="json") for item in detections]},
        )

    async def search(self, request: SearchRequest) -> Sequence[RetrievedSegment]:
        raw = await self._read("search", request.model_dump(mode="json"))
        try:
            if not isinstance(raw, Sequence):
                raise SponsorMalformedResponse("vast search returned a non-list")
            return tuple(RetrievedSegment.model_validate(item) for item in raw)
        except ValidationError as exc:
            raise SponsorMalformedResponse(
                "vast search returned invalid RetrievedSegment rows"
            ) from exc

    async def put_lineage(self, record: LineageRecord) -> None:
        await self._write("put_lineage", record.model_dump(mode="json"))

    async def put_report(self, report: EvidenceReport) -> None:
        await self._write("put_report", report.model_dump(mode="json"))
