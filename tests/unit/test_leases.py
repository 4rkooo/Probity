"""Lease claim exclusivity, owner checks, expiry recovery, migrated schema match."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import inspect

from probity.domain.enums import JobState
from probity.domain.errors import InvalidStateTransition, NotFound
from probity.domain.ids import new_uuid7
from probity.jobs import SqlJobStore
from probity.persistence.db import create_engine_for, run_migrations
from probity.persistence.schema import metadata
from probity.ports import IngestPayload, JobSpec


def _engine(path: Path):
    url = f"sqlite:///{path}"
    engine = create_engine_for(url)
    run_migrations(url, engine=engine)
    return engine, url


def _spec() -> JobSpec:
    video_id = new_uuid7()
    return JobSpec(
        job_id=new_uuid7(),
        case_id=new_uuid7(),
        subject_id=video_id,
        payload=IngestPayload(video_id=video_id),
        total_units=3,
        correlation_id=new_uuid7(),
    )


class _Clock:
    def __init__(self) -> None:
        self.t = datetime(2026, 10, 9, 16, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t = self.t + timedelta(seconds=seconds)


def test_migrated_schema_matches_metadata(tmp_path: Path) -> None:
    engine, _url = _engine(tmp_path / "cases.db")
    insp = inspect(engine)
    assert set(insp.get_table_names()) >= set(metadata.tables)
    for table in metadata.tables.values():
        cols = {c["name"] for c in insp.get_columns(table.name)}
        expected = {c.name for c in table.columns}
        assert cols == expected, table.name
    engine.dispose()


def test_two_threads_claim_next_exactly_one_winner(tmp_path: Path) -> None:
    db = tmp_path / "cases.db"
    engine, url = _engine(db)
    SqlJobStore(engine, lease_seconds=30).create(_spec())
    engine.dispose()
    barrier = threading.Barrier(2)
    results: list[object] = []
    lock = threading.Lock()

    def claim(owner: str) -> None:
        local = create_engine_for(url)
        store = SqlJobStore(local, lease_seconds=30)
        barrier.wait()
        got = store.claim_next(owner, 30)
        with lock:
            results.append(got)
        local.dispose()

    t1 = threading.Thread(target=claim, args=("owner-a",))
    t2 = threading.Thread(target=claim, args=("owner-b",))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    assert len(results) == 2


def test_non_owner_renew_fails(tmp_path: Path) -> None:
    engine, _url = _engine(tmp_path / "cases.db")
    store = SqlJobStore(engine, lease_seconds=30)
    store.create(_spec())
    claimed = store.claim_next("owner-a", 30)
    assert claimed is not None
    assert store.renew_lease(claimed.job.job_id, "owner-b", 30) is False
    assert store.renew_lease(claimed.job.job_id, "owner-a", 30) is True
    with pytest.raises(InvalidStateTransition):
        store.report_progress(claimed.job.job_id, "owner-b", claimed.job.stage, 1, 3)  # type: ignore[arg-type]
    engine.dispose()


def test_expiry_and_startup_recovery(tmp_path: Path) -> None:
    clock = _Clock()
    engine, _url = _engine(tmp_path / "cases.db")
    store = SqlJobStore(engine, clock=clock, lease_seconds=5)
    created = store.create(_spec())
    claimed = store.claim_next("dead-worker", 5)
    assert claimed is not None
    clock.advance(1)
    assert store.recover_expired() == []
    clock.advance(30)
    recovered = store.recover_expired()
    assert len(recovered) == 1
    view = store.get(created.job_id)
    assert view.state is JobState.FAILED
    assert view.error is not None and view.error.retryable is True
    engine.dispose()


def test_expired_cancelling_becomes_cancelled(tmp_path: Path) -> None:
    clock = _Clock()
    engine, _url = _engine(tmp_path / "cases.db")
    store = SqlJobStore(engine, clock=clock, lease_seconds=5)
    spec = _spec()
    store.create(spec)
    store.claim_next("w", 5)
    store.request_cancel(spec.job_id)
    assert store.get(spec.job_id).state is JobState.CANCELLING
    clock.advance(30)
    recovered = store.recover_expired()
    assert recovered[0].state is JobState.CANCELLED
    engine.dispose()


def test_get_missing_job_is_not_found(tmp_path: Path) -> None:
    engine, _url = _engine(tmp_path / "cases.db")
    store = SqlJobStore(engine, lease_seconds=30)
    with pytest.raises(NotFound):
        store.get(new_uuid7())
    engine.dispose()
