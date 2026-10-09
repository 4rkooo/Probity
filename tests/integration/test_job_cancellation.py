"""QUEUED cancel is immediate; RUNNING cancel checkpoints and keeps partial coverage."""

from __future__ import annotations

import asyncio
from pathlib import Path

from probity.domain.enums import JobKind, JobStage, JobState
from probity.domain.ids import new_uuid7, segment_id
from probity.domain.models import PartialCoverage, TimeRangeUs
from probity.jobs import SqlJobStore
from probity.persistence.db import create_engine_for, run_migrations
from probity.ports import ClaimedJob, IngestPayload, JobOutcome, JobSpec
from probity.worker import Worker, WorkerJobContext


def _engine(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'cases.db'}"
    engine = create_engine_for(url)
    run_migrations(url, engine=engine)
    return engine


def _spec(video_id: str | None = None) -> JobSpec:
    video_id = video_id or new_uuid7()
    return JobSpec(
        job_id=new_uuid7(),
        case_id=new_uuid7(),
        subject_id=video_id,
        payload=IngestPayload(video_id=video_id),
        total_units=4,
        correlation_id=new_uuid7(),
    )


def test_queued_cancel_is_immediately_cancelled_with_two_events(tmp_path: Path) -> None:
    store = SqlJobStore(_engine(tmp_path), lease_seconds=30)
    spec = _spec()
    created = store.create(spec)
    assert created.state is JobState.QUEUED
    view = store.request_cancel(spec.job_id)
    assert view.state is JobState.CANCELLED
    assert view.finished_at is not None
    events = store.events(spec.job_id)
    hops = [e for e in events if e.to_state in {JobState.CANCELLING, JobState.CANCELLED}]
    assert len(hops) == 2
    assert hops[0].from_state is JobState.QUEUED and hops[0].to_state is JobState.CANCELLING
    assert hops[1].from_state is JobState.CANCELLING and hops[1].to_state is JobState.CANCELLED
    again = store.request_cancel(spec.job_id)
    assert again.state is JobState.CANCELLED
    assert len(store.events(spec.job_id)) == 4


class _GatedHandler:
    kind = JobKind.INGEST

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, claimed: ClaimedJob, ctx: WorkerJobContext) -> JobOutcome:
        video_id = claimed.payload.video_id  # type: ignore[union-attr]
        ctx.progress(JobStage.INDEX, 1, 4)
        ctx.record_partial(
            PartialCoverage(
                committed_segment_ids=(segment_id(video_id, 0),),
                indexed_ranges=(TimeRangeUs(start_pts_us=0, end_pts_us=1_000_000),),
            )
        )
        self.started.set()
        await self.release.wait()
        ctx.cancel.checkpoint()
        return JobOutcome(state=JobState.SUCCEEDED)


def test_running_cancel_at_checkpoint_keeps_partial(tmp_path: Path) -> None:
    store = SqlJobStore(_engine(tmp_path), lease_seconds=30)
    spec = _spec()
    store.create(spec)
    handler = _GatedHandler()
    worker = Worker(store, [handler], "worker-1", lease_seconds=30, poll_seconds=0.01)

    async def _run() -> None:
        task = asyncio.create_task(worker.run())
        await asyncio.wait_for(handler.started.wait(), timeout=5)
        view = store.request_cancel(spec.job_id)
        assert view.state is JobState.CANCELLING
        handler.release.set()
        for _ in range(200):
            if store.get(spec.job_id).state is JobState.CANCELLED:
                break
            await asyncio.sleep(0.02)
        worker.request_stop()
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(_run())
    done = store.get(spec.job_id)
    assert done.state is JobState.CANCELLED
    assert done.partial is not None
    assert len(done.partial.committed_segment_ids) == 1
