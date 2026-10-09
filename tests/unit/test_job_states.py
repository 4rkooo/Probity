"""Job state machine: every JobState pair, PARTIAL/REFUSED rules, JobView validators."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from probity.domain.enums import (
    ALLOWED_JOB_TRANSITIONS,
    TERMINAL_JOB_STATES,
    JobKind,
    JobStage,
    JobState,
    ReasonCode,
    SubjectType,
)
from probity.domain.errors import InvalidStateTransition
from probity.domain.ids import new_uuid7, segment_id
from probity.domain.models import JobView, PartialCoverage, TimeRangeUs
from probity.jobs import SqlJobStore, validate_transition
from probity.persistence.db import create_engine_for, run_migrations, utc_clock
from probity.ports import IngestPayload, JobOutcome, JobSpec, TrackPayload, TrackRequest


def _engine(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'cases.db'}"
    engine = create_engine_for(url)
    run_migrations(url, engine=engine)
    return engine


def _ingest_spec(**kwargs: object) -> JobSpec:
    video_id = str(kwargs.pop("video_id", new_uuid7()))
    return JobSpec(
        job_id=str(kwargs.pop("job_id", new_uuid7())),
        case_id=str(kwargs.pop("case_id", new_uuid7())),
        subject_id=video_id,
        payload=IngestPayload(video_id=video_id),
        total_units=int(kwargs.pop("total_units", 4)),
        correlation_id=str(kwargs.pop("correlation_id", new_uuid7())),
    )


def _track_spec() -> JobSpec:
    video_id = new_uuid7()
    track_id = new_uuid7()
    case_id = new_uuid7()
    return JobSpec(
        job_id=new_uuid7(),
        case_id=case_id,
        subject_id=track_id,
        payload=TrackPayload(
            request=TrackRequest(
                track_id=track_id,
                case_id=case_id,
                video_id=video_id,
                source_sha256="0" * 64,
                subject_type=SubjectType.LICENSE_PLATE,
                seed_frame_id=f"{video_id}:f0",
                seed_bbox_px=(0, 0, 48, 20),
            )
        ),
        total_units=1,
        correlation_id=new_uuid7(),
    )


def _partial(video_id: str) -> PartialCoverage:
    return PartialCoverage(
        committed_segment_ids=(segment_id(video_id, 0),),
        indexed_ranges=(TimeRangeUs(start_pts_us=0, end_pts_us=1_000_000),),
    )


@pytest.mark.parametrize("src", list(JobState))
@pytest.mark.parametrize("dst", list(JobState))
def test_every_transition_pair(src: JobState, dst: JobState) -> None:
    if dst in ALLOWED_JOB_TRANSITIONS[src]:
        validate_transition(src, dst)
    else:
        with pytest.raises(InvalidStateTransition):
            validate_transition(src, dst)


@given(st.sampled_from(list(JobState)), st.sampled_from(list(JobState)))
@settings(max_examples=80)
def test_transition_property(src: JobState, dst: JobState) -> None:
    allowed = dst in ALLOWED_JOB_TRANSITIONS[src]
    if allowed:
        validate_transition(src, dst)
    else:
        with pytest.raises(InvalidStateTransition):
            validate_transition(src, dst)


def test_terminal_states_have_no_exits() -> None:
    for state in TERMINAL_JOB_STATES:
        assert ALLOWED_JOB_TRANSITIONS[state] == frozenset()


def test_create_returns_queued_with_created_event(tmp_path: Path) -> None:
    store = SqlJobStore(_engine(tmp_path), clock=utc_clock, lease_seconds=30)
    spec = _ingest_spec()
    view = store.create(spec)
    assert view.state is JobState.QUEUED
    assert view.job_id == spec.job_id
    events = store.events(spec.job_id)
    assert [e.to_state for e in events] == [JobState.CREATED, JobState.QUEUED]
    assert events[0].from_state is None
    JobView.model_validate(view.model_dump(mode="json"))


def test_claim_sets_first_stage(tmp_path: Path) -> None:
    store = SqlJobStore(_engine(tmp_path), lease_seconds=30)
    ingest = store.create(_ingest_spec())
    claimed = store.claim_next("worker-a", 30)
    assert claimed is not None
    assert claimed.job.job_id == ingest.job_id
    assert claimed.job.state is JobState.RUNNING
    assert claimed.job.stage is JobStage.VALIDATE
    assert claimed.lease_owner == "worker-a"
    track = store.create(_track_spec())
    claimed_t = store.claim_next("worker-a", 30)
    assert claimed_t is not None and claimed_t.job.job_id == track.job_id
    assert claimed_t.job.stage is JobStage.TRACK


def test_partial_only_ingest_with_committed_segment(tmp_path: Path) -> None:
    store = SqlJobStore(_engine(tmp_path), lease_seconds=30)
    spec = _ingest_spec()
    store.create(spec)
    claimed = store.claim_next("w", 30)
    assert claimed is not None
    video_id = claimed.payload.video_id  # type: ignore[union-attr]
    ok = store.complete(
        spec.job_id,
        "w",
        JobOutcome(state=JobState.PARTIAL, partial=_partial(video_id)),
    )
    assert ok.state is JobState.PARTIAL
    assert ok.partial is not None

    track = _track_spec()
    store.create(track)
    claimed_t = store.claim_next("w", 30)
    assert claimed_t is not None
    with pytest.raises(InvalidStateTransition, match="PARTIAL"):
        store.complete(
            track.job_id,
            "w",
            JobOutcome(state=JobState.PARTIAL, partial=_partial(claimed_t.job.subject_id)),
        )


def test_refused_only_policy_kinds_with_reasons(tmp_path: Path) -> None:
    store = SqlJobStore(_engine(tmp_path), lease_seconds=30)
    ingest = _ingest_spec()
    store.create(ingest)
    store.claim_next("w", 30)
    with pytest.raises(InvalidStateTransition, match="REFUSED"):
        store.complete(
            ingest.job_id,
            "w",
            JobOutcome(
                state=JobState.REFUSED, refusal_reasons=(ReasonCode.TRACK_NOT_CONFIRMED,)
            ),
        )

    track = _track_spec()
    store.create(track)
    store.claim_next("w", 30)
    with pytest.raises(InvalidStateTransition, match="REFUSED"):
        store.complete(track.job_id, "w", JobOutcome(state=JobState.REFUSED))
    view = store.complete(
        track.job_id,
        "w",
        JobOutcome(
            state=JobState.REFUSED, refusal_reasons=(ReasonCode.TRACK_NOT_CONFIRMED,)
        ),
    )
    assert view.state is JobState.REFUSED
    assert view.refusal_reasons == (ReasonCode.TRACK_NOT_CONFIRMED,)


def test_jobview_create_enforces_state_rules() -> None:
    base = {
        "job_id": new_uuid7(),
        "kind": JobKind.INGEST,
        "case_id": new_uuid7(),
        "subject_id": new_uuid7(),
        "completed_units": 0,
        "total_units": 1,
        "attempt": 1,
        "mode": "FIXTURE",
        "correlation_id": new_uuid7(),
        "updated_at": "2026-10-09T16:00:00.000000Z",
        "created_at": "2026-10-09T16:00:00.000000Z",
    }
    JobView.create(**base, state=JobState.QUEUED)
    with pytest.raises(ValidationError, match="RUNNING jobs carry a stage"):
        JobView.create(**base, state=JobState.RUNNING)
    with pytest.raises(ValidationError, match="PARTIAL requires"):
        JobView.create(**base, state=JobState.PARTIAL, finished_at="2026-10-09T16:00:01.000000Z")
    with pytest.raises(ValidationError, match="FAILED jobs require an error"):
        JobView.create(**base, state=JobState.FAILED, finished_at="2026-10-09T16:00:01.000000Z")


def test_complete_cancelling_keeps_partial(tmp_path: Path) -> None:
    store = SqlJobStore(_engine(tmp_path), lease_seconds=30)
    spec = _ingest_spec()
    store.create(spec)
    claimed = store.claim_next("w", 30)
    assert claimed is not None
    store.request_cancel(spec.job_id)
    video_id = claimed.payload.video_id  # type: ignore[union-attr]
    view = store.complete(
        spec.job_id,
        "w",
        JobOutcome(state=JobState.SUCCEEDED, partial=_partial(video_id)),
    )
    assert view.state is JobState.CANCELLED
    assert view.partial is not None


class _Clock:
    def __init__(self) -> None:
        self.t = datetime(2026, 10, 9, 16, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t = self.t + timedelta(seconds=seconds)


def test_recover_expired_running_is_failed_interrupted(tmp_path: Path) -> None:
    clock = _Clock()
    store = SqlJobStore(_engine(tmp_path), clock=clock, lease_seconds=5)
    store.create(_ingest_spec())
    claimed = store.claim_next("dead", 5)
    assert claimed is not None
    clock.advance(30)
    recovered = store.recover_expired()
    assert len(recovered) == 1
    assert recovered[0].state is JobState.FAILED
    assert recovered[0].error is not None
    assert recovered[0].error.code.value == "WORKER_INTERRUPTED"
    assert recovered[0].error.retryable is True
