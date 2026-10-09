from __future__ import annotations

from probity.domain.enums import ALLOWED_JOB_TRANSITIONS, JobState
from probity.domain.errors import IdempotencyConflict, InvalidStateTransition
from probity.domain.ids import new_uuid7, utc_now
from probity.ports import IdempotencyRecord, IngestPayload, JobSpec
from tests.api.fakes import FakeIdempotencyStore, FakeJobStore


def test_job_store_honors_transitions() -> None:
    store = FakeJobStore()
    spec = JobSpec(
        job_id=new_uuid7(),
        case_id=new_uuid7(),
        subject_id=new_uuid7(),
        payload=IngestPayload(video_id=new_uuid7()),
        correlation_id=new_uuid7(),
    )
    job = store.create(spec)
    assert job.state is JobState.QUEUED
    claimed = store.claim_next("w1", 30)
    assert claimed is not None
    assert claimed.job.state is JobState.RUNNING
    cancelling = store.request_cancel(claimed.job.job_id)
    assert cancelling.state is JobState.CANCELLING
    again = store.request_cancel(claimed.job.job_id)
    assert again.state is JobState.CANCELLING
    for state, allowed in ALLOWED_JOB_TRANSITIONS.items():
        assert JobState.CREATED not in allowed or state is JobState.CREATED


def test_job_store_rejects_illegal_transition() -> None:
    store = FakeJobStore()
    spec = JobSpec(
        job_id=new_uuid7(),
        case_id=new_uuid7(),
        subject_id=new_uuid7(),
        payload=IngestPayload(video_id=new_uuid7()),
        correlation_id=new_uuid7(),
    )
    job = store.create(spec)
    try:
        store._transition(job, JobState.SUCCEEDED)
        raise AssertionError("expected InvalidStateTransition")
    except InvalidStateTransition:
        pass


def test_idempotency_conflict_on_different_hash() -> None:
    store = FakeIdempotencyStore()
    now = utc_now()
    record = IdempotencyRecord(
        route="POST /v1/cases",
        case_id=None,
        key="abc12345",
        request_sha256="a" * 64,
        status_code=201,
        response_body="{}",
        created_at=now,
        expires_at=now,
    )
    store.save(record)
    replay = store.save(record)
    assert replay.request_sha256 == record.request_sha256
    other = record.model_copy(update={"request_sha256": "b" * 64})
    try:
        store.save(other)
        raise AssertionError("expected IdempotencyConflict")
    except IdempotencyConflict:
        pass
