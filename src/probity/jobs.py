"""Job state machine and SQLite ``JobStore`` (section 7)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from pydantic import TypeAdapter, ValidationError
from sqlalchemy import Engine, func, insert, select, update
from sqlalchemy.engine import Connection, Row

from probity.domain.enums import (
    ALLOWED_JOB_TRANSITIONS,
    TERMINAL_JOB_STATES,
    ErrorCode,
    InferenceMode,
    JobKind,
    JobStage,
    JobState,
)
from probity.domain.errors import InvalidStateTransition, NotFound, ValidationFailed
from probity.domain.ids import canonical_sha256, format_utc, new_uuid7
from probity.domain.models import JobError, JobEvent, JobView, PartialCoverage
from probity.persistence.audit import append_audit
from probity.persistence.db import (
    Clock,
    read_transaction,
    to_epoch_us,
    utc_clock,
    write_transaction,
)
from probity.persistence.schema import job_events, jobs
from probity.ports import ClaimedJob, JobOutcome, JobPayload, JobSpec

PAYLOAD_ADAPTER: TypeAdapter[JobPayload] = TypeAdapter(JobPayload)

INITIAL_STAGE: dict[JobKind, JobStage] = {
    JobKind.INGEST: JobStage.VALIDATE,
    JobKind.TRACK: JobStage.TRACK,
    JobKind.RECONSTRUCT: JobStage.ALIGN,
    JobKind.REPORT: JobStage.REPORT,
}

WORKER_INTERRUPTED = JobError(
    code=ErrorCode.WORKER_INTERRUPTED,
    message=(
        "Worker lease expired; the job was interrupted. Retry with the original idempotent input."
    ),
    retryable=True,
)

POLICY_KINDS = frozenset({JobKind.TRACK, JobKind.RECONSTRUCT, JobKind.REPORT})
_KEEP = object()


def validate_transition(from_state: JobState, to_state: JobState) -> None:
    """Raise ``InvalidStateTransition`` unless ``(from_state, to_state)`` is in the frozen table."""
    allowed = ALLOWED_JOB_TRANSITIONS[from_state]
    if to_state not in allowed:
        raise InvalidStateTransition(
            f"job transition {from_state.value} -> {to_state.value} is not allowed",
            details={"from_state": from_state.value, "to_state": to_state.value},
        )


def _dump_view(view: JobView) -> str:
    return view.model_dump_json()


def _load_view(raw: str) -> JobView:
    return JobView.model_validate_json(raw)


def _load_payload(raw: str) -> JobPayload:
    return PAYLOAD_ADAPTER.validate_json(raw)


def _rebuild(view: JobView, **changes: Any) -> JobView:
    data = view.model_dump(mode="python", exclude={"content_sha256"})
    data.update(changes)
    try:
        return JobView.create(**data)
    except ValidationError as exc:
        raise ValidationFailed(str(exc)) from exc


class SqlJobStore:
    """SQLite ``JobStore``. Claims use ``BEGIN IMMEDIATE`` so two workers never share a job."""

    def __init__(
        self,
        engine: Engine,
        clock: Clock = utc_clock,
        lease_seconds: int = 30,
        *,
        actor: str = "system",
        mode: InferenceMode = InferenceMode.FIXTURE,
    ) -> None:
        self._engine = engine
        self._clock = clock
        self._lease_seconds = lease_seconds
        self._actor = actor
        self._mode = mode

    def create(self, spec: JobSpec) -> JobView:
        now = format_utc(self._clock())
        kind = spec.payload.kind
        created = JobView.create(
            job_id=spec.job_id,
            kind=kind,
            case_id=spec.case_id,
            subject_id=spec.subject_id,
            state=JobState.CREATED,
            stage=None,
            completed_units=0,
            total_units=spec.total_units,
            attempt=1,
            error=None,
            refusal_reasons=(),
            partial=None,
            mode=self._mode,
            correlation_id=spec.correlation_id,
            updated_at=now,
            started_at=None,
            finished_at=None,
            cancel_requested_at=None,
            created_at=now,
        )
        payload_json = spec.payload.model_dump_json()
        spec_sha = canonical_sha256(spec.model_dump(mode="json"))
        with write_transaction(self._engine) as conn:
            next_seq = conn.execute(select(func.max(jobs.c.queue_seq))).scalar()
            queue_seq = 1 if next_seq is None else int(next_seq) + 1
            conn.execute(
                insert(jobs).values(
                    job_id=created.job_id,
                    queue_seq=queue_seq,
                    kind=kind.value,
                    case_id=created.case_id,
                    subject_id=created.subject_id,
                    correlation_id=created.correlation_id,
                    state=created.state.value,
                    stage=None,
                    attempt=created.attempt,
                    spec_sha256=spec_sha,
                    payload=payload_json,
                    view=_dump_view(created),
                    content_sha256=created.content_sha256,
                    lease_owner=None,
                    lease_expires_us=None,
                    created_at=created.created_at,
                    updated_at=created.updated_at,
                )
            )
            self._record_event(
                conn,
                created,
                from_state=None,
                to_state=JobState.CREATED,
                actor=self._actor,
            )
            queued = self._apply(
                conn,
                created,
                JobState.QUEUED,
                actor=self._actor,
                updated_at=now,
            )
        return queued

    def get(self, job_id: str) -> JobView:
        with read_transaction(self._engine) as conn:
            row = conn.execute(select(jobs).where(jobs.c.job_id == job_id)).first()
        if row is None:
            raise NotFound(f"job {job_id} was not found")
        return _load_view(row.view)

    def events(self, job_id: str) -> Sequence[JobEvent]:
        with read_transaction(self._engine) as conn:
            rows = conn.execute(
                select(job_events)
                .where(job_events.c.job_id == job_id)
                .order_by(job_events.c.sequence)
            ).all()
        return [JobEvent.model_validate_json(r.payload) for r in rows]

    def request_cancel(self, job_id: str) -> JobView:
        now = format_utc(self._clock())
        with write_transaction(self._engine) as conn:
            view, _row = self._load(conn, job_id)
            if view.state in TERMINAL_JOB_STATES or view.state is JobState.CANCELLING:
                return view
            if view.state is JobState.QUEUED:
                cancelling = self._apply(
                    conn,
                    view,
                    JobState.CANCELLING,
                    actor=self._actor,
                    cancel_requested_at=now,
                    updated_at=now,
                )
                return self._apply(
                    conn,
                    cancelling,
                    JobState.CANCELLED,
                    actor=self._actor,
                    finished_at=now,
                    updated_at=now,
                    lease_owner=None,
                    lease_expires_us=None,
                )
            if view.state is JobState.RUNNING:
                return self._apply(
                    conn,
                    view,
                    JobState.CANCELLING,
                    actor=self._actor,
                    cancel_requested_at=now,
                    updated_at=now,
                )
            raise InvalidStateTransition(
                f"job {job_id} cannot be cancelled from {view.state.value}"
            )

    def claim_next(self, owner: str, lease_seconds: int) -> ClaimedJob | None:
        now = self._clock()
        now_str = format_utc(now)
        expires_us = to_epoch_us(now + timedelta(seconds=lease_seconds))
        with write_transaction(self._engine) as conn:
            row = conn.execute(
                select(jobs)
                .where(jobs.c.state == JobState.QUEUED.value)
                .order_by(jobs.c.created_at, jobs.c.queue_seq)
                .limit(1)
            ).first()
            if row is None:
                return None
            view = _load_view(row.view)
            stage = INITIAL_STAGE[view.kind]
            running = self._apply(
                conn,
                view,
                JobState.RUNNING,
                actor=owner,
                stage=stage,
                started_at=view.started_at or now_str,
                updated_at=now_str,
                lease_owner=owner,
                lease_expires_us=expires_us,
            )
            payload = _load_payload(row.payload)
        return ClaimedJob(
            job=running,
            payload=payload,
            lease_owner=owner,
            lease_expires_at=format_utc(now + timedelta(seconds=lease_seconds)),
        )

    def renew_lease(self, job_id: str, owner: str, lease_seconds: int) -> bool:
        now = self._clock()
        expires_us = to_epoch_us(now + timedelta(seconds=lease_seconds))
        with write_transaction(self._engine) as conn:
            row = conn.execute(select(jobs).where(jobs.c.job_id == job_id)).first()
            if row is None:
                return False
            view = _load_view(row.view)
            if row.lease_owner != owner or view.state not in {
                JobState.RUNNING,
                JobState.CANCELLING,
            }:
                return False
            conn.execute(
                update(jobs)
                .where(jobs.c.job_id == job_id)
                .values(lease_expires_us=expires_us, updated_at=format_utc(now))
            )
        return True

    def report_progress(
        self, job_id: str, owner: str, stage: JobStage, completed: int, total: int
    ) -> JobView:
        now = format_utc(self._clock())
        with write_transaction(self._engine) as conn:
            view, row = self._load(conn, job_id)
            if row.lease_owner != owner or view.state not in {
                JobState.RUNNING,
                JobState.CANCELLING,
            }:
                raise InvalidStateTransition(
                    f"progress rejected for job {job_id}: owner or state mismatch"
                )
            updated = _rebuild(
                view, stage=stage, completed_units=completed, total_units=total, updated_at=now
            )
            self._write_view(
                conn,
                updated,
                lease_owner=row.lease_owner,
                lease_expires_us=row.lease_expires_us,
            )
            append_audit(
                conn,
                case_id=updated.case_id,
                occurred_at=now,
                actor=owner,
                action="job.progress",
                entity_type="job",
                entity_id=job_id,
                correlation_id=updated.correlation_id,
                params={
                    "stage": stage.value,
                    "completed": completed,
                    "total": total,
                },
            )
        return updated

    def is_cancel_requested(self, job_id: str) -> bool:
        return self.get(job_id).state is JobState.CANCELLING

    def complete(self, job_id: str, owner: str, outcome: JobOutcome) -> JobView:
        now = format_utc(self._clock())
        with write_transaction(self._engine) as conn:
            view, row = self._load(conn, job_id)
            if row.lease_owner != owner:
                raise InvalidStateTransition(f"complete rejected: {owner} does not own {job_id}")
            if view.state is JobState.CANCELLING:
                return self._apply(
                    conn,
                    view,
                    JobState.CANCELLED,
                    actor=owner,
                    partial=outcome.partial or view.partial,
                    finished_at=now,
                    updated_at=now,
                    lease_owner=None,
                    lease_expires_us=None,
                )
            if view.state is not JobState.RUNNING:
                raise InvalidStateTransition(
                    f"complete rejected: job {job_id} is {view.state.value}"
                )
            self._validate_outcome(view.kind, outcome)
            return self._apply(
                conn,
                view,
                outcome.state,
                actor=owner,
                error=outcome.error,
                refusal_reasons=outcome.refusal_reasons,
                partial=outcome.partial,
                finished_at=now,
                updated_at=now,
                lease_owner=None,
                lease_expires_us=None,
            )

    def mark_cancelled(
        self, job_id: str, owner: str, partial: PartialCoverage | None
    ) -> JobView:
        now = format_utc(self._clock())
        with write_transaction(self._engine) as conn:
            view, row = self._load(conn, job_id)
            if view.state is JobState.CANCELLED:
                return view
            if view.state is not JobState.CANCELLING:
                raise InvalidStateTransition(
                    f"mark_cancelled rejected: job {job_id} is {view.state.value}"
                )
            if row.lease_owner not in {owner, None}:
                raise InvalidStateTransition(
                    f"mark_cancelled rejected: {owner} does not own {job_id}"
                )
            return self._apply(
                conn,
                view,
                JobState.CANCELLED,
                actor=owner,
                partial=partial or view.partial,
                finished_at=now,
                updated_at=now,
                lease_owner=None,
                lease_expires_us=None,
            )

    def recover_expired(self) -> Sequence[JobView]:
        now = self._clock()
        now_str = format_utc(now)
        now_us = to_epoch_us(now)
        recovered: list[JobView] = []
        with write_transaction(self._engine) as conn:
            rows = conn.execute(
                select(jobs).where(
                    jobs.c.state.in_((JobState.RUNNING.value, JobState.CANCELLING.value)),
                    jobs.c.lease_expires_us.is_not(None),
                    jobs.c.lease_expires_us < now_us,
                )
            ).all()
            for row in rows:
                view = _load_view(row.view)
                if view.state is JobState.RUNNING:
                    recovered.append(
                        self._apply(
                            conn,
                            view,
                            JobState.FAILED,
                            actor=self._actor,
                            error=WORKER_INTERRUPTED,
                            finished_at=now_str,
                            updated_at=now_str,
                            lease_owner=None,
                            lease_expires_us=None,
                            reason_code=ErrorCode.WORKER_INTERRUPTED.value,
                        )
                    )
                else:
                    recovered.append(
                        self._apply(
                            conn,
                            view,
                            JobState.CANCELLED,
                            actor=self._actor,
                            finished_at=now_str,
                            updated_at=now_str,
                            lease_owner=None,
                            lease_expires_us=None,
                        )
                    )
        return recovered

    def _validate_outcome(self, kind: JobKind, outcome: JobOutcome) -> None:
        if outcome.state is JobState.PARTIAL:
            if kind is not JobKind.INGEST or outcome.partial is None:
                raise InvalidStateTransition(
                    "PARTIAL is valid only for INGEST jobs with at least one committed segment"
                )
        if outcome.state is JobState.REFUSED:
            if kind not in POLICY_KINDS or not outcome.refusal_reasons:
                raise InvalidStateTransition(
                    "REFUSED is valid only for TRACK/RECONSTRUCT/REPORT with refusal_reasons"
                )
        if outcome.state is JobState.FAILED and outcome.error is None:
            raise InvalidStateTransition("FAILED jobs require an error")
        if outcome.state not in {
            JobState.SUCCEEDED,
            JobState.PARTIAL,
            JobState.REFUSED,
            JobState.FAILED,
        }:
            raise InvalidStateTransition(f"complete cannot move to {outcome.state.value}")

    def _load(self, conn: Connection, job_id: str) -> tuple[JobView, Row[Any]]:
        row = conn.execute(select(jobs).where(jobs.c.job_id == job_id)).first()
        if row is None:
            raise NotFound(f"job {job_id} was not found")
        return _load_view(row.view), row

    def _apply(
        self,
        conn: Connection,
        view: JobView,
        to_state: JobState,
        *,
        actor: str,
        reason_code: str | None = None,
        lease_owner: str | None | object = _KEEP,
        lease_expires_us: int | None | object = _KEEP,
        **changes: Any,
    ) -> JobView:
        validate_transition(view.state, to_state)
        updated = _rebuild(view, state=to_state, **changes)
        current = conn.execute(select(jobs).where(jobs.c.job_id == view.job_id)).one()
        owner = current.lease_owner if lease_owner is _KEEP else lease_owner
        expires = current.lease_expires_us if lease_expires_us is _KEEP else lease_expires_us
        self._write_view(conn, updated, lease_owner=owner, lease_expires_us=expires)
        self._record_event(conn, updated, from_state=view.state, to_state=to_state, actor=actor)
        append_audit(
            conn,
            case_id=updated.case_id,
            occurred_at=updated.updated_at,
            actor=actor[:128],
            action="job.transition",
            entity_type="job",
            entity_id=updated.job_id,
            correlation_id=updated.correlation_id,
            reason_code=reason_code,
            params={
                "from_state": view.state.value,
                "to_state": to_state.value,
                "stage": None if updated.stage is None else updated.stage.value,
            },
        )
        return updated

    def _write_view(
        self,
        conn: Connection,
        view: JobView,
        *,
        lease_owner: str | None,
        lease_expires_us: int | None,
    ) -> None:
        conn.execute(
            update(jobs)
            .where(jobs.c.job_id == view.job_id)
            .values(
                state=view.state.value,
                stage=None if view.stage is None else view.stage.value,
                attempt=view.attempt,
                view=_dump_view(view),
                content_sha256=view.content_sha256,
                lease_owner=lease_owner,
                lease_expires_us=lease_expires_us,
                updated_at=view.updated_at,
            )
        )

    def _record_event(
        self,
        conn: Connection,
        view: JobView,
        *,
        from_state: JobState | None,
        to_state: JobState,
        actor: str,
    ) -> None:
        next_seq = conn.execute(
            select(func.max(job_events.c.sequence)).where(job_events.c.job_id == view.job_id)
        ).scalar()
        sequence = 0 if next_seq is None else int(next_seq) + 1
        event = JobEvent.create(
            event_id=new_uuid7(),
            job_id=view.job_id,
            sequence=sequence,
            from_state=from_state,
            to_state=to_state,
            stage=view.stage,
            completed_units=view.completed_units,
            total_units=view.total_units,
            error_code=None if view.error is None else view.error.code,
            reason_codes=view.refusal_reasons,
            actor=actor[:64],
            correlation_id=view.correlation_id,
            created_at=view.updated_at,
        )
        conn.execute(
            insert(job_events).values(
                event_id=event.event_id,
                job_id=event.job_id,
                sequence=event.sequence,
                from_state=None if from_state is None else from_state.value,
                to_state=to_state.value,
                created_at=event.created_at,
                content_sha256=event.content_sha256,
                payload=event.model_dump_json(),
            )
        )
