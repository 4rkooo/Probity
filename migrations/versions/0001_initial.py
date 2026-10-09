"""Initial Probity schema: revisioned evidence records, jobs, idempotency, audit log.

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-09
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_REVISIONED: list[str] = []


def _revisioned(name: str, *columns: sa.Column, indexes: tuple[str, ...] = ()) -> None:
    op.create_table(
        name,
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("entity_id", sa.String(200), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        *columns,
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.String(32), nullable=False),
        sa.Column("recorded_at", sa.String(32), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.UniqueConstraint("entity_id", "revision", name=f"uq_{name}_entity_revision"),
    )
    for col in indexes:
        op.create_index(f"ix_{name}_{col}", name, [col])
    _REVISIONED.append(name)


def _no_update(table: str) -> None:
    op.execute(
        f"CREATE TRIGGER trg_{table}_no_update BEFORE UPDATE ON {table} "
        f"BEGIN SELECT RAISE(ABORT, '{table} rows are immutable; insert a new revision'); END"
    )


def _no_delete(table: str) -> None:
    op.execute(
        f"CREATE TRIGGER trg_{table}_no_delete BEFORE DELETE ON {table} "
        f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END"
    )


def upgrade() -> None:
    _REVISIONED.clear()
    s = sa.String
    _revisioned("cases", sa.Column("case_id", s(36), nullable=False), indexes=("case_id",))
    _revisioned(
        "source_videos",
        sa.Column("video_id", s(36), nullable=False),
        sa.Column("case_id", s(36), nullable=False),
        sa.Column("sha256", s(64), nullable=False),
        sa.Column("ingest_state", s(16), nullable=False),
        indexes=("case_id", "sha256"),
    )
    _revisioned(
        "frame_manifests",
        sa.Column("video_id", s(36), nullable=False),
        sa.Column("source_sha256", s(64), nullable=False),
        indexes=("video_id",),
    )
    _revisioned(
        "segments",
        sa.Column("segment_id", s(48), nullable=False),
        sa.Column("video_id", s(36), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("index_state", s(16), nullable=False),
        indexes=("video_id",),
    )
    _revisioned(
        "embeddings",
        sa.Column("video_id", s(36), nullable=False),
        sa.Column("embedding_ref", s(200), nullable=False),
        sa.Column("model_id", s(200), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
        sa.Column("vector_dtype", s(8), nullable=False),
        indexes=("video_id",),
    )
    _revisioned(
        "detections",
        sa.Column("detection_id", s(36), nullable=False),
        sa.Column("video_id", s(36), nullable=False),
        sa.Column("frame_id", s(48), nullable=False),
        sa.Column("pts_us", sa.BigInteger(), nullable=False),
        indexes=("video_id", "frame_id"),
    )
    _revisioned(
        "searches",
        sa.Column("search_id", s(36), nullable=False),
        sa.Column("case_id", s(36), nullable=False),
        sa.Column("video_id", s(36), nullable=False),
        indexes=("case_id", "video_id"),
    )
    _revisioned(
        "tracks",
        sa.Column("track_id", s(36), nullable=False),
        sa.Column("case_id", s(36), nullable=False),
        sa.Column("video_id", s(36), nullable=False),
        sa.Column("state", s(16), nullable=False),
        indexes=("case_id", "video_id"),
    )
    _revisioned(
        "reconstruction_runs",
        sa.Column("run_id", s(36), nullable=False),
        sa.Column("case_id", s(36), nullable=False),
        sa.Column("video_id", s(36), nullable=False),
        sa.Column("track_id", s(36), nullable=False),
        sa.Column("state", s(16), nullable=False),
        indexes=("case_id", "track_id"),
    )
    _revisioned(
        "policy_decisions",
        sa.Column("decision_id", s(36), nullable=False),
        sa.Column("run_id", s(36), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        indexes=("run_id",),
    )
    _revisioned(
        "pixel_provenance",
        sa.Column("run_id", s(36), nullable=False),
        sa.Column("artifact_sha256", s(64), nullable=False),
        indexes=("run_id",),
    )
    _revisioned(
        "human_reviews",
        sa.Column("review_id", s(36), nullable=False),
        sa.Column("run_id", s(36), nullable=False),
        sa.Column("decision", s(16), nullable=False),
        indexes=("run_id",),
    )
    _revisioned(
        "evidence_reports",
        sa.Column("report_id", s(36), nullable=False),
        sa.Column("case_id", s(36), nullable=False),
        sa.Column("run_id", s(36), nullable=False),
        indexes=("case_id", "run_id"),
    )
    _revisioned(
        "assets",
        sa.Column("asset_id", s(36), nullable=False),
        sa.Column("case_id", s(36), nullable=False),
        sa.Column("kind", s(32), nullable=False),
        sa.Column("sha256", s(64), nullable=False),
        indexes=("case_id",),
    )
    _revisioned(
        "lineage",
        sa.Column("lineage_id", s(36), nullable=False),
        sa.Column("video_id", s(36), nullable=False),
        sa.Column("kind", s(16), nullable=False),
        sa.Column("subject_entity_id", s(120), nullable=False),
        sa.Column("artifact_sha256", s(64), nullable=False),
        indexes=("video_id", "subject_entity_id"),
    )

    op.create_table(
        "jobs",
        sa.Column("job_id", s(36), primary_key=True),
        sa.Column("queue_seq", sa.Integer(), nullable=False, unique=True),
        sa.Column("kind", s(16), nullable=False),
        sa.Column("case_id", s(36), nullable=False),
        sa.Column("subject_id", s(36), nullable=False),
        sa.Column("correlation_id", s(36), nullable=False),
        sa.Column("state", s(16), nullable=False),
        sa.Column("stage", s(16), nullable=True),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("spec_sha256", s(64), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("view", sa.Text(), nullable=False),
        sa.Column("content_sha256", s(64), nullable=False),
        sa.Column("lease_owner", s(128), nullable=True),
        sa.Column("lease_expires_us", sa.BigInteger(), nullable=True),
        sa.Column("created_at", s(32), nullable=False),
        sa.Column("updated_at", s(32), nullable=False),
    )
    op.create_index("ix_jobs_state_queue", "jobs", ["state", "queue_seq"])
    op.create_index("ix_jobs_state_lease", "jobs", ["state", "lease_expires_us"])
    op.create_index("ix_jobs_case_id", "jobs", ["case_id"])
    op.create_index("ix_jobs_subject_id", "jobs", ["subject_id"])

    op.create_table(
        "job_events",
        sa.Column("event_id", s(36), primary_key=True),
        sa.Column("job_id", s(36), sa.ForeignKey("jobs.job_id"), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("from_state", s(16), nullable=True),
        sa.Column("to_state", s(16), nullable=False),
        sa.Column("created_at", s(32), nullable=False),
        sa.Column("content_sha256", s(64), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.UniqueConstraint("job_id", "sequence", name="uq_job_events_job_sequence"),
    )

    op.create_table(
        "idempotency_records",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("route", s(200), nullable=False),
        sa.Column("case_scope", s(36), nullable=False),
        sa.Column("idem_key", s(128), nullable=False),
        sa.Column("request_sha256", s(64), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("response_body", sa.Text(), nullable=False),
        sa.Column("created_at", s(32), nullable=False),
        sa.Column("expires_at", s(32), nullable=False),
        sa.Column("expires_us", sa.BigInteger(), nullable=False),
        sa.UniqueConstraint("route", "case_scope", "idem_key", name="uq_idempotency_scope_key"),
    )
    op.create_index("ix_idempotency_expires_us", "idempotency_records", ["expires_us"])

    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("chain_key", s(36), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("occurred_at", s(32), nullable=False),
        sa.Column("actor", s(128), nullable=False),
        sa.Column("action", s(64), nullable=False),
        sa.Column("entity_type", s(32), nullable=False),
        sa.Column("entity_id", s(200), nullable=False),
        sa.Column("correlation_id", s(36), nullable=True),
        sa.Column("reason_code", s(64), nullable=True),
        sa.Column("params", sa.Text(), nullable=False),
        sa.Column("previous_row_hash", s(64), nullable=False),
        sa.Column("row_hash", s(64), nullable=False),
        sa.UniqueConstraint("chain_key", "sequence", name="uq_audit_chain_sequence"),
    )
    op.create_index("ix_audit_entity", "audit_log", ["entity_type", "entity_id"])

    for table in _REVISIONED:
        _no_update(table)
    for table in ("audit_log", "job_events"):
        _no_update(table)
        _no_delete(table)


def downgrade() -> None:
    for table in (
        "audit_log",
        "idempotency_records",
        "job_events",
        "jobs",
        "lineage",
        "assets",
        "evidence_reports",
        "human_reviews",
        "pixel_provenance",
        "policy_decisions",
        "reconstruction_runs",
        "tracks",
        "searches",
        "detections",
        "embeddings",
        "segments",
        "frame_manifests",
        "source_videos",
        "cases",
    ):
        op.drop_table(table)
