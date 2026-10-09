"""SQLAlchemy Core table definitions mirroring migration ``0001_initial``.

Evidence-bearing tables are revisioned: one row per ``(entity_id, revision)``; the latest revision
is the current value. Each row stores the record's canonical JSON ``payload`` plus the indexed
scalar columns used for lookups. ``tests/unit/test_leases.py`` asserts this metadata matches the
migrated schema exactly.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)

metadata = MetaData()


def _revisioned(name: str, *columns: Column[object], indexes: tuple[str, ...] = ()) -> Table:
    table = Table(
        name,
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("entity_id", String(200), nullable=False),
        Column("revision", Integer, nullable=False),
        *columns,
        Column("content_sha256", String(64), nullable=False),
        Column("created_at", String(32), nullable=False),
        Column("recorded_at", String(32), nullable=False),
        Column("payload", Text, nullable=False),
        UniqueConstraint("entity_id", "revision", name=f"uq_{name}_entity_revision"),
    )
    for col in indexes:
        Index(f"ix_{name}_{col}", table.c[col])
    return table


cases = _revisioned("cases", Column("case_id", String(36), nullable=False), indexes=("case_id",))

source_videos = _revisioned(
    "source_videos",
    Column("video_id", String(36), nullable=False),
    Column("case_id", String(36), nullable=False),
    Column("sha256", String(64), nullable=False),
    Column("ingest_state", String(16), nullable=False),
    indexes=("case_id", "sha256"),
)

frame_manifests = _revisioned(
    "frame_manifests",
    Column("video_id", String(36), nullable=False),
    Column("source_sha256", String(64), nullable=False),
    indexes=("video_id",),
)

segments = _revisioned(
    "segments",
    Column("segment_id", String(48), nullable=False),
    Column("video_id", String(36), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("index_state", String(16), nullable=False),
    indexes=("video_id",),
)

embeddings = _revisioned(
    "embeddings",
    Column("video_id", String(36), nullable=False),
    Column("embedding_ref", String(200), nullable=False),
    Column("model_id", String(200), nullable=False),
    Column("dim", Integer, nullable=False),
    Column("vector", LargeBinary, nullable=False),
    Column("vector_dtype", String(8), nullable=False),
    indexes=("video_id",),
)

detections = _revisioned(
    "detections",
    Column("detection_id", String(36), nullable=False),
    Column("video_id", String(36), nullable=False),
    Column("frame_id", String(48), nullable=False),
    Column("pts_us", BigInteger, nullable=False),
    indexes=("video_id", "frame_id"),
)

searches = _revisioned(
    "searches",
    Column("search_id", String(36), nullable=False),
    Column("case_id", String(36), nullable=False),
    Column("video_id", String(36), nullable=False),
    indexes=("case_id", "video_id"),
)

tracks = _revisioned(
    "tracks",
    Column("track_id", String(36), nullable=False),
    Column("case_id", String(36), nullable=False),
    Column("video_id", String(36), nullable=False),
    Column("state", String(16), nullable=False),
    indexes=("case_id", "video_id"),
)

reconstruction_runs = _revisioned(
    "reconstruction_runs",
    Column("run_id", String(36), nullable=False),
    Column("case_id", String(36), nullable=False),
    Column("video_id", String(36), nullable=False),
    Column("track_id", String(36), nullable=False),
    Column("state", String(16), nullable=False),
    indexes=("case_id", "track_id"),
)

policy_decisions = _revisioned(
    "policy_decisions",
    Column("decision_id", String(36), nullable=False),
    Column("run_id", String(36), nullable=False),
    Column("sequence", Integer, nullable=False),
    indexes=("run_id",),
)

pixel_provenance = _revisioned(
    "pixel_provenance",
    Column("run_id", String(36), nullable=False),
    Column("artifact_sha256", String(64), nullable=False),
    indexes=("run_id",),
)

human_reviews = _revisioned(
    "human_reviews",
    Column("review_id", String(36), nullable=False),
    Column("run_id", String(36), nullable=False),
    Column("decision", String(16), nullable=False),
    indexes=("run_id",),
)

evidence_reports = _revisioned(
    "evidence_reports",
    Column("report_id", String(36), nullable=False),
    Column("case_id", String(36), nullable=False),
    Column("run_id", String(36), nullable=False),
    indexes=("case_id", "run_id"),
)

assets = _revisioned(
    "assets",
    Column("asset_id", String(36), nullable=False),
    Column("case_id", String(36), nullable=False),
    Column("kind", String(32), nullable=False),
    Column("sha256", String(64), nullable=False),
    indexes=("case_id",),
)

lineage = _revisioned(
    "lineage",
    Column("lineage_id", String(36), nullable=False),
    Column("video_id", String(36), nullable=False),
    Column("kind", String(16), nullable=False),
    Column("subject_entity_id", String(120), nullable=False),
    Column("artifact_sha256", String(64), nullable=False),
    indexes=("video_id", "subject_entity_id"),
)

jobs = Table(
    "jobs",
    metadata,
    Column("job_id", String(36), primary_key=True),
    Column("queue_seq", Integer, nullable=False, unique=True),
    Column("kind", String(16), nullable=False),
    Column("case_id", String(36), nullable=False),
    Column("subject_id", String(36), nullable=False),
    Column("correlation_id", String(36), nullable=False),
    Column("state", String(16), nullable=False),
    Column("stage", String(16), nullable=True),
    Column("attempt", Integer, nullable=False),
    Column("spec_sha256", String(64), nullable=False),
    Column("payload", Text, nullable=False),
    Column("view", Text, nullable=False),
    Column("content_sha256", String(64), nullable=False),
    Column("lease_owner", String(128), nullable=True),
    Column("lease_expires_us", BigInteger, nullable=True),
    Column("created_at", String(32), nullable=False),
    Column("updated_at", String(32), nullable=False),
)
Index("ix_jobs_state_queue", jobs.c.state, jobs.c.queue_seq)
Index("ix_jobs_state_lease", jobs.c.state, jobs.c.lease_expires_us)
Index("ix_jobs_case_id", jobs.c.case_id)
Index("ix_jobs_subject_id", jobs.c.subject_id)

job_events = Table(
    "job_events",
    metadata,
    Column("event_id", String(36), primary_key=True),
    Column("job_id", String(36), ForeignKey("jobs.job_id"), nullable=False),
    Column("sequence", Integer, nullable=False),
    Column("from_state", String(16), nullable=True),
    Column("to_state", String(16), nullable=False),
    Column("created_at", String(32), nullable=False),
    Column("content_sha256", String(64), nullable=False),
    Column("payload", Text, nullable=False),
    UniqueConstraint("job_id", "sequence", name="uq_job_events_job_sequence"),
)

idempotency_records = Table(
    "idempotency_records",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("route", String(200), nullable=False),
    Column("case_scope", String(36), nullable=False),
    Column("idem_key", String(128), nullable=False),
    Column("request_sha256", String(64), nullable=False),
    Column("status_code", Integer, nullable=False),
    Column("response_body", Text, nullable=False),
    Column("created_at", String(32), nullable=False),
    Column("expires_at", String(32), nullable=False),
    Column("expires_us", BigInteger, nullable=False),
    UniqueConstraint("route", "case_scope", "idem_key", name="uq_idempotency_scope_key"),
)
Index("ix_idempotency_expires_us", idempotency_records.c.expires_us)

audit_log = Table(
    "audit_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("chain_key", String(36), nullable=False),
    Column("sequence", Integer, nullable=False),
    Column("occurred_at", String(32), nullable=False),
    Column("actor", String(128), nullable=False),
    Column("action", String(64), nullable=False),
    Column("entity_type", String(32), nullable=False),
    Column("entity_id", String(200), nullable=False),
    Column("correlation_id", String(36), nullable=True),
    Column("reason_code", String(64), nullable=True),
    Column("params", Text, nullable=False),
    Column("previous_row_hash", String(64), nullable=False),
    Column("row_hash", String(64), nullable=False),
    UniqueConstraint("chain_key", "sequence", name="uq_audit_chain_sequence"),
)
Index("ix_audit_entity", audit_log.c.entity_type, audit_log.c.entity_id)

REVISIONED_TABLES: tuple[Table, ...] = (
    cases,
    source_videos,
    frame_manifests,
    segments,
    embeddings,
    detections,
    searches,
    tracks,
    reconstruction_runs,
    policy_decisions,
    pixel_provenance,
    human_reviews,
    evidence_reports,
    assets,
    lineage,
)

APPEND_ONLY_TABLES: tuple[str, ...] = ("audit_log", "job_events")
