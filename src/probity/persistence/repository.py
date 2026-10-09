"""SQLAlchemy implementation of ``ports.Repository``.

Evidence tables are insert-only and revisioned. ``get_*`` returns the highest revision of an
entity, re-validating the frozen Pydantic model. An identical ``content_sha256`` re-put is a
no-op; a different hash inserts a new revision.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, TypeVar

import numpy as np
from sqlalchemy import Engine, Table, func, insert, select
from sqlalchemy.engine import Connection, Row

from probity.domain.errors import NotFound
from probity.domain.ids import canonical_sha256, format_utc, sha256_hex
from probity.domain.models import (
    AssetRef,
    CaseWorkspace,
    Detection,
    Embedding,
    EvidenceReport,
    FrameManifest,
    FrozenModel,
    HumanReview,
    LineageRecord,
    PixelProvenance,
    PolicyDecision,
    ReconstructionRun,
    Record,
    SearchEvidence,
    SourceVideo,
    Track,
    VideoSegment,
)
from probity.persistence.audit import append_audit
from probity.persistence.db import read_transaction, utc_clock, write_transaction
from probity.persistence.schema import (
    assets,
    cases,
    detections,
    embeddings,
    evidence_reports,
    frame_manifests,
    human_reviews,
    lineage,
    pixel_provenance,
    policy_decisions,
    reconstruction_runs,
    searches,
    segments,
    source_videos,
    tracks,
)

T = TypeVar("T", bound=FrozenModel)
Clock = Callable[[], datetime]
VECTOR_DTYPE = np.dtype("<f8")


def _now(clock: Clock) -> str:
    return format_utc(clock())


def _dump(model: FrozenModel) -> str:
    return model.model_dump_json()


def _content_sha(model: FrozenModel) -> str:
    if isinstance(model, Record):
        return model.content_sha256
    return canonical_sha256(model.model_dump(mode="json"))


def _created_at(model: FrozenModel) -> str:
    value = getattr(model, "created_at", None)
    if isinstance(value, str):
        return value
    return ""


def _latest(conn: Connection, table: Table, entity_id: str) -> Row[Any] | None:
    return conn.execute(
        select(table)
        .where(table.c.entity_id == entity_id)
        .order_by(table.c.revision.desc())
        .limit(1)
    ).first()


def _latest_by(
    conn: Connection, table: Table, column: str, value: str, *, order: str | None = None
) -> Sequence[Row[Any]]:
    ranked = (
        select(table.c.entity_id, func.max(table.c.revision).label("rev"))
        .where(table.c[column] == value)
        .group_by(table.c.entity_id)
        .subquery()
    )
    stmt = select(table).join(
        ranked,
        (table.c.entity_id == ranked.c.entity_id) & (table.c.revision == ranked.c.rev),
    )
    if order is not None:
        stmt = stmt.order_by(table.c[order])
    return conn.execute(stmt).all()


def _put_revision(
    conn: Connection,
    table: Table,
    *,
    entity_id: str,
    content_sha256: str,
    created_at: str,
    recorded_at: str,
    payload: str,
    scalars: Mapping[str, Any],
) -> bool:
    existing = conn.execute(
        select(table.c.content_sha256, table.c.revision)
        .where(table.c.entity_id == entity_id)
        .order_by(table.c.revision.desc())
        .limit(1)
    ).first()
    if existing is not None and existing.content_sha256 == content_sha256:
        return False
    revision = 0 if existing is None else int(existing.revision) + 1
    conn.execute(
        insert(table).values(
            entity_id=entity_id,
            revision=revision,
            content_sha256=content_sha256,
            created_at=created_at,
            recorded_at=recorded_at,
            payload=payload,
            **scalars,
        )
    )
    return True


def _require(row: Row[Any] | None, what: str, ident: str) -> Row[Any]:
    if row is None:
        raise NotFound(f"{what} {ident} was not found")
    return row


def _load_record(model: type[T], payload: str) -> T:
    return model.model_validate_json(payload)


def _embedding_sha(emb: Embedding, blob: bytes) -> str:
    meta = {k: v for k, v in emb.model_dump(mode="json").items() if k != "vector"}
    return sha256_hex(canonical_sha256(meta).encode("ascii") + b":" + blob)


def _vector_blob(emb: Embedding) -> bytes:
    return np.asarray(emb.vector, dtype=VECTOR_DTYPE).tobytes()


def _load_embedding(row: Row[Any]) -> Embedding:
    data = json.loads(row.payload)
    vector = np.frombuffer(row.vector, dtype=VECTOR_DTYPE)
    data["vector"] = [float(v) for v in vector]
    data["dimension"] = int(row.dim)
    return Embedding.model_validate(data)


class SqlRepository:
    """SQLite-backed ``Repository``. Construct with ``SqlRepository(engine)``."""

    def __init__(self, engine: Engine, *, clock: Clock = utc_clock, actor: str = "system") -> None:
        self._engine = engine
        self._clock = clock
        self._actor = actor

    def _audit(
        self,
        conn: Connection,
        *,
        case_id: str | None,
        action: str,
        entity_type: str,
        entity_id: str,
        inserted: int,
    ) -> None:
        if inserted <= 0:
            return
        append_audit(
            conn,
            case_id=case_id,
            occurred_at=_now(self._clock),
            actor=self._actor,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            params={"inserted": inserted},
        )

    def _put_one(
        self,
        table: Table,
        model: FrozenModel,
        entity_id: str,
        scalars: Mapping[str, Any],
        *,
        case_id: str | None,
        action: str,
        entity_type: str,
        content_sha256: str | None = None,
        payload: str | None = None,
    ) -> None:
        recorded = _now(self._clock)
        body = payload if payload is not None else _dump(model)
        sha = content_sha256 if content_sha256 is not None else _content_sha(model)
        created = _created_at(model) or recorded
        with write_transaction(self._engine) as conn:
            inserted = _put_revision(
                conn,
                table,
                entity_id=entity_id,
                content_sha256=sha,
                created_at=created,
                recorded_at=recorded,
                payload=body,
                scalars=scalars,
            )
            self._audit(
                conn,
                case_id=case_id,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                inserted=int(inserted),
            )

    def _get_one(self, table: Table, model: type[T], entity_id: str, what: str) -> T:
        with read_transaction(self._engine) as conn:
            row = _require(_latest(conn, table, entity_id), what, entity_id)
        return _load_record(model, row.payload)

    def put_case(self, case: CaseWorkspace) -> None:
        self._put_one(
            cases,
            case,
            case.case_id,
            {"case_id": case.case_id},
            case_id=case.case_id,
            action="put_case",
            entity_type="case",
        )

    def get_case(self, case_id: str) -> CaseWorkspace:
        return self._get_one(cases, CaseWorkspace, case_id, "case")

    def put_video(self, video: SourceVideo) -> None:
        self._put_one(
            source_videos,
            video,
            video.video_id,
            {
                "video_id": video.video_id,
                "case_id": video.case_id,
                "sha256": video.sha256,
                "ingest_state": video.ingest_state.value,
            },
            case_id=video.case_id,
            action="put_video",
            entity_type="source_video",
        )

    def get_video(self, video_id: str) -> SourceVideo:
        return self._get_one(source_videos, SourceVideo, video_id, "source video")

    def list_videos(self, case_id: str) -> Sequence[SourceVideo]:
        with read_transaction(self._engine) as conn:
            rows = _latest_by(conn, source_videos, "case_id", case_id, order="created_at")
        return [_load_record(SourceVideo, r.payload) for r in rows]

    def put_manifest(self, manifest: FrameManifest) -> None:
        self._put_one(
            frame_manifests,
            manifest,
            manifest.video_id,
            {"video_id": manifest.video_id, "source_sha256": manifest.source_sha256},
            case_id=None,
            action="put_manifest",
            entity_type="frame_manifest",
        )

    def get_manifest(self, video_id: str) -> FrameManifest:
        return self._get_one(frame_manifests, FrameManifest, video_id, "frame manifest")

    def put_segments(self, segments_in: Sequence[VideoSegment]) -> None:
        recorded = _now(self._clock)
        inserted = 0
        video_id = segments_in[0].video_id if segments_in else ""
        with write_transaction(self._engine) as conn:
            for segment in segments_in:
                inserted += int(
                    _put_revision(
                        conn,
                        segments,
                        entity_id=segment.segment_id,
                        content_sha256=segment.content_sha256,
                        created_at=segment.created_at,
                        recorded_at=recorded,
                        payload=_dump(segment),
                        scalars={
                            "segment_id": segment.segment_id,
                            "video_id": segment.video_id,
                            "ordinal": segment.ordinal,
                            "index_state": segment.index_state.value,
                        },
                    )
                )
            self._audit(
                conn,
                case_id=None,
                action="put_segments",
                entity_type="segment",
                entity_id=video_id,
                inserted=inserted,
            )

    def list_segments(self, video_id: str) -> Sequence[VideoSegment]:
        with read_transaction(self._engine) as conn:
            rows = _latest_by(conn, segments, "video_id", video_id, order="ordinal")
        return [_load_record(VideoSegment, r.payload) for r in rows]

    def put_embeddings(self, video_id: str, items: Sequence[Embedding]) -> None:
        recorded = _now(self._clock)
        inserted = 0
        with write_transaction(self._engine) as conn:
            for emb in items:
                blob = _vector_blob(emb)
                sha = _embedding_sha(emb, blob)
                meta = emb.model_dump(mode="json")
                meta["vector"] = []
                inserted += int(
                    _put_revision(
                        conn,
                        embeddings,
                        entity_id=emb.embedding_ref,
                        content_sha256=sha,
                        created_at=recorded,
                        recorded_at=recorded,
                        payload=json.dumps(meta, separators=(",", ":"), sort_keys=True),
                        scalars={
                            "video_id": video_id,
                            "embedding_ref": emb.embedding_ref,
                            "model_id": emb.model_id,
                            "dim": emb.dimension,
                            "vector": blob,
                            "vector_dtype": "float64",
                        },
                    )
                )
            self._audit(
                conn,
                case_id=None,
                action="put_embeddings",
                entity_type="embedding",
                entity_id=video_id,
                inserted=inserted,
            )

    def list_embeddings(self, video_id: str) -> Sequence[Embedding]:
        with read_transaction(self._engine) as conn:
            rows = _latest_by(conn, embeddings, "video_id", video_id, order="embedding_ref")
        return [_load_embedding(r) for r in rows]

    def put_detections(self, items: Sequence[Detection]) -> None:
        recorded = _now(self._clock)
        inserted = 0
        video_id = items[0].video_id if items else ""
        with write_transaction(self._engine) as conn:
            for det in items:
                inserted += int(
                    _put_revision(
                        conn,
                        detections,
                        entity_id=det.detection_id,
                        content_sha256=det.content_sha256,
                        created_at=det.created_at,
                        recorded_at=recorded,
                        payload=_dump(det),
                        scalars={
                            "detection_id": det.detection_id,
                            "video_id": det.video_id,
                            "frame_id": det.frame_id,
                            "pts_us": det.pts_us,
                        },
                    )
                )
            self._audit(
                conn,
                case_id=None,
                action="put_detections",
                entity_type="detection",
                entity_id=video_id,
                inserted=inserted,
            )

    def list_detections(self, video_id: str) -> Sequence[Detection]:
        with read_transaction(self._engine) as conn:
            rows = _latest_by(conn, detections, "video_id", video_id, order="pts_us")
        return [_load_record(Detection, r.payload) for r in rows]

    def put_search(self, search: SearchEvidence) -> None:
        self._put_one(
            searches,
            search,
            search.search_id,
            {
                "search_id": search.search_id,
                "case_id": search.case_id,
                "video_id": search.video_id,
            },
            case_id=search.case_id,
            action="put_search",
            entity_type="search",
        )

    def get_search(self, search_id: str) -> SearchEvidence:
        return self._get_one(searches, SearchEvidence, search_id, "search")

    def put_track(self, track: Track) -> None:
        self._put_one(
            tracks,
            track,
            track.track_id,
            {
                "track_id": track.track_id,
                "case_id": track.case_id,
                "video_id": track.video_id,
                "state": track.state.value,
            },
            case_id=track.case_id,
            action="put_track",
            entity_type="track",
        )

    def get_track(self, track_id: str) -> Track:
        return self._get_one(tracks, Track, track_id, "track")

    def put_run(self, run: ReconstructionRun) -> None:
        self._put_one(
            reconstruction_runs,
            run,
            run.run_id,
            {
                "run_id": run.run_id,
                "case_id": run.case_id,
                "video_id": run.video_id,
                "track_id": run.track_id,
                "state": run.state.value,
            },
            case_id=run.case_id,
            action="put_run",
            entity_type="reconstruction_run",
        )

    def get_run(self, run_id: str) -> ReconstructionRun:
        return self._get_one(reconstruction_runs, ReconstructionRun, run_id, "reconstruction run")

    def put_decisions(self, decisions: Sequence[PolicyDecision]) -> None:
        recorded = _now(self._clock)
        inserted = 0
        run_id = decisions[0].run_id if decisions else ""
        case_id: str | None = None
        with write_transaction(self._engine) as conn:
            for decision in decisions:
                inserted += int(
                    _put_revision(
                        conn,
                        policy_decisions,
                        entity_id=decision.decision_id,
                        content_sha256=decision.content_sha256,
                        created_at=decision.created_at,
                        recorded_at=recorded,
                        payload=_dump(decision),
                        scalars={
                            "decision_id": decision.decision_id,
                            "run_id": decision.run_id,
                            "sequence": decision.sequence,
                        },
                    )
                )
            self._audit(
                conn,
                case_id=case_id,
                action="put_decisions",
                entity_type="policy_decision",
                entity_id=run_id,
                inserted=inserted,
            )

    def list_decisions(self, run_id: str) -> Sequence[PolicyDecision]:
        with read_transaction(self._engine) as conn:
            rows = _latest_by(conn, policy_decisions, "run_id", run_id, order="sequence")
        return [_load_record(PolicyDecision, r.payload) for r in rows]

    def put_provenance(self, provenance: PixelProvenance) -> None:
        self._put_one(
            pixel_provenance,
            provenance,
            provenance.run_id,
            {"run_id": provenance.run_id, "artifact_sha256": provenance.artifact_sha256},
            case_id=None,
            action="put_provenance",
            entity_type="pixel_provenance",
        )

    def get_provenance(self, run_id: str) -> PixelProvenance:
        return self._get_one(pixel_provenance, PixelProvenance, run_id, "pixel provenance")

    def put_review(self, review: HumanReview) -> None:
        self._put_one(
            human_reviews,
            review,
            review.review_id,
            {
                "review_id": review.review_id,
                "run_id": review.run_id,
                "decision": review.decision.value,
            },
            case_id=None,
            action="put_review",
            entity_type="human_review",
        )

    def latest_review(self, run_id: str) -> HumanReview | None:
        with read_transaction(self._engine) as conn:
            row = conn.execute(
                select(human_reviews)
                .where(human_reviews.c.run_id == run_id)
                .order_by(human_reviews.c.recorded_at.desc(), human_reviews.c.id.desc())
                .limit(1)
            ).first()
        if row is None:
            return None
        return _load_record(HumanReview, row.payload)

    def put_report(self, report: EvidenceReport) -> None:
        self._put_one(
            evidence_reports,
            report,
            report.report_id,
            {
                "report_id": report.report_id,
                "case_id": report.case_id,
                "run_id": report.run_id,
            },
            case_id=report.case_id,
            action="put_report",
            entity_type="evidence_report",
        )

    def get_report(self, report_id: str) -> EvidenceReport:
        return self._get_one(evidence_reports, EvidenceReport, report_id, "evidence report")

    def put_asset(self, asset: AssetRef) -> None:
        self._put_one(
            assets,
            asset,
            asset.asset_id,
            {
                "asset_id": asset.asset_id,
                "case_id": asset.case_id,
                "kind": asset.kind.value,
                "sha256": asset.sha256,
            },
            case_id=asset.case_id,
            action="put_asset",
            entity_type="asset",
        )

    def get_asset(self, asset_id: str) -> AssetRef:
        return self._get_one(assets, AssetRef, asset_id, "asset")

    def put_lineage(self, record: LineageRecord) -> None:
        self._put_one(
            lineage,
            record,
            record.lineage_id,
            {
                "lineage_id": record.lineage_id,
                "video_id": record.video_id,
                "kind": record.kind.value,
                "subject_entity_id": record.entity_id,
                "artifact_sha256": record.artifact_sha256,
            },
            case_id=None,
            action="put_lineage",
            entity_type="lineage",
        )
