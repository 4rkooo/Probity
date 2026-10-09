"""Worker units, kill/restart recovery, repository round-trip, revisioning, audit tamper."""

from __future__ import annotations

import asyncio
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import insert

from probity.domain.enums import InferenceMode, JobKind, JobStage, JobState, LineageKind
from probity.domain.ids import new_uuid7
from probity.domain.models import (
    AssetRef,
    CaseWorkspace,
    Detection,
    Embedding,
    EvidenceReport,
    FrameManifest,
    HumanReview,
    LineageRecord,
    PixelProvenance,
    PolicyDecision,
    ReconstructionRun,
    SearchEvidence,
    SourceVideo,
    Track,
    VideoSegment,
)
from probity.jobs import SqlJobStore
from probity.persistence.audit import verify_chain
from probity.persistence.db import create_engine_for, run_migrations, write_transaction
from probity.persistence.repository import SqlRepository
from probity.persistence.schema import audit_log
from probity.ports import ClaimedJob, IngestPayload, JobOutcome, JobSpec
from probity.worker import Worker, WorkerJobContext, health_line, redact_db_path

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "fixtures/contracts"

FIXTURE_SINGLES: list[tuple[str, type, str]] = [
    ("case_workspace.json", CaseWorkspace, "put_case"),
    ("source_video.json", SourceVideo, "put_video"),
    ("frame_manifest.json", FrameManifest, "put_manifest"),
    ("search_evidence.json", SearchEvidence, "put_search"),
    ("search_evidence_needs_clarification.json", SearchEvidence, "put_search"),
    ("track_confirmed.json", Track, "put_track"),
    ("track_not_confirmed.json", Track, "put_track"),
    ("reconstruction_run_succeeded.json", ReconstructionRun, "put_run"),
    ("reconstruction_run_refused.json", ReconstructionRun, "put_run"),
    ("reconstruction_run_failed.json", ReconstructionRun, "put_run"),
    ("pixel_provenance.json", PixelProvenance, "put_provenance"),
    ("human_review_approve.json", HumanReview, "put_review"),
    ("human_review_veto.json", HumanReview, "put_review"),
    ("evidence_report.json", EvidenceReport, "put_report"),
]


def _engine(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'cases.db'}"
    engine = create_engine_for(url)
    run_migrations(url, engine=engine)
    return engine, url


def _load(name: str) -> list[dict[str, object]]:
    data = json.loads((CONTRACTS / name).read_text())
    return data if isinstance(data, list) else [data]


def _spec(video_id: str | None = None, units: int = 5) -> JobSpec:
    video_id = video_id or new_uuid7()
    return JobSpec(
        job_id=new_uuid7(),
        case_id=new_uuid7(),
        subject_id=video_id,
        payload=IngestPayload(video_id=video_id),
        total_units=units,
        correlation_id=new_uuid7(),
    )


class _CountingHandler:
    kind = JobKind.INGEST

    def __init__(self, units: int) -> None:
        self.units = units
        self.seen = 0

    async def run(self, claimed: ClaimedJob, ctx: WorkerJobContext) -> JobOutcome:
        for i in range(self.units):
            ctx.cancel.checkpoint()
            ctx.progress(JobStage.INDEX, i + 1, self.units)
            self.seen = i + 1
            await asyncio.sleep(0)
        return JobOutcome(state=JobState.SUCCEEDED)


class _Clock:
    def __init__(self) -> None:
        self.t = datetime(2026, 10, 9, 16, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t = self.t + timedelta(seconds=seconds)


def test_fake_handler_n_units(tmp_path: Path) -> None:
    engine, _url = _engine(tmp_path)
    store = SqlJobStore(engine, lease_seconds=30)
    units = 7
    spec = _spec(units=units)
    store.create(spec)
    handler = _CountingHandler(units)
    worker = Worker(store, [handler], "w1", lease_seconds=30, poll_seconds=0.01)

    async def _run() -> None:
        task = asyncio.create_task(worker.run())
        for _ in range(200):
            if store.get(spec.job_id).state is JobState.SUCCEEDED:
                break
            await asyncio.sleep(0.02)
        worker.request_stop()
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(_run())
    assert handler.seen == units
    assert store.get(spec.job_id).state is JobState.SUCCEEDED
    assert store.get(spec.job_id).completed_units == units


def test_kill_restart_recovery(tmp_path: Path) -> None:
    clock = _Clock()
    engine, url = _engine(tmp_path)
    store = SqlJobStore(engine, clock=clock, lease_seconds=5)
    spec = _spec()
    store.create(spec)
    claimed = store.claim_next("dead-worker", 5)
    assert claimed is not None
    clock.advance(60)
    worker = Worker(store, [], "fresh-worker", lease_seconds=5, poll_seconds=0.01)
    assert worker.recover() == 1
    view = store.get(spec.job_id)
    assert view.state is JobState.FAILED
    assert view.error is not None and view.error.retryable is True
    line = health_line(worker.owner_id, url, worker.recovered_count)
    payload = json.loads(line)
    assert payload["process"] == "worker"
    assert payload["recovered"] == 1
    assert payload["database"] == redact_db_path(url)


def test_missing_handler_fails_internal(tmp_path: Path) -> None:
    engine, _url = _engine(tmp_path)
    store = SqlJobStore(engine, lease_seconds=30)
    spec = _spec()
    store.create(spec)
    worker = Worker(store, [], "w1", lease_seconds=30, poll_seconds=0.01)

    async def _run() -> None:
        task = asyncio.create_task(worker.run())
        for _ in range(200):
            if store.get(spec.job_id).state is JobState.FAILED:
                break
            await asyncio.sleep(0.02)
        worker.request_stop()
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(_run())
    view = store.get(spec.job_id)
    assert view.state is JobState.FAILED
    assert view.error is not None
    assert view.error.code.value == "INTERNAL_ERROR"
    assert "Traceback" not in view.error.message
    assert view.error.message == "Internal error"


def test_put_segments_twice_no_duplicate(tmp_path: Path) -> None:
    engine, _url = _engine(tmp_path)
    repo = SqlRepository(engine)
    raw = _load("video_segments.json")
    segs = [VideoSegment.model_validate(item) for item in raw]
    repo.put_segments(segs)
    repo.put_segments(segs)
    listed = repo.list_segments(segs[0].video_id)
    assert len(listed) == len(segs)
    assert [s.segment_id for s in listed] == [s.segment_id for s in segs]
    assert listed[0].content_sha256 == segs[0].content_sha256


def test_source_video_revisioning(tmp_path: Path) -> None:
    engine, _url = _engine(tmp_path)
    repo = SqlRepository(engine)
    stored = SourceVideo.model_validate(_load("source_video.json")[0])
    partial = SourceVideo.model_validate(_load("source_video_partial.json")[0])
    repo.put_video(stored)
    assert repo.get_video(stored.video_id).content_sha256 == stored.content_sha256
    repo.put_video(stored)
    repo.put_video(partial)
    got = repo.get_video(stored.video_id)
    assert got.model_dump(mode="json") == partial.model_dump(mode="json")
    assert got.content_sha256 == partial.content_sha256
    assert got.content_sha256 != stored.content_sha256


def test_repository_round_trip_contract_fixtures(tmp_path: Path) -> None:
    engine, _url = _engine(tmp_path)
    repo = SqlRepository(engine)
    for name, model, method in FIXTURE_SINGLES:
        for raw in _load(name):
            obj = model.model_validate(raw)
            getattr(repo, method)(obj)
    case_raw = _load("case_workspace.json")[0]
    assert repo.get_case(case_raw["case_id"]).model_dump(mode="json") == case_raw
    video_raw = _load("source_video.json")[0]
    assert repo.get_video(video_raw["video_id"]).content_sha256 == video_raw["content_sha256"]
    manifest_raw = _load("frame_manifest.json")[0]
    assert repo.get_manifest(manifest_raw["video_id"]).model_dump(mode="json") == manifest_raw
    search = _load("search_evidence.json")[0]
    assert repo.get_search(search["search_id"]).model_dump(mode="json") == search
    track = _load("track_confirmed.json")[0]
    assert repo.get_track(track["track_id"]).model_dump(mode="json") == track
    run = _load("reconstruction_run_succeeded.json")[0]
    assert repo.get_run(run["run_id"]).model_dump(mode="json") == run
    prov = _load("pixel_provenance.json")[0]
    assert repo.get_provenance(prov["run_id"]).model_dump(mode="json") == prov
    report = _load("evidence_report.json")[0]
    assert repo.get_report(report["report_id"]).model_dump(mode="json") == report
    approve = HumanReview.model_validate(_load("human_review_approve.json")[0])
    veto = HumanReview.model_validate(_load("human_review_veto.json")[0])
    latest = repo.latest_review(approve.run_id)
    assert latest is not None
    assert latest.review_id == veto.review_id

    segs = [VideoSegment.model_validate(x) for x in _load("video_segments.json")]
    repo.put_segments(segs)
    listed = repo.list_segments(segs[0].video_id)
    assert [s.model_dump(mode="json") for s in listed] == [s.model_dump(mode="json") for s in segs]

    dets = [Detection.model_validate(x) for x in _load("detections.json")]
    repo.put_detections(dets)
    listed_d = repo.list_detections(dets[0].video_id)
    listed_dump = [d.model_dump(mode="json") for d in listed_d]
    assert listed_dump == [d.model_dump(mode="json") for d in dets]

    decisions = [PolicyDecision.model_validate(x) for x in _load("policy_decisions_succeeded.json")]
    repo.put_decisions(decisions)
    listed_p = repo.list_decisions(decisions[0].run_id)
    assert [p.model_dump(mode="json") for p in listed_p] == [
        p.model_dump(mode="json") for p in decisions
    ]
    refused = [PolicyDecision.model_validate(x) for x in _load("policy_decisions_refused.json")]
    repo.put_decisions(refused)
    listed_r = repo.list_decisions(refused[0].run_id)
    assert [p.model_dump(mode="json") for p in listed_r] == [
        p.model_dump(mode="json") for p in refused
    ]

    assets = [AssetRef.model_validate(x) for x in _load("asset_refs.json")]
    for asset in assets:
        repo.put_asset(asset)
    assert repo.get_asset(assets[0].asset_id).model_dump(mode="json") == assets[0].model_dump(
        mode="json"
    )


def test_embedding_float64_round_trip(tmp_path: Path) -> None:
    engine, _url = _engine(tmp_path)
    repo = SqlRepository(engine)
    video_id = new_uuid7()
    vector = (0.5, 1.0 / 3.0, math.pi / 8.0, -0.25)
    emb = Embedding(
        embedding_ref=f"{video_id}:s0000",
        model_id="cosmos-embed/v1",
        dimension=4,
        vector=vector,
        normalized=False,
        input_sha256="ab" * 32,
        mode=InferenceMode.FIXTURE,
    )
    repo.put_embeddings(video_id, [emb])
    repo.put_embeddings(video_id, [emb])
    got = repo.list_embeddings(video_id)
    assert len(got) == 1
    assert got[0].vector == emb.vector
    assert got[0].dimension == 4
    assert got[0].embedding_ref == emb.embedding_ref


def test_lineage_put_and_audit_tamper(tmp_path: Path) -> None:
    engine, _url = _engine(tmp_path)
    repo = SqlRepository(engine)
    store = SqlJobStore(engine, lease_seconds=30)
    spec = _spec()
    store.create(spec)
    assert verify_chain(engine, spec.case_id).valid is True
    record = LineageRecord.create(
        lineage_id=new_uuid7(),
        kind=LineageKind.SOURCE,
        entity_id=spec.subject_id,
        video_id=spec.subject_id,
        parent_sha256="ab" * 32,
        artifact_uri="source/sha256/ab/" + "ab" * 32 + "/original.mp4",
        artifact_sha256="ab" * 32,
        algorithm_version="ingest-v1",
        mode=InferenceMode.FIXTURE,
    )
    repo.put_lineage(record)
    with write_transaction(engine) as conn:
        conn.execute(
            insert(audit_log).values(
                chain_key=spec.case_id,
                sequence=99,
                occurred_at="2026-10-09T16:00:00.000000Z",
                actor="attacker",
                action="tamper",
                entity_type="job",
                entity_id=spec.job_id,
                correlation_id=None,
                reason_code=None,
                params="{}",
                previous_row_hash="1" * 64,
                row_hash="2" * 64,
            )
        )
    result = verify_chain(engine, spec.case_id)
    assert result.valid is False
    assert result.first_broken_sequence == 99
