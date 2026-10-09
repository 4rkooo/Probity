"""Coordinator slice: SQLite-backed search, and ingest that keeps committed segments."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from probity.adapters.fixture.cosmos import FixtureVideoUnderstanding
from probity.adapters.fixture.vast import FixtureEvidenceStore
from probity.domain.enums import (
    AdapterMode,
    ErrorCode,
    IndexState,
    InferenceMode,
    IngestState,
    JobKind,
    JobStage,
    JobState,
    ReasonCode,
    SearchStatus,
)
from probity.domain.errors import SponsorUnavailable, ValidationFailed
from probity.domain.fixtures import FixtureCatalog
from probity.domain.ids import new_uuid7, parse_segment_id, utc_now
from probity.domain.models import (
    Embedding,
    FrameManifest,
    FrameManifestEntry,
    JobView,
    SegmentDescription,
    SourceVideo,
    VideoSegment,
)
from probity.domain.policy import default_policy
from probity.domain.prompts import COSMOS_INGESTION_PROMPT, COSMOS_INGESTION_PROMPT_SHA256
from probity.ingest import IngestHandler
from probity.persistence import SqlRepository, create_engine_for, run_migrations
from probity.ports import (
    ClaimedJob,
    ClipReference,
    IngestPayload,
    SearchQuery,
    SegmentWindow,
    SourceVerification,
)
from probity.search.service import LocalSearchService
from probity.search.sqlite_store import SqlEvidenceStore

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "fixtures" / "demo"
CONTRACTS = ROOT / "fixtures" / "contracts"
QUERIES = json.loads((DEMO / "search" / "queries.json").read_text(encoding="utf-8"))
CORRELATION = "01a12166-60c0-7e9e-8227-4170b5a9bc3d"
SHA = "ab" * 32


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_search_fixture_catalog_loads() -> None:
    raw = (DEMO / "manifest.json").read_text(encoding="utf-8")
    catalog = FixtureCatalog.model_validate_json(raw)
    listed = {item.path: item.sha256 for item in catalog.fixtures[0].files}
    assert listed["search/descriptions.jsonl"].startswith("caa27cb0")
    assert listed["search/embeddings.npy"].startswith("c8c0b810")
    assert "search/queries.json" in listed


def _engine(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'cases.db'}"
    run_migrations(url)
    return create_engine_for(url)


@pytest.mark.anyio
async def test_sqlite_search_returns_timestamped_segment_ids(tmp_path: Path) -> None:
    understanding = FixtureVideoUnderstanding(DEMO, DEMO / "manifest.json")
    health = await understanding.health()
    assert health.status.value == "OK"
    video = SourceVideo.model_validate_json((CONTRACTS / "source_video.json").read_text())
    segments = [
        VideoSegment.model_validate(row)
        for row in json.loads((CONTRACTS / "video_segments.json").read_text())
    ]
    descriptions = [
        SegmentDescription.model_validate(row)
        for row in json.loads((CONTRACTS / "segment_descriptions.json").read_text())
    ]
    embeddings = list(await understanding.embed_segments(descriptions))
    assert [item.embedding_ref for item in embeddings] == [item.embedding_ref for item in segments]

    repository = SqlRepository(_engine(tmp_path))
    repository.put_video(video)
    repository.put_segments(segments)
    repository.put_embeddings(video.video_id, embeddings)

    store = SqlEvidenceStore(repository)
    assert store.inner.indexed_snapshot() == {}
    service = LocalSearchService(understanding, store, default_policy())
    spec = QUERIES["prepared_query"]
    evidence = await service.search(
        video, SearchQuery(query=spec["query"], max_results=5), CORRELATION
    )

    assert evidence.status is SearchStatus.OK
    assert evidence.embedding_model_id == "fixture/cosmos-embed-v1"
    gold = spec["golden_ranks"]
    assert [parse_segment_id(row.segment_id)[1] for row in evidence.results] == [
        row["ordinal"] for row in gold
    ]
    assert [row.segment_id.rsplit(":", 1)[-1] for row in evidence.results[:2]] == ["s0002", "s0001"]
    for result, row in zip(evidence.results, gold, strict=True):
        assert result.start_pts_us == row["start_pts_us"]
        assert result.end_pts_us == row["end_pts_us"]
        assert result.end_pts_us > result.start_pts_us
        assert abs(result.score - row["score"]) < 1e-6


class _Clock:
    def __init__(self) -> None:
        self.cancel = self

    def is_cancelled(self) -> bool:
        return False

    def checkpoint(self) -> None:
        return None

    def progress(self, stage: JobStage, completed: int, total: int) -> None:
        return None


class _Source:
    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def data_dir(self) -> Path:
        return self._root

    def verify(self, storage_uri: str, expected_sha256: str) -> SourceVerification:
        return SourceVerification(
            storage_uri=storage_uri,
            expected_sha256=expected_sha256,
            observed_sha256=expected_sha256,
            verified=True,
            checked_at=utc_now(),
            reason_code=ReasonCode.SOURCE_HASH_VERIFIED,
        )

    def resolve_read_path(self, storage_uri: str) -> str:
        return str(self._root / "source.mp4")

    def derived_path(self, video_id: str, kind: str, entity_id: str, filename: str) -> str:
        path = self._root / "derived" / video_id / kind / entity_id / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)


class _Prober:
    def __init__(self, manifest: FrameManifest) -> None:
        self._manifest = manifest

    def frame_manifest(self, path: str, video_id: str, source_sha256: str) -> FrameManifest:
        return self._manifest


class _Segmenter:
    def __init__(self, windows: tuple[SegmentWindow, ...]) -> None:
        self._windows = windows

    def plan(self, manifest: FrameManifest, duration_us: int) -> tuple[SegmentWindow, ...]:
        return self._windows


class _Thumbs:
    def extract_thumbnail(
        self, source_path: str, window: SegmentWindow, manifest: FrameManifest, out_path: str
    ) -> int:
        Path(out_path).write_bytes(b"\xff\xd8\xff")
        return window.start_frame


class _Understanding:
    adapter_name = "cosmos-fixture"
    model_id = "fixture/cosmos-embed-v1"
    mode = AdapterMode.FIXTURE
    schema_version = "1.0"

    def __init__(self, fail: BaseException) -> None:
        self._fail = fail

    async def describe(self, clip: ClipReference, prompt: str) -> SegmentDescription:
        assert prompt == COSMOS_INGESTION_PROMPT
        ordinal = int(clip.ordinal)
        if ordinal > 0:
            raise self._fail
        video_id = str(clip.video_id)
        return SegmentDescription(
            segment_id=str(clip.segment_id),
            video_id=video_id,
            source_sha256=str(clip.source_sha256),
            time_range={
                "start_pts_us": int(clip.start_pts_us),
                "end_pts_us": int(clip.end_pts_us),
            },
            model_id="fixture/cosmos-reason-describe-v1",
            prompt_sha256=COSMOS_INGESTION_PROMPT_SHA256,
            raw_output_sha256=SHA,
            summary="A blue sedan with a partly visible rear plate.",
            search_terms=("blue sedan", "license plate"),
            confidence=0.8,
            mode=InferenceMode.FIXTURE,
        )

    async def embed_segments(self, items: tuple[SegmentDescription, ...]) -> tuple[Embedding, ...]:
        return tuple(
            Embedding(
                embedding_ref=f"mem/{item.segment_id}",
                model_id="fixture/cosmos-embed-v1",
                dimension=2,
                vector=(1.0, 0.0),
                normalized=True,
                input_sha256=SHA,
                mode=InferenceMode.FIXTURE,
            )
            for item in items
        )


def _claimed(video: SourceVideo) -> ClaimedJob:
    now = utc_now()
    job = JobView.create(
        job_id=new_uuid7(),
        kind=JobKind.INGEST,
        case_id=video.case_id,
        subject_id=video.video_id,
        state=JobState.RUNNING,
        stage=JobStage.VALIDATE,
        mode=InferenceMode.FIXTURE,
        correlation_id=CORRELATION,
        updated_at=now,
        created_at=now,
    )
    return ClaimedJob(
        job=job,
        payload=IngestPayload(video_id=video.video_id),
        lease_owner="test-worker",
        lease_expires_at=now,
    )


class _FailSecondUpsert(FixtureEvidenceStore):
    def __init__(self) -> None:
        super().__init__()
        self.upserts = 0

    async def upsert_segments(self, segments, embeddings) -> None:  # noqa: ANN001
        self.upserts += 1
        if self.upserts > 1:
            raise RuntimeError("index write failed")
        await super().upsert_segments(segments, embeddings)


def _handler(
    tmp_path: Path,
    video: SourceVideo,
    fail: BaseException,
    evidence_store: FixtureEvidenceStore | None = None,
) -> tuple[IngestHandler, SqlRepository]:
    repository = SqlRepository(_engine(tmp_path))
    stored = video.revise(ingest_state=IngestState.STORED, indexed_range=None)
    repository.put_video(stored)
    manifest = FrameManifest.create(
        video_id=video.video_id,
        source_sha256=video.sha256,
        time_base=video.time_base,
        extractor="test",
        frames=(
            FrameManifestEntry(
                frame_number=0, pts_us=0, pts=0, is_keyframe=True, width_px=16, height_px=16
            ),
            FrameManifestEntry(
                frame_number=1, pts_us=33333, pts=512, is_keyframe=False, width_px=16, height_px=16
            ),
        ),
    )
    windows = (
        SegmentWindow(ordinal=0, start_pts_us=0, end_pts_us=8_000_000, start_frame=0, end_frame=1),
        SegmentWindow(
            ordinal=1, start_pts_us=6_000_000, end_pts_us=14_000_000, start_frame=0, end_frame=1
        ),
    )
    handler = IngestHandler(
        repository=repository,
        source_store=_Source(tmp_path),
        prober=_Prober(manifest),
        segmenter=_Segmenter(windows),
        thumbnails=_Thumbs(),
        understanding=_Understanding(fail),
        evidence_store=evidence_store or FixtureEvidenceStore(),
        new_id=new_uuid7,
    )
    return handler, repository


@pytest.mark.anyio
async def test_failed_segment_keeps_committed_rows_as_partial(tmp_path: Path) -> None:
    video = SourceVideo.model_validate_json((CONTRACTS / "source_video.json").read_text())
    handler, repository = _handler(tmp_path, video, ValidationFailed("segment broke"))
    outcome = await handler.run(_claimed(video), _Clock())
    saved = repository.get_video(video.video_id)
    segments = repository.list_segments(video.video_id)
    assert outcome.state is JobState.PARTIAL
    assert outcome.partial is not None
    assert len(outcome.partial.committed_segment_ids) == 1
    assert len(segments) == 1
    assert segments[0].index_state is IndexState.INDEXED
    assert segments[0].start_pts_us == 0
    assert saved.ingest_state is IngestState.PARTIAL
    assert saved.indexed_range is not None


@pytest.mark.anyio
async def test_index_failure_does_not_claim_uncommitted_segment(tmp_path: Path) -> None:
    video = SourceVideo.model_validate_json((CONTRACTS / "source_video.json").read_text())
    evidence = _FailSecondUpsert()
    handler, _repository = _handler(
        tmp_path, video, ValidationFailed("unused"), evidence_store=evidence
    )
    outcome = await handler.run(_claimed(video), _Clock())
    assert outcome.state is JobState.PARTIAL
    assert outcome.partial is not None
    failed_id = f"{video.video_id}:s0001"
    assert failed_id in outcome.partial.failed_segment_ids
    assert failed_id not in outcome.partial.committed_segment_ids
    assert len(outcome.partial.committed_segment_ids) == 1
    assert len(outcome.partial.indexed_ranges) == 1


@pytest.mark.anyio
async def test_sponsor_error_is_retryable_and_keeps_rows(tmp_path: Path) -> None:
    video = SourceVideo.model_validate_json((CONTRACTS / "source_video.json").read_text())
    handler, repository = _handler(
        tmp_path, video, SponsorUnavailable("cosmos live adapter is disabled")
    )
    outcome = await handler.run(_claimed(video), _Clock())
    assert outcome.state is JobState.FAILED
    assert outcome.error is not None
    assert outcome.error.code is ErrorCode.SPONSOR_UNAVAILABLE
    assert outcome.error.retryable is True
    assert len(repository.list_segments(video.video_id)) == 1
    assert repository.get_video(video.video_id).ingest_state is IngestState.PARTIAL
