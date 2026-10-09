"""Fixture search integration: prepared query ranks and policy clarification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from probity.adapters.fallback import FallbackEvidenceStore, FallbackVideoUnderstanding
from probity.adapters.fixture.cosmos import FixtureVideoUnderstanding
from probity.adapters.fixture.vast import FixtureEvidenceStore
from probity.adapters.live.cosmos import LiveCosmosUnderstanding
from probity.adapters.live.vast import LiveVastEvidenceStore
from probity.domain.enums import AdapterMode, InferenceMode, OperatingMode, SearchStatus
from probity.domain.ids import parse_segment_id
from probity.domain.models import Embedding, SourceVideo, VideoSegment
from probity.domain.policy import default_policy
from probity.ports import SearchQuery
from probity.search.service import LocalSearchService

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "fixtures/demo"
CONTRACTS = ROOT / "fixtures/contracts"
QUERIES = json.loads((DEMO / "search/queries.json").read_text(encoding="utf-8"))
CORRELATION_OK = "01a12166-60c0-7e9e-8227-4170b5a9bc3d"
CORRELATION_CLARIFY = "01a12166-87d0-7557-b5f1-bd07ba19cb1e"

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _service() -> tuple[LocalSearchService, SourceVideo]:
    understanding = FixtureVideoUnderstanding(DEMO, DEMO / "manifest.json")
    store = FixtureEvidenceStore()
    video_path = CONTRACTS / "source_video.json"
    video = SourceVideo.model_validate_json(video_path.read_text(encoding="utf-8"))
    segments = [
        VideoSegment.model_validate(row)
        for row in json.loads((CONTRACTS / "video_segments.json").read_text(encoding="utf-8"))
    ]
    descriptions = json.loads((CONTRACTS / "segment_descriptions.json").read_text(encoding="utf-8"))
    from probity.domain.models import SegmentDescription

    items = [SegmentDescription.model_validate(row) for row in descriptions]
    embeddings = await understanding.embed_segments(items)
    await store.upsert_segments(segments, embeddings)
    return LocalSearchService(understanding, store, default_policy()), video


async def test_prepared_query_returns_timestamped_evidence_with_golden_ranks() -> None:
    service, video = await _service()
    spec = QUERIES["prepared_query"]
    evidence = await service.search(
        video, SearchQuery(query=spec["query"], max_results=5), CORRELATION_OK
    )
    assert evidence.status is SearchStatus.OK
    assert evidence.query_plan.model_dump(mode="json") == spec["expected_plan"]
    gold = spec["golden_ranks"]
    assert [parse_segment_id(row.segment_id)[1] for row in evidence.results] == [
        row["ordinal"] for row in gold
    ]
    assert [parse_segment_id(row.segment_id)[1] for row in evidence.results[:2]] == [2, 1]
    for result, row in zip(evidence.results, gold, strict=True):
        assert abs(result.score - row["score"]) < 1e-9
        assert result.start_pts_us == row["start_pts_us"]
        assert result.end_pts_us == row["end_pts_us"]
        assert result.end_pts_us > result.start_pts_us
        assert result.explanation == row["explanation"]
        assert result.source_sha256 == video.sha256
        assert result.video_id == video.video_id
        assert "Segment " in result.explanation
    assert evidence.mode.value == "FIXTURE"
    assert evidence.indexed_range == video.indexed_range
    assert evidence.correlation_id == CORRELATION_OK


async def test_guilty_driver_query_needs_clarification() -> None:
    service, video = await _service()
    spec = QUERIES["clarification_query"]
    evidence = await service.search(video, SearchQuery(query=spec["query"]), CORRELATION_CLARIFY)
    assert evidence.status is SearchStatus.NEEDS_CLARIFICATION
    assert evidence.results == ()
    assert evidence.candidates == ()
    assert evidence.query_plan.model_dump(mode="json") == spec["expected_plan"]
    assert evidence.query_plan.needs_clarification is True


class _SleepLog:
    async def __call__(self, seconds: float) -> None:
        return None


class _TimeoutTransport:
    def __init__(self) -> None:
        self.health_calls = 0

    async def execute(self, operation: str, payload: dict[str, Any]) -> Any:
        if operation == "health":
            self.health_calls += 1
            return {"ok": True}
        raise TimeoutError


class _ModeAdapter:
    def __init__(self, initial: AdapterMode, final: AdapterMode) -> None:
        self._mode = initial
        self._final = final
        self.model_id = "live/cosmos-embed-v1"

    @property
    def mode(self) -> AdapterMode:
        return self._mode

    def _use(self) -> None:
        self._mode = self._final
        if self._final is AdapterMode.FIXTURE:
            self.model_id = "fixture/cosmos-embed-v1"


class _ModeUnderstanding(_ModeAdapter):
    async def embed_query(self, query: str) -> Embedding:
        self._use()
        produced = InferenceMode.FIXTURE if self.mode is AdapterMode.FIXTURE else InferenceMode.LIVE
        return Embedding(
            embedding_ref="test/query",
            model_id=self.model_id,
            dimension=1,
            vector=(1.0,),
            normalized=True,
            input_sha256="ab" * 32,
            mode=produced,
        )


class _ModeStore(_ModeAdapter):
    async def search(self, request: object) -> tuple[()]:
        self._use()
        return ()


def _video() -> SourceVideo:
    raw = (CONTRACTS / "source_video.json").read_text(encoding="utf-8")
    return SourceVideo.model_validate_json(raw)


async def test_search_records_fixture_mode_when_retrieval_falls_back() -> None:
    understanding = _ModeUnderstanding(AdapterMode.LIVE, AdapterMode.FIXTURE)
    store = _ModeStore(AdapterMode.LIVE, AdapterMode.FIXTURE)
    service = LocalSearchService(understanding, store, default_policy())
    evidence = await service.search(
        _video(),
        SearchQuery(query=QUERIES["prepared_query"]["query"]),
        CORRELATION_OK,
    )
    assert understanding.mode is AdapterMode.FIXTURE
    assert store.mode is AdapterMode.FIXTURE
    assert evidence.mode is InferenceMode.FIXTURE
    assert evidence.embedding_model_id == "fixture/cosmos-embed-v1"
    assert evidence.status is SearchStatus.NO_RESULTS


async def test_search_records_live_mode_when_retrieval_stays_live() -> None:
    understanding = _ModeUnderstanding(AdapterMode.LIVE, AdapterMode.LIVE)
    store = _ModeStore(AdapterMode.LIVE, AdapterMode.LIVE)
    service = LocalSearchService(understanding, store, default_policy())
    evidence = await service.search(
        _video(),
        SearchQuery(query=QUERIES["prepared_query"]["query"]),
        CORRELATION_OK,
    )
    assert evidence.mode is InferenceMode.LIVE
    assert evidence.embedding_model_id == "live/cosmos-embed-v1"


async def test_auto_timeout_fallback_is_disclosed_as_fixture() -> None:
    cosmos_transport = _TimeoutTransport()
    vast_transport = _TimeoutTransport()
    understanding = FallbackVideoUnderstanding(
        LiveCosmosUnderstanding(
            enabled=True,
            endpoint="https://example.invalid",
            token=SecretStr("unused"),
            transport=cosmos_transport,
            sleep=_SleepLog(),
        ),
        FixtureVideoUnderstanding(DEMO, DEMO / "manifest.json"),
        OperatingMode.AUTO,
    )
    store = FallbackEvidenceStore(
        LiveVastEvidenceStore(
            enabled=True,
            endpoint="https://example.invalid",
            token=SecretStr("unused"),
            transport=vast_transport,
            sleep=_SleepLog(),
        ),
        FixtureEvidenceStore(),
        OperatingMode.AUTO,
    )
    assert understanding.mode is AdapterMode.DEGRADED
    assert store.mode is AdapterMode.DEGRADED
    service = LocalSearchService(understanding, store, default_policy())
    evidence = await service.search(
        _video(),
        SearchQuery(query=QUERIES["prepared_query"]["query"]),
        CORRELATION_OK,
    )
    assert evidence.mode is InferenceMode.FIXTURE
    assert understanding.mode is AdapterMode.FIXTURE
    assert store.mode is AdapterMode.FIXTURE
    assert cosmos_transport.health_calls == 1
    assert vast_transport.health_calls == 1
    assert evidence.status is SearchStatus.NO_RESULTS


class _StuckDegraded:
    mode = AdapterMode.DEGRADED
    model_id = "live/cosmos-embed-v1"


async def test_unresolved_degraded_clarification_is_not_recorded_as_live() -> None:
    service = LocalSearchService(_StuckDegraded(), _StuckDegraded(), default_policy())
    evidence = await service.search(
        _video(),
        SearchQuery(query=QUERIES["clarification_query"]["query"]),
        CORRELATION_CLARIFY,
    )
    assert evidence.status is SearchStatus.NEEDS_CLARIFICATION
    assert evidence.mode is InferenceMode.FIXTURE


async def test_clarification_uses_startup_health_not_unresolved_degraded() -> None:
    understanding = FallbackVideoUnderstanding(
        LiveCosmosUnderstanding(
            enabled=True,
            endpoint="https://example.invalid",
            token=SecretStr("unused"),
            transport=_TimeoutTransport(),
            sleep=_SleepLog(),
        ),
        FixtureVideoUnderstanding(DEMO, DEMO / "manifest.json"),
        OperatingMode.AUTO,
    )
    store = FallbackEvidenceStore(
        LiveVastEvidenceStore(
            enabled=True,
            endpoint="https://example.invalid",
            token=SecretStr("unused"),
            transport=_TimeoutTransport(),
            sleep=_SleepLog(),
        ),
        FixtureEvidenceStore(),
        OperatingMode.AUTO,
    )
    assert understanding.mode is AdapterMode.DEGRADED
    service = LocalSearchService(understanding, store, default_policy())
    evidence = await service.search(
        _video(),
        SearchQuery(query=QUERIES["clarification_query"]["query"]),
        CORRELATION_CLARIFY,
    )
    assert evidence.status is SearchStatus.NEEDS_CLARIFICATION
    assert evidence.mode is InferenceMode.LIVE
    assert understanding.mode is AdapterMode.LIVE
    assert store.mode is AdapterMode.LIVE
