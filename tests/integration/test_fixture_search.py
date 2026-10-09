"""Fixture search integration: prepared query ranks and policy clarification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from probity.adapters.fixture.cosmos import FixtureVideoUnderstanding
from probity.adapters.fixture.vast import FixtureEvidenceStore
from probity.domain.enums import SearchStatus
from probity.domain.ids import parse_segment_id
from probity.domain.models import SourceVideo, VideoSegment
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
