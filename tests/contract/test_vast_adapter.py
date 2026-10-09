"""VAST fixture vs live-stub contract: search parity, malformed, retries, fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from probity.adapters.fallback import FallbackEvidenceStore, snapshot_value
from probity.adapters.fixture.cosmos import EMBED_MODEL_ID, FIXTURE_ID, FixtureVideoUnderstanding
from probity.adapters.fixture.vast import FixtureEvidenceStore
from probity.adapters.live.vast import LiveVastEvidenceStore, UncommittedWrite
from probity.domain.enums import AdapterMode, HealthStatus, OperatingMode
from probity.domain.errors import SponsorMalformedResponse, SponsorTimeout, SponsorUnavailable
from probity.domain.ids import parse_segment_id
from probity.domain.models import Embedding, SourceVideo, VideoSegment
from probity.ports import SearchRequest
from probity.search.embedding import embedding_input_text, input_sha256

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "fixtures/demo"
CONTRACTS = ROOT / "fixtures/contracts"

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class SleepLog:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


class FixtureBackedVastTransport:
    def __init__(self, inner: FixtureEvidenceStore) -> None:
        self.inner = inner
        self.write_calls = 0

    async def execute(self, operation: str, payload: dict[str, Any]) -> Any:
        if operation == "health":
            return {"ok": True}
        if operation == "search":
            request = SearchRequest.model_validate(payload)
            rows = await self.inner.search(request)
            return [row.model_dump(mode="json") for row in rows]
        if operation == "upsert_segments":
            self.write_calls += 1
            segments = [VideoSegment.model_validate(row) for row in payload["segments"]]
            embeddings = [Embedding.model_validate(row) for row in payload["embeddings"]]
            await self.inner.upsert_segments(segments, embeddings)
            return {"ok": True}
        if operation == "put_source":
            video = SourceVideo.model_validate(payload["video"])
            return await self.inner.put_source(video, str(payload["local_path"]))
        raise SponsorMalformedResponse(f"unknown operation {operation}")


class TimeoutTransport:
    async def execute(self, operation: str, payload: dict[str, Any]) -> Any:
        if operation == "health":
            return {"ok": True}
        raise TimeoutError


class MalformedTransport:
    async def execute(self, operation: str, payload: dict[str, Any]) -> Any:
        if operation == "health":
            return {"ok": True}
        return {"not": "a list"}


class OnceUncommitted:
    def __init__(self) -> None:
        self.attempts = 0

    async def execute(self, operation: str, payload: dict[str, Any]) -> Any:
        if operation == "health":
            return {"ok": True}
        if operation == "upsert_segments":
            self.attempts += 1
            if self.attempts == 1:
                raise UncommittedWrite("not committed")
            return {"ok": True}
        raise SponsorMalformedResponse(operation)


def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
    skip = {"mode", "adapter_name", "checked_at", "latency_ms", "detail"}
    return {key: value for key, value in payload.items() if key not in skip}


async def _indexed_store() -> tuple[FixtureEvidenceStore, SourceVideo, Embedding]:
    store = FixtureEvidenceStore()
    understanding = FixtureVideoUnderstanding(DEMO, DEMO / "manifest.json")
    video_path = CONTRACTS / "source_video.json"
    video = SourceVideo.model_validate_json(video_path.read_text(encoding="utf-8"))
    segments = [
        VideoSegment.model_validate(row)
        for row in json.loads((CONTRACTS / "video_segments.json").read_text(encoding="utf-8"))
    ]
    descriptions = json.loads((CONTRACTS / "segment_descriptions.json").read_text(encoding="utf-8"))
    embeddings: list[Embedding] = []
    for segment, raw in zip(segments, descriptions, strict=True):
        from probity.domain.models import SegmentDescription

        desc = SegmentDescription.model_validate(raw)
        text = embedding_input_text(desc.summary, desc.search_terms)
        vector = (await understanding.embed_segments([desc]))[0]
        embeddings.append(
            Embedding(
                embedding_ref=f"{FIXTURE_ID}/embeddings.npy#{segment.ordinal}",
                model_id=EMBED_MODEL_ID,
                dimension=64,
                vector=vector.vector,
                normalized=True,
                input_sha256=input_sha256(text),
                mode=desc.mode,
            )
        )
    await store.upsert_segments(segments, embeddings)
    query = await understanding.embed_query("blue sedan rear license plate most visible car")
    return store, video, query


async def test_fixture_health_ok() -> None:
    health = await FixtureEvidenceStore().health()
    assert health.status is HealthStatus.OK
    assert health.adapter_name == "vast-fixture"
    assert health.mode is AdapterMode.FIXTURE


async def test_search_does_not_mutate_inputs() -> None:
    store, video, query = await _indexed_store()
    request = SearchRequest(
        video_id=video.video_id,
        source_sha256=video.sha256,
        query_embedding=query,
        top_k=12,
    )
    before = snapshot_value(request)
    rows = await store.search(request)
    assert snapshot_value(request) == before
    assert rows
    assert parse_segment_id(rows[0].segment.segment_id)[1] == 2


async def test_fixture_and_live_stub_search_parity() -> None:
    store, video, query = await _indexed_store()
    live_store = FixtureEvidenceStore()
    transport = FixtureBackedVastTransport(live_store)
    live = LiveVastEvidenceStore(
        enabled=True,
        endpoint="https://example.invalid",
        token=SecretStr("unused"),
        transport=transport,
        sleep=SleepLog(),
    )
    segments = list(store.indexed_snapshot().values())
    embeddings = []
    understanding = FixtureVideoUnderstanding(DEMO, DEMO / "manifest.json")
    descriptions = {
        parse_segment_id(row["segment_id"])[1]: row
        for row in json.loads((CONTRACTS / "segment_descriptions.json").read_text(encoding="utf-8"))
    }
    from probity.domain.models import SegmentDescription

    for segment in segments:
        desc = SegmentDescription.model_validate(descriptions[segment.ordinal])
        embeddings.append((await understanding.embed_segments([desc]))[0])
    await live.upsert_segments(segments, embeddings)
    request = SearchRequest(
        video_id=video.video_id,
        source_sha256=video.sha256,
        query_embedding=query,
        top_k=12,
        subject_classes=("car", "license_plate"),
    )
    left = await store.search(request)
    right = await live.search(request)
    assert [_normalize(row.model_dump(mode="json")) for row in left] == [
        _normalize(row.model_dump(mode="json")) for row in right
    ]


async def test_malformed_search_fails_closed() -> None:
    sleeper = SleepLog()
    live = LiveVastEvidenceStore(
        enabled=True,
        endpoint="https://example.invalid",
        token=SecretStr("unused"),
        transport=MalformedTransport(),
        sleep=sleeper,
    )
    store, video, query = await _indexed_store()
    _ = store
    request = SearchRequest(
        video_id=video.video_id, source_sha256=video.sha256, query_embedding=query
    )
    with pytest.raises(SponsorMalformedResponse):
        await live.search(request)
    assert sleeper.delays == []


async def test_read_retries_sleep_2_then_5() -> None:
    sleeper = SleepLog()
    live = LiveVastEvidenceStore(
        enabled=True,
        endpoint="https://example.invalid",
        token=SecretStr("unused"),
        transport=TimeoutTransport(),
        sleep=sleeper,
    )
    store, video, query = await _indexed_store()
    _ = store
    with pytest.raises(SponsorTimeout):
        await live.search(
            SearchRequest(
                video_id=video.video_id,
                source_sha256=video.sha256,
                query_embedding=query,
            )
        )
    assert sleeper.delays == [2.0, 5.0]


async def test_write_retries_once_when_uncommitted() -> None:
    transport = OnceUncommitted()
    live = LiveVastEvidenceStore(
        enabled=True,
        endpoint="https://example.invalid",
        token=SecretStr("unused"),
        transport=transport,
        sleep=SleepLog(),
    )
    store, video, query = await _indexed_store()
    _ = query
    segments = list(store.indexed_snapshot().values())[:1]
    embeddings = [
        Embedding(
            embedding_ref="demo-plate-90s/embeddings.npy#0",
            model_id=EMBED_MODEL_ID,
            dimension=64,
            vector=(
                await FixtureVideoUnderstanding(DEMO, DEMO / "manifest.json").embed_query("road")
            ).vector,
            normalized=True,
            input_sha256="a" * 64,
            mode=segments[0].mode,
        )
    ]
    # input_sha256 must match vector's actual hash? Embedding doesn't check that.
    await live.upsert_segments(segments, embeddings)
    assert transport.attempts == 2
    _ = video


async def test_disabled_health() -> None:
    live = LiveVastEvidenceStore(enabled=False, endpoint=None, token=None)
    health = await live.health()
    assert health.status is HealthStatus.DISABLED
    with pytest.raises(SponsorUnavailable):
        await live.search(
            SearchRequest.model_validate(
                {
                    "video_id": "01a12164-8fe8-7cab-9c96-17404faf2b24",
                    "source_sha256": (
                        "9806895754d3af16f50ba66a99514f4b7c1b6659b278fdf7f8260cfd12bf05e5"
                    ),
                    "query_embedding": (
                        await FixtureVideoUnderstanding(DEMO, DEMO / "manifest.json").embed_query(
                            "x"
                        )
                    ).model_dump(mode="json"),
                }
            )
        )


async def test_auto_forced_failure_falls_back_without_mutation() -> None:
    fixture, video, query = await _indexed_store()
    live = LiveVastEvidenceStore(
        enabled=True,
        endpoint="https://example.invalid",
        token=SecretStr("unused"),
        transport=TimeoutTransport(),
        sleep=SleepLog(),
    )
    wrapped = FallbackEvidenceStore(live, fixture, OperatingMode.AUTO)
    request = SearchRequest(
        video_id=video.video_id,
        source_sha256=video.sha256,
        query_embedding=query,
        top_k=12,
        subject_classes=("car", "license_plate"),
    )
    before = snapshot_value(request)
    rows = await wrapped.search(request)
    assert snapshot_value(request) == before
    expected = await fixture.search(request)
    assert [row.segment.segment_id for row in rows] == [row.segment.segment_id for row in expected]
    assert wrapped.mode is AdapterMode.FIXTURE


class _HealthOnceTransport:
    def __init__(self, *, succeed_writes: int) -> None:
        self.health_calls = 0
        self.writes = 0
        self.succeed_writes = succeed_writes

    async def execute(self, operation: str, payload: dict[str, Any]) -> Any:
        if operation == "health":
            self.health_calls += 1
            return {"ok": True}
        if operation == "put_source":
            self.writes += 1
            if self.writes <= self.succeed_writes:
                return str(payload["video"]["storage_uri"])
            raise TimeoutError
        raise TimeoutError


class _CallLogStore(FixtureEvidenceStore):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def put_source(self, video: SourceVideo, local_path: str) -> str:
        self.calls.append("put_source")
        return await super().put_source(video, local_path)


def _source_video() -> SourceVideo:
    raw = (CONTRACTS / "source_video.json").read_text(encoding="utf-8")
    return SourceVideo.model_validate_json(raw)


async def test_auto_health_resolves_once_and_live_write_does_not_continue_on_fixture() -> None:
    video = _source_video()
    transport = _HealthOnceTransport(succeed_writes=1)
    fixture = _CallLogStore()
    live = LiveVastEvidenceStore(
        enabled=True,
        endpoint="https://example.invalid",
        token=SecretStr("unused"),
        transport=transport,
        sleep=SleepLog(),
    )
    wrapped = FallbackEvidenceStore(live, fixture, OperatingMode.AUTO)
    stored = await wrapped.put_source(video, "ignored.mp4")
    assert stored == video.storage_uri
    assert transport.health_calls == 1
    assert fixture.calls == []
    assert wrapped.mode is AdapterMode.LIVE
    with pytest.raises(SponsorTimeout):
        await wrapped.put_source(video, "ignored.mp4")
    assert fixture.calls == []
    assert transport.health_calls == 1
    assert transport.writes == 2
    assert wrapped.mode is AdapterMode.LIVE


async def test_auto_uncommitted_write_replays_same_request_on_fixture() -> None:
    video = _source_video()
    transport = _HealthOnceTransport(succeed_writes=0)
    fixture = _CallLogStore()
    live = LiveVastEvidenceStore(
        enabled=True,
        endpoint="https://example.invalid",
        token=SecretStr("unused"),
        transport=transport,
        sleep=SleepLog(),
    )
    wrapped = FallbackEvidenceStore(live, fixture, OperatingMode.AUTO)
    stored = await wrapped.put_source(video, "ignored.mp4")
    assert stored == video.storage_uri
    assert fixture.calls == ["put_source"]
    assert wrapped.mode is AdapterMode.FIXTURE
    assert transport.health_calls == 1
    again = await wrapped.put_source(video, "ignored.mp4")
    assert again == video.storage_uri
    assert fixture.calls == ["put_source", "put_source"]
    assert transport.health_calls == 1
    assert transport.writes == 1
