"""Cosmos fixture vs live-stub contract: parity, malformed fail-closed, retries, fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from probity.adapters.fallback import FallbackVideoUnderstanding, snapshot_value
from probity.adapters.fixture.cosmos import FixtureVideoUnderstanding
from probity.adapters.live.cosmos import LiveCosmosUnderstanding
from probity.domain.enums import AdapterMode, HealthStatus, OperatingMode
from probity.domain.errors import (
    FixtureNotFound,
    SponsorMalformedResponse,
    SponsorTimeout,
    SponsorUnavailable,
)
from probity.domain.prompts import COSMOS_INGESTION_PROMPT
from probity.ports import ClipReference

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "fixtures/demo"
CONTRACTS = ROOT / "fixtures/contracts"
SOURCE_SHA = "9806895754d3af16f50ba66a99514f4b7c1b6659b278fdf7f8260cfd12bf05e5"
VIDEO_ID = "01a12164-8fe8-7cab-9c96-17404faf2b24"

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _clip(ordinal: int = 2) -> ClipReference:
    descriptions = json.loads((CONTRACTS / "segment_descriptions.json").read_text(encoding="utf-8"))
    row = descriptions[ordinal]
    return ClipReference(
        video_id=VIDEO_ID,
        segment_id=row["segment_id"],
        ordinal=ordinal,
        source_sha256=SOURCE_SHA,
        storage_uri=f"source/sha256/{SOURCE_SHA[:2]}/{SOURCE_SHA}/original.mp4",
        start_pts_us=row["time_range"]["start_pts_us"],
        end_pts_us=row["time_range"]["end_pts_us"],
    )


def _fixture() -> FixtureVideoUnderstanding:
    return FixtureVideoUnderstanding(DEMO, DEMO / "manifest.json")


class FixtureBackedCosmosTransport:
    def __init__(self, inner: FixtureVideoUnderstanding) -> None:
        self.inner = inner

    async def execute(self, operation: str, payload: dict[str, Any]) -> Any:
        if operation == "health":
            return {"ok": True}
        if operation == "describe":
            clip = ClipReference.model_validate(payload["clip"])
            result = await self.inner.describe(clip, payload["prompt"])
            return result.model_dump(mode="json")
        if operation == "embed_segments":
            from probity.domain.models import SegmentDescription

            items = [SegmentDescription.model_validate(row) for row in payload["items"]]
            embedded = await self.inner.embed_segments(items)
            return embedded[0].model_dump(mode="json")
        if operation == "embed_query":
            embedded = await self.inner.embed_query(str(payload["query"]))
            return embedded.model_dump(mode="json")
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
        return {"not": "a segment description"}


class SleepLog:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
    skip = {"mode", "adapter_name", "checked_at", "latency_ms", "detail"}
    return {key: value for key, value in payload.items() if key not in skip}


async def test_fixture_health_ok_when_hashes_verify() -> None:
    health = await _fixture().health()
    assert health.status is HealthStatus.OK
    assert health.mode is AdapterMode.FIXTURE
    assert health.adapter_name == "cosmos-fixture"


async def test_describe_requires_exact_prompt() -> None:
    with pytest.raises(Exception, match="exact Cosmos ingestion prompt"):
        await _fixture().describe(_clip(), "a different prompt")


async def test_unknown_source_is_fixture_not_found() -> None:
    clip = _clip()
    bad = ClipReference(
        video_id=clip.video_id,
        segment_id=clip.segment_id,
        ordinal=clip.ordinal,
        source_sha256="0" * 64,
        storage_uri=clip.storage_uri,
        start_pts_us=clip.start_pts_us,
        end_pts_us=clip.end_pts_us,
    )
    with pytest.raises(FixtureNotFound):
        await _fixture().describe(bad, COSMOS_INGESTION_PROMPT)


async def test_fixture_and_live_stub_parity() -> None:
    fixture = _fixture()
    live = LiveCosmosUnderstanding(
        enabled=True,
        endpoint="https://example.invalid",
        token=SecretStr("unused"),
        transport=FixtureBackedCosmosTransport(fixture),
        model_id="fixture/cosmos-reason-describe-v1",
        sleep=SleepLog(),
    )
    clip = _clip(2)
    left = await fixture.describe(clip, COSMOS_INGESTION_PROMPT)
    right = await live.describe(clip, COSMOS_INGESTION_PROMPT)
    assert _normalize(left.model_dump(mode="json")) == _normalize(right.model_dump(mode="json"))
    q_left = await fixture.embed_query("blue sedan rear plate")
    q_right = await live.embed_query("blue sedan rear plate")
    assert _normalize(q_left.model_dump(mode="json")) == _normalize(q_right.model_dump(mode="json"))


async def test_malformed_fails_closed_without_retry() -> None:
    sleeper = SleepLog()
    live = LiveCosmosUnderstanding(
        enabled=True,
        endpoint="https://example.invalid",
        token=SecretStr("unused"),
        transport=MalformedTransport(),
        sleep=sleeper,
    )
    with pytest.raises(SponsorMalformedResponse):
        await live.describe(_clip(), COSMOS_INGESTION_PROMPT)
    assert sleeper.delays == []


async def test_read_retries_sleep_2_then_5() -> None:
    sleeper = SleepLog()
    live = LiveCosmosUnderstanding(
        enabled=True,
        endpoint="https://example.invalid",
        token=SecretStr("unused"),
        transport=TimeoutTransport(),
        sleep=sleeper,
    )
    with pytest.raises(SponsorTimeout):
        await live.embed_query("blue sedan")
    assert sleeper.delays == [2.0, 5.0]


async def test_disabled_health_and_calls() -> None:
    live = LiveCosmosUnderstanding(enabled=False, endpoint=None, token=None)
    health = await live.health()
    assert health.status is HealthStatus.DISABLED
    assert health.mode is AdapterMode.DISABLED
    with pytest.raises(SponsorUnavailable):
        await live.embed_query("blue sedan")


async def test_auto_forced_failure_falls_back_without_mutation() -> None:
    fixture = _fixture()
    sleeper = SleepLog()
    live = LiveCosmosUnderstanding(
        enabled=True,
        endpoint="https://example.invalid",
        token=SecretStr("unused"),
        transport=TimeoutTransport(),
        sleep=sleeper,
    )
    wrapped = FallbackVideoUnderstanding(live, fixture, OperatingMode.AUTO)
    clip = _clip(1)
    before = snapshot_value(clip)
    prompt = COSMOS_INGESTION_PROMPT
    result = await wrapped.describe(clip, prompt)
    assert snapshot_value(clip) == before
    expected = await fixture.describe(clip, prompt)
    assert result == expected
    assert sleeper.delays == [2.0, 5.0]
    assert wrapped.mode is AdapterMode.FIXTURE


async def test_live_mode_never_falls_back() -> None:
    fixture = _fixture()
    live = LiveCosmosUnderstanding(
        enabled=True,
        endpoint="https://example.invalid",
        token=SecretStr("unused"),
        transport=TimeoutTransport(),
        sleep=SleepLog(),
    )
    wrapped = FallbackVideoUnderstanding(live, fixture, OperatingMode.LIVE)
    with pytest.raises(SponsorTimeout):
        await wrapped.describe(_clip(), COSMOS_INGESTION_PROMPT)
