"""Unit tests for Person 3 W&B reasoner, Weave sink, and hallucination validator."""

import pytest

from probity.adapters.fixture.wandb import WandbFixtureAdapter
from probity.adapters.fixture.weave import WeaveFixtureAdapter, sanitize_attributes
from probity.adapters.live.wandb import WandbLiveAdapter
from probity.adapters.live.weave import WeaveLiveAdapter
from probity.domain.enums import AdapterMode, ExplanationSource, HealthStatus, ReasonCode
from probity.domain.models import RetrievedSegment, VideoSegment
from probity.ports import QueryPlanRequest, ReportFact, ReportFacts, ReportNarrative
from probity.reports.validator import (
    generate_deterministic_fallback,
    validate_or_fallback_narrative,
    validate_paragraph_against_facts,
)


@pytest.fixture
def sample_report_facts() -> ReportFacts:
    return ReportFacts(
        case_id="01a12164-8c00-7bfc-9416-0a1e8674dac6",
        run_id="01a12169-1fe0-739f-a1f0-04111b204a6b",
        review_id="01a1216c-2d20-7f23-8ae4-086d01307eb7",
        facts=(
            ReportFact(key="target_frame_id", value="01a12164-8fe8-7cab-9c96-17404faf2b24:f417", citations=("frame 417",)),
            ReportFact(key="target_pts_us", value=13900000, citations=("13.9s", "13900000")),
            ReportFact(key="policy_profile", value="demo-conservative-v1", citations=()),
            ReportFact(key="algorithm_version", value="probity-tile-v1", citations=()),
            ReportFact(key="accepted_donor_frame_ids", value="f409, f424", citations=("f409", "f424")),
            ReportFact(key="integrity_score", value=94, citations=("94/100",)),
            ReportFact(key="supported_coverage_pct", value=100.0, citations=("100.0%",)),
        ),
    )


def test_validator_detects_uncited_timestamp(sample_report_facts: ReportFacts) -> None:
    # 13.9s is cited, but 25.5s is not
    valid_text = "Target observed near 13.9s was processed."
    ok, _ = validate_paragraph_against_facts(valid_text, sample_report_facts)
    assert ok is True

    invalid_text = "The vehicle was moving at 25.5s."
    ok, reason = validate_paragraph_against_facts(invalid_text, sample_report_facts)
    assert ok is False
    assert "Uncited timestamp" in str(reason)


def test_validator_rejects_legal_conclusions(sample_report_facts: ReportFacts) -> None:
    bad_text = "This constitutes conclusive proof of guilty behavior."
    ok, reason = validate_paragraph_against_facts(bad_text, sample_report_facts)
    assert ok is False
    assert "Disallowed conclusive or legal terminology" in str(reason)


def test_validator_fallback_replaces_hallucinated_prose(sample_report_facts: ReportFacts) -> None:
    hallucinated = ReportNarrative(
        paragraphs=("The suspect entered the restricted area at 45.2s.",),
        model_id="wandb-llm",
        source=ExplanationSource.REASONER,
    )
    result = validate_or_fallback_narrative(hallucinated, sample_report_facts)
    assert result.source is ExplanationSource.TEMPLATE
    assert len(result.paragraphs) == 2
    assert "94/100" in result.paragraphs[1]


@pytest.mark.anyio
async def test_wandb_fixture_adapter(sample_report_facts: ReportFacts) -> None:
    adapter = WandbFixtureAdapter()
    assert adapter.mode is AdapterMode.FIXTURE

    health = await adapter.health()
    assert health.status is HealthStatus.OK

    # Query planning with policy filtering
    plan = await adapter.plan_query(
        QueryPlanRequest(
            query="Find the stolen blue sedan",
            video_id="01a12164-8fe8-7cab-9c96-17404faf2b24",
            allowed_subject_classes=("car", "license_plate"),
        )
    )
    assert ReasonCode.QUERY_POLICY_FILTERED in plan.policy_reason_codes
    assert "criminal_intent" in plan.filtered_terms

    # Draft report
    narrative = await adapter.draft_report(sample_report_facts)
    assert narrative.source is ExplanationSource.TEMPLATE
    assert "probity-tile-v1" in narrative.paragraphs[0]


@pytest.mark.anyio
async def test_wandb_live_adapter_offline_graceful_fallback(sample_report_facts: ReportFacts) -> None:
    adapter = WandbLiveAdapter(api_key=None)
    health = await adapter.health()
    assert health.status is HealthStatus.UNAVAILABLE
    assert health.mode is AdapterMode.DEGRADED

    # Without API key, falls back seamlessly to fixture
    plan = await adapter.plan_query(
        QueryPlanRequest(
            query="Find the blue sedan",
            video_id="01a12164-8fe8-7cab-9c96-17404faf2b24",
            allowed_subject_classes=("car", "license_plate"),
        )
    )
    assert plan.semantic_query == "Find the blue sedan"


def test_weave_attribute_redaction() -> None:
    raw_attrs = {
        "correlation_id": "01a12164-8c00-7bfc-9416-0a1e8674dac6",
        "video_bytes": b"\x00\x00\x00\x18ftypmp42",
        "frame_image": "data:image/png;base64,iVBORw0KGgo...",
        "crop_donor_px": b"image_pixels",
        "api_key": "secret_key_12345",
        "analyst_notes": "Private suspect notes",
        "score": 94,
        "mode": "FIXTURE",
    }
    clean = sanitize_attributes(raw_attrs)
    assert "correlation_id" in clean
    assert "score" in clean
    assert "mode" in clean
    assert "video_bytes" not in clean
    assert "frame_image" not in clean
    assert "crop_donor_px" not in clean
    assert "api_key" not in clean
    assert "analyst_notes" not in clean


def test_weave_fixture_span_recording() -> None:
    sink = WeaveFixtureAdapter()
    with sink.span("search.plan", {"correlation_id": "c1", "query_hash": "abc", "secret_token": "hidden"}):
        pass

    assert len(sink.spans) == 1
    assert sink.spans[0]["span_name"] == "search.plan"
    assert "secret_token" not in sink.spans[0]["attributes"]
    assert sink.spans[0]["attributes"]["correlation_id"] == "c1"
