"""Query-plan validator: identity/legal/intent, multiword, clarification, fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from probity.domain.enums import ReasonCode
from probity.domain.models import SearchPlan
from probity.domain.policy import default_policy
from probity.ports import QueryPlanRequest
from probity.search.plan import RuleBasedPlanner, plan_search_query, sanitize_search_plan

ROOT = Path(__file__).resolve().parents[2]
QUERIES = json.loads((ROOT / "fixtures/demo/search/queries.json").read_text(encoding="utf-8"))
VIDEO_ID = "01a12164-8fe8-7cab-9c96-17404faf2b24"


def _request(query: str) -> QueryPlanRequest:
    return QueryPlanRequest(
        query=query,
        video_id=VIDEO_ID,
        allowed_subject_classes=("car", "truck", "license_plate", "sign"),
    )


def test_identity_terms_are_filtered() -> None:
    policy = default_policy()
    planner = RuleBasedPlanner(policy)
    plan = planner.plan(_request("Identify the driver and name the owner of the blue sedan"))
    assert "driver" in plan.filtered_terms
    assert "name" in plan.filtered_terms
    assert "owner" in plan.filtered_terms
    assert ReasonCode.QUERY_POLICY_FILTERED in plan.policy_reason_codes
    assert "car" in plan.subject_classes
    assert plan.needs_clarification is False


def test_legal_terms_are_filtered() -> None:
    planner = RuleBasedPlanner(default_policy())
    plan = planner.plan(_request("Show the illegal crime involving the red van"))
    assert "illegal" in plan.filtered_terms
    assert "crime" in plan.filtered_terms
    assert "truck" in plan.subject_classes


def test_intent_terms_are_filtered() -> None:
    planner = RuleBasedPlanner(default_policy())
    plan = planner.plan(_request("Why did the blue sedan move with suspicious motive"))
    assert "why" in plan.filtered_terms
    assert "motive" in plan.filtered_terms or "suspicious" in plan.filtered_terms
    assert plan.needs_clarification is False
    assert "car" in plan.subject_classes


def test_multiword_disallowed_term() -> None:
    plan = RuleBasedPlanner(default_policy()).plan(_request("Is this evidence of the blue sedan"))
    assert "evidence of" in plan.filtered_terms
    assert ReasonCode.QUERY_POLICY_FILTERED in plan.policy_reason_codes
    assert "car" in plan.subject_classes


def test_nothing_objective_needs_clarification() -> None:
    planner = RuleBasedPlanner(default_policy())
    plan = planner.plan(_request("Who is the guilty driver and why did they flee?"))
    expected = SearchPlan.model_validate(QUERIES["clarification_query"]["expected_plan"])
    assert plan == expected
    assert plan.needs_clarification is True
    assert plan.semantic_query == ""
    assert plan.objective_terms == ()


def test_prepared_query_plan_matches_golden() -> None:
    plan = RuleBasedPlanner(default_policy()).plan(_request(QUERIES["prepared_query"]["query"]))
    expected = SearchPlan.model_validate(QUERIES["prepared_query"]["expected_plan"])
    assert plan == expected


@pytest.mark.anyio
async def test_invalid_reasoner_plan_falls_back_to_rules() -> None:
    class BadReasoner:
        adapter_name = "wandb-test"
        model_id = "fixture/wandb-planner-v1"
        mode = "LIVE"
        schema_version = "1.0"

        async def health(self):  # pragma: no cover - protocol completeness
            raise NotImplementedError

        async def plan_query(self, request: QueryPlanRequest) -> SearchPlan:
            return SearchPlan(
                semantic_query="see frame 417 of the suspect",
                objective_terms=("suspect",),
                needs_clarification=False,
            )

        async def explain_results(self, query, evidence):  # pragma: no cover
            raise NotImplementedError

        async def draft_report(self, facts):  # pragma: no cover
            raise NotImplementedError

    request = _request(QUERIES["prepared_query"]["query"])
    plan, model_id = await plan_search_query(request, default_policy(), BadReasoner())
    assert model_id == "probity/rule-based-planner-v1"
    assert plan == SearchPlan.model_validate(QUERIES["prepared_query"]["expected_plan"])


@pytest.mark.anyio
async def test_reasoner_plan_is_still_sanitized() -> None:
    class IdentityReasoner:
        adapter_name = "wandb-test"
        model_id = "fixture/wandb-planner-v1"
        mode = "LIVE"
        schema_version = "1.0"

        async def health(self):  # pragma: no cover
            raise NotImplementedError

        async def plan_query(self, request: QueryPlanRequest) -> SearchPlan:
            return SearchPlan(
                semantic_query="guilty driver blue sedan",
                objective_terms=("guilty", "driver", "blue sedan"),
                subject_classes=("car",),
            )

        async def explain_results(self, query, evidence):  # pragma: no cover
            raise NotImplementedError

        async def draft_report(self, facts):  # pragma: no cover
            raise NotImplementedError

    request = _request("guilty driver blue sedan")
    plan, model_id = await plan_search_query(request, default_policy(), IdentityReasoner())
    assert model_id == "fixture/wandb-planner-v1"
    assert "guilty" in plan.filtered_terms
    assert "driver" in plan.filtered_terms
    assert "blue sedan" in plan.objective_terms
    assert ReasonCode.QUERY_POLICY_FILTERED in plan.policy_reason_codes


def test_sanitize_empty_plan_sets_clarification() -> None:
    raw = SearchPlan(semantic_query="who is guilty", objective_terms=("who", "guilty"))
    cleaned = sanitize_search_plan(raw, default_policy().search, "who is guilty")
    assert cleaned.needs_clarification is True
    assert cleaned.semantic_query == ""


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
