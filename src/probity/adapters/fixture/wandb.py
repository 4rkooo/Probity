"""Fixture adapter for W&B-hosted inference (Person 3).

Implements EvidenceReasoner Protocol using deterministic, contract-verified fixture data
for offline testing and cached demo execution.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from probity.domain.enums import (
    AdapterMode,
    ExplanationSource,
    HealthStatus,
    ReasonCode,
)
from probity.domain.ids import utc_now
from probity.domain.models import (
    AdapterHealth,
    GroundedExplanation,
    RetrievedSegment,
    SearchPlan,
    TimeRangeUs,
)
from probity.ports import (
    EvidenceReasoner,
    QueryPlanRequest,
    ReportFacts,
    ReportNarrative,
)
from probity.reports.validator import generate_deterministic_fallback


class WandbFixtureAdapter(EvidenceReasoner):
    """Fixture implementation of W&B inference reasoner."""

    adapter_name: str = "wandb-fixture"
    model_id: str | None = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    mode: AdapterMode = AdapterMode.FIXTURE
    schema_version: str = "1.0"

    async def health(self) -> AdapterHealth:
        return AdapterHealth(
            adapter_name=self.adapter_name,
            model_id=self.model_id,
            mode=self.mode,
            status=HealthStatus.OK,
            schema_version="1.0",
            checked_at=utc_now(),
            latency_ms=1,
            detail="W&B fixture adapter operating in verified cached mode.",
        )

    async def plan_query(self, request: QueryPlanRequest) -> SearchPlan:
        q = request.query.strip().lower()
        if not q:
            return SearchPlan(
                semantic_query="",
                objective_terms=(),
                subject_classes=(),
                time_range=None,
                needs_clarification=True,
                filtered_terms=(),
                policy_reason_codes=(),
            )

        # Check for filtered policy terms (intent/speculation)
        filtered_terms: list[str] = []
        policy_codes: list[ReasonCode] = []
        if any(term in q for term in ("stolen", "suspect", "guilty", "criminal", "fleeing")):
            filtered_terms.append("criminal_intent")
            policy_codes.append(ReasonCode.QUERY_POLICY_FILTERED)

        return SearchPlan(
            semantic_query=request.query,
            objective_terms=("blue sedan", "rear plate") if "sedan" in q or "plate" in q else ("vehicle",),
            subject_classes=("car", "license_plate") if "sedan" in q or "plate" in q else ("car",),
            time_range=TimeRangeUs(start_pts_us=12_000_000, end_pts_us=20_000_000),
            needs_clarification=False,
            filtered_terms=tuple(filtered_terms),
            policy_reason_codes=tuple(policy_codes),
        )

    async def explain_results(
        self, query: str, evidence: Sequence[RetrievedSegment]
    ) -> Sequence[GroundedExplanation]:
        explanations: list[GroundedExplanation] = []
        for item in evidence:
            seg = item.segment
            text = (
                f"Retrieved segment {seg.segment_id} (PTS {seg.start_pts_us}..{seg.end_pts_us} us) "
                f"matches query with similarity {item.similarity:.2f}. Description: {seg.description[:120]}."
            )
            input_hash = hashlib.sha256(f"{query}:{seg.segment_id}".encode()).hexdigest()
            explanations.append(
                GroundedExplanation(
                    segment_id=seg.segment_id,
                    text=text,
                    cited_detection_ids=(),
                    cited_pts_us=(seg.start_pts_us, seg.end_pts_us),
                    source=ExplanationSource.TEMPLATE,
                    model_id=self.model_id,
                    input_sha256=input_hash,
                )
            )
        return tuple(explanations)

    async def draft_report(self, facts: ReportFacts) -> ReportNarrative:
        return generate_deterministic_fallback(facts)
