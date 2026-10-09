"""Review and export gate application service (Person 3).

One place decides whether a reconstruction may be approved and whether an evidence
report may be exported (design sections 9 and 11). The UI, the mock API client, and the
report renderer all consult these functions so that the rules cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from probity.domain.enums import ReconstructionState, ReviewDecision
from probity.domain.models import HumanReview, ReconstructionRun

MIN_INTEGRITY_SCORE = 70


@dataclass(frozen=True)
class GateBlocker:
    """A rule-coded reason that an approval or export is not allowed."""

    code: str
    message: str


def approval_blockers(run: ReconstructionRun, *, source_verified: bool) -> list[GateBlocker]:
    """Return every reason the run cannot be approved; empty means approval is allowed."""
    blockers: list[GateBlocker] = []

    if run.state is not ReconstructionState.SUCCEEDED:
        reasons = ", ".join(str(r) for r in run.refusal_reasons) or "no artifact produced"
        blockers.append(
            GateBlocker(
                code=f"RUN_{run.state}",
                message=f"Run state is {run.state} ({reasons}); only SUCCEEDED runs can be reviewed.",
            )
        )
    if run.result_png_sha256 is None or run.provenance_sha256 is None:
        blockers.append(
            GateBlocker(
                code="ARTIFACT_HASH_CONFLICT",
                message="Result or provenance artifact hash is missing.",
            )
        )
    if run.integrity is None or run.integrity.score_0_100 < MIN_INTEGRITY_SCORE:
        score = run.integrity.score_0_100 if run.integrity else "n/a"
        blockers.append(
            GateBlocker(
                code="INTEGRITY_BELOW_MINIMUM",
                message=f"Integrity score {score} is below the minimum of {MIN_INTEGRITY_SCORE}.",
            )
        )
    if run.provenance is None or not run.provenance.coverage_complete:
        blockers.append(
            GateBlocker(
                code="PROVENANCE_INCOMPLETE",
                message="Not every output pixel resolves to a recorded source.",
            )
        )
    generated = (
        run.provenance is not None and run.provenance.subject_coverage_pct.GENERATED_BLEND > 0
    ) or (run.integrity is not None and run.integrity.semantic_generated_pct > 0)
    if generated:
        blockers.append(
            GateBlocker(
                code="GENERATED_SEMANTIC_PIXEL",
                message="Generated pixels appear inside the semantic subject region.",
            )
        )
    if not source_verified:
        blockers.append(
            GateBlocker(
                code="SOURCE_HASH_MISMATCH",
                message="Source bytes no longer match the ingest SHA-256; the case is quarantined.",
            )
        )
    return blockers


def review_is_current(run: ReconstructionRun, review: HumanReview | None) -> bool:
    """True when the review is bound to the run's exact current artifact hashes."""
    return (
        review is not None
        and review.run_id == run.run_id
        and review.reviewed_result_sha256 == run.result_png_sha256
        and review.reviewed_provenance_sha256 == run.provenance_sha256
    )


def export_blockers(
    run: ReconstructionRun, review: HumanReview | None, *, source_verified: bool
) -> list[GateBlocker]:
    """Return every reason a report cannot be exported; empty means export is allowed."""
    blockers: list[GateBlocker] = []
    if review is None:
        blockers.append(
            GateBlocker(code="REVIEW_REQUIRED", message="No human review exists for this run.")
        )
    elif review.decision is ReviewDecision.VETO:
        blockers.append(
            GateBlocker(
                code="HUMAN_VETOED",
                message=f"Reconstruction was vetoed ({review.reason_code}); export is permanently locked.",
            )
        )
    elif not review_is_current(run, review):
        blockers.append(
            GateBlocker(
                code="STALE_APPROVAL",
                message="Approval was recorded for different artifact hashes; review again.",
            )
        )
    blockers.extend(approval_blockers(run, source_verified=source_verified))
    return blockers
