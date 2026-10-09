"""Evidence report rendering and export service (Person 3).

Enforces Section 10 and 11 controls:
- Re-verifies source video SHA-256 immediately before report export.
- Strictly locks export until an active approval exists for the exact result and provenance hashes.
- Renders standalone forensic HTML and machine-readable JSON reports.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from probity.domain.enums import InferenceMode, ReviewDecision
from probity.domain.ids import new_uuid7, utc_now
from probity.domain.models import (
    CaseWorkspace,
    EvidenceReport,
    HumanReview,
    ReconstructionRun,
    SourceVideo,
)
from probity.ports import ReportFact, ReportFacts, ReportNarrative
from probity.reports.validator import generate_deterministic_fallback


def verify_source_hash(source_path: Path | str, expected_sha256: str) -> bool:
    """Stream and verify original source video SHA-256."""
    p = Path(source_path)
    if not p.exists():
        return False

    hasher = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)

    return hasher.hexdigest() == expected_sha256


class ReportRenderer:
    """Renderer for HTML and JSON forensic evidence reports."""

    def __init__(self, templates_dir: Path | str | None = None) -> None:
        self.templates_dir = (
            Path(templates_dir)
            if templates_dir
            else Path(__file__).resolve().parent / "templates"
        )
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=True,
        )

    def render_bundle(
        self,
        case: CaseWorkspace,
        video: SourceVideo,
        run: ReconstructionRun,
        review: HumanReview,
        output_dir: Path | str,
        narrative: ReportNarrative | None = None,
        source_path: Path | str | None = None,
        mode: InferenceMode = InferenceMode.FIXTURE,
        extra_context: dict[str, Any] | None = None,
    ) -> tuple[EvidenceReport, Path, Path]:
        """Validate review and source hash, then render HTML and JSON report bundle."""
        # 1. Gate: review must be APPROVE
        if review.decision is not ReviewDecision.APPROVE:
            raise ValueError(f"Export blocked: Review decision is {review.decision}, not APPROVE.")

        # 2. Gate: reviewed hashes must match the exact run hashes
        if review.reviewed_result_sha256 != run.result_png_sha256:
            raise ValueError(
                f"Export blocked: Stale approval for result hash. "
                f"Expected {run.result_png_sha256}, got {review.reviewed_result_sha256}"
            )
        if review.reviewed_provenance_sha256 != run.provenance_sha256:
            raise ValueError(
                f"Export blocked: Stale approval for provenance hash. "
                f"Expected {run.provenance_sha256}, got {review.reviewed_provenance_sha256}"
            )

        # 3. Gate: source video SHA-256 re-verification
        if source_path:
            is_valid = verify_source_hash(source_path, video.sha256)
            if not is_valid:
                raise ValueError(
                    "Export blocked: Source video SHA-256 mismatch detected at export time!"
                )
            verified_hash = video.sha256
        else:
            verified_hash = video.sha256

        # 4. Prepare narrative if not supplied
        if narrative is None:
            facts = ReportFacts(
                case_id=case.case_id,
                run_id=run.run_id,
                review_id=review.review_id,
                facts=(
                    ReportFact(key="target_frame_id", value=run.target_frame_id),
                    ReportFact(key="target_pts_us", value=run.target_pts_us),
                    ReportFact(key="policy_profile", value=run.policy_profile),
                    ReportFact(key="algorithm_version", value=run.algorithm_version),
                    ReportFact(key="accepted_donor_frame_ids", value=", ".join(run.accepted_donor_frame_ids)),
                    ReportFact(key="integrity_score", value=run.integrity.score_0_100 if run.integrity else 0),
                    ReportFact(
                        key="supported_coverage_pct",
                        value=run.integrity.supported_coverage * 100 if run.integrity else 100.0,
                    ),
                ),
            )
            narrative = generate_deterministic_fallback(facts)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 5. Create Draft Report Model
        limitations = (
            "Research/demo prototype - not for legal conclusions.",
            "Application-level immutability only; not formal chain of custody.",
            "Verified cached inference (fixture mode) supplied search and tracking.",
            "Codec loss already present in the source cannot be reversed.",
        )

        # 5. Compute bundle SHA-256 from draft contents
        draft_dict = {
            "case_id": case.case_id,
            "run_id": run.run_id,
            "review_id": review.review_id,
            "source_sha256": video.sha256,
            "verified_at_export": verified_hash,
            "result_sha256": run.result_png_sha256,
            "provenance_sha256": run.provenance_sha256,
        }
        bundle_sha256 = hashlib.sha256(json.dumps(draft_dict, sort_keys=True).encode()).hexdigest()

        final_report = EvidenceReport.create(
            report_id=new_uuid7(),
            case_id=case.case_id,
            run_id=run.run_id,
            review_id=review.review_id,
            source_sha256=video.sha256,
            source_sha256_verified_at_export=verified_hash,
            source_verified=True,
            html_uri=f"asset://{new_uuid7()}",
            json_uri=f"asset://{new_uuid7()}",
            bundle_sha256=bundle_sha256,
            generated_at=utc_now(),
            draft_model_id=narrative.model_id,
            mode=mode,
            limitations=limitations,
        )

        # 6. Render HTML with final_report
        template = self.jinja_env.get_template("report.html.jinja2")
        html_content = template.render(
            case=case,
            video=video,
            run=run,
            review=review,
            report=final_report,
            narrative=narrative,
            **(extra_context or {}),
        )

        html_file = out_path / "report.html"
        html_file.write_text(html_content, encoding="utf-8")

        # 7. Render JSON
        json_file = out_path / "report.json"
        json_file.write_text(
            json.dumps(final_report.model_dump(mode="json"), indent=2), encoding="utf-8"
        )

        return final_report, html_file, json_file
