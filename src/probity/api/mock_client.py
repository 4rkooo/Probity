"""Mock API client for Person 3 UI development and smoke/E2E testing.

Loads frozen fixtures from ``fixtures/contracts/`` and ``fixtures/demo/`` to provide
a realistic, stateful, offline-first client that satisfies the API contracts without
requiring Person 1's backend worker or Person 2's live reconstructor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from probity.domain.enums import (
    JobKind,
    JobStage,
    JobState,
    ProvenanceClass,
    ReasonCode,
    ReviewDecision,
    TrackState,
    VetoReason,
)
from probity.domain.ids import new_uuid7
from probity.domain.models import (
    CaseWorkspace,
    EvidenceReport,
    HumanReview,
    JobView,
    PixelOrigin,
    PolicyDecision,
    ReconstructionRun,
    SearchEvidence,
    SourceVideo,
    Track,
    VideoSegment,
)


class MockApiClient:
    """Stateful mock API client for the analyst UI workflow."""

    def __init__(self, repo_root: Path | str | None = None) -> None:
        self.root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
        self.contracts_dir = self.root / "fixtures" / "contracts"
        self.artifacts_dir = self.contracts_dir / "artifacts"
        self.demo_source_dir = self.root / "fixtures" / "demo" / "source"

        self._load_fixtures()
        self._reviews: dict[str, HumanReview] = {}
        self._reports: dict[str, EvidenceReport] = {}

    def _load_fixtures(self) -> None:
        # Load baseline contract entities
        with (self.contracts_dir / "case_workspace.json").open() as f:
            self.case = CaseWorkspace.model_validate_json(f.read())

        with (self.contracts_dir / "source_video.json").open() as f:
            self.source_video = SourceVideo.model_validate_json(f.read())

        with (self.contracts_dir / "video_segments.json").open() as f:
            self.segments = [VideoSegment.model_validate(s) for s in json.load(f)]

        with (self.contracts_dir / "search_evidence.json").open() as f:
            self.search_evidence = SearchEvidence.model_validate_json(f.read())

        with (self.contracts_dir / "search_evidence_needs_clarification.json").open() as f:
            self.search_needs_clarification = SearchEvidence.model_validate_json(f.read())

        with (self.contracts_dir / "track_confirmed.json").open() as f:
            self.track_confirmed = Track.model_validate_json(f.read())

        with (self.contracts_dir / "track_not_confirmed.json").open() as f:
            self.track_not_confirmed = Track.model_validate_json(f.read())

        with (self.contracts_dir / "reconstruction_run_succeeded.json").open() as f:
            self.run_succeeded = ReconstructionRun.model_validate_json(f.read())

        with (self.contracts_dir / "reconstruction_run_refused.json").open() as f:
            self.run_refused = ReconstructionRun.model_validate_json(f.read())

        with (self.contracts_dir / "policy_decisions_succeeded.json").open() as f:
            self.decisions_succeeded = [PolicyDecision.model_validate(d) for d in json.load(f)]

        with (self.contracts_dir / "policy_decisions_refused.json").open() as f:
            self.decisions_refused = [PolicyDecision.model_validate(d) for d in json.load(f)]

        with (self.contracts_dir / "job_views.json").open() as f:
            self.job_views = [JobView.model_validate(j) for j in json.load(f)]

        with (self.contracts_dir / "evidence_report.json").open() as f:
            self.fixture_report = EvidenceReport.model_validate_json(f.read())

        # Load provenance NPZ
        npz_path = self.artifacts_dir / "provenance.npz"
        if npz_path.exists():
            self._npz_data = np.load(npz_path)
        else:
            self._npz_data = None

    # -----------------------------------------------------------------------------------------
    # Case management
    # -----------------------------------------------------------------------------------------

    def get_case(self, case_id: str | None = None) -> CaseWorkspace:
        return self.case

    def create_case(
        self, display_name: str, purpose: str = "DEMO_RESEARCH", owner_alias: str = "analyst-1"
    ) -> CaseWorkspace:
        self.case = self.case.revise(display_name=display_name, owner_alias=owner_alias)
        return self.case

    # -----------------------------------------------------------------------------------------
    # Videos and Ingestion
    # -----------------------------------------------------------------------------------------

    def get_video(self, video_id: str | None = None) -> SourceVideo:
        return self.source_video

    def list_segments(self, video_id: str | None = None) -> list[VideoSegment]:
        return list(self.segments)

    def start_ingest_job(self, video_id: str) -> JobView:
        for j in self.job_views:
            if j.kind == JobKind.INGEST and j.state == JobState.RUNNING:
                return j
        return self.job_views[0]

    def get_job(self, job_id: str) -> JobView:
        for j in self.job_views:
            if j.job_id == job_id:
                return j
        # Fallback to succeeded ingest job
        for j in self.job_views:
            if j.state == JobState.SUCCEEDED:
                return j
        return self.job_views[0]

    def cancel_job(self, job_id: str) -> JobView:
        for j in self.job_views:
            if j.state == JobState.CANCELLED:
                return j
        return self.job_views[0].revise(
            state=JobState.CANCELLED, stage=JobStage.EXTRACT, completed_units=0
        )

    # -----------------------------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------------------------

    def search(self, query: str, max_results: int = 5) -> SearchEvidence:
        cleaned = query.strip()
        if not cleaned:
            return self.search_needs_clarification
        return self.search_evidence

    # -----------------------------------------------------------------------------------------
    # Tracking
    # -----------------------------------------------------------------------------------------

    def get_track(self, confirmed: bool = True) -> Track:
        return self.track_confirmed if confirmed else self.track_not_confirmed

    # -----------------------------------------------------------------------------------------
    # Reconstruction & Provenance
    # -----------------------------------------------------------------------------------------

    def get_reconstruction(self, run_id: str | None = None, refused: bool = False) -> ReconstructionRun:
        return self.run_refused if refused else self.run_succeeded

    def list_decisions(self, run_id: str | None = None, refused: bool = False) -> list[PolicyDecision]:
        return self.decisions_refused if refused else self.decisions_succeeded

    def resolve_pixel(self, x: int, y: int, run_id: str | None = None) -> PixelOrigin | None:
        """Resolve pixel origin from provenance NPZ and run source LUT."""
        if self._npz_data is None:
            return None

        h, w = self._npz_data["class"].shape
        if not (0 <= x < w and 0 <= y < h):
            return None

        cls_val = int(self._npz_data["class"][y, x])
        src_idx = int(self._npz_data["source_index"][y, x])
        src_x = float(self._npz_data["source_x"][y, x])
        src_y = float(self._npz_data["source_y"][y, x])

        cls_name = ProvenanceClass(cls_val).name
        run = self.run_succeeded
        lut = run.provenance.source_lut if run.provenance else ()
        entry = lut[src_idx] if src_idx < len(lut) else None

        if entry is None:
            return None

        return PixelOrigin(
            run_id=run.run_id,
            x=x,
            y=y,
            provenance_class=cls_name,  # type: ignore[arg-type]
            source_index=src_idx,
            source_frame_id=entry.frame_id,
            source_pts_us=entry.pts_us,
            source_x=src_x,
            source_y=src_y,
            transform_id=entry.transform_id,
            alignment_method=entry.alignment_method,
            color_gain=entry.color_gain,
            color_bias=entry.color_bias,
            decision_ids=entry.decision_ids,
        )

    # -----------------------------------------------------------------------------------------
    # Human Review Gate
    # -----------------------------------------------------------------------------------------

    def get_latest_review(self, run_id: str) -> HumanReview | None:
        return self._reviews.get(run_id)

    def submit_review(
        self,
        run_id: str,
        decision: Literal["APPROVE", "VETO"],
        reviewer_alias: str,
        comment: str,
        reviewed_result_sha256: str,
        reviewed_provenance_sha256: str,
        veto_reason: VetoReason | None = None,
    ) -> HumanReview:
        run = self.get_reconstruction(run_id)

        # Enforce exact hash binding
        if run.state is not JobState.SUCCEEDED and run.state.name != "SUCCEEDED":
            if decision == "APPROVE":
                raise ValueError("Cannot approve a run that did not SUCCEED.")

        if decision == "APPROVE":
            if run.result_png_sha256 != reviewed_result_sha256:
                raise ValueError(
                    f"Result hash mismatch! Expected {run.result_png_sha256}, got {reviewed_result_sha256}"
                )
            if run.provenance_sha256 != reviewed_provenance_sha256:
                raise ValueError(
                    f"Provenance hash mismatch! Expected {run.provenance_sha256}, got {reviewed_provenance_sha256}"
                )
            if run.integrity and run.integrity.score_0_100 < 70:
                raise ValueError("Integrity score below 70; approval disallowed.")

            review = HumanReview.create(
                review_id=new_uuid7(),
                run_id=run.run_id,
                reviewer_alias=reviewer_alias,
                decision=ReviewDecision.APPROVE,
                reason_code=None,
                comment=comment,
                reviewed_result_sha256=reviewed_result_sha256,
                reviewed_provenance_sha256=reviewed_provenance_sha256,
            )
        else:
            if veto_reason is None:
                raise ValueError("Veto reason code is required when vetoing.")
            if not comment.strip():
                raise ValueError("Veto comment cannot be empty.")

            review = HumanReview.create(
                review_id=new_uuid7(),
                run_id=run.run_id,
                reviewer_alias=reviewer_alias,
                decision=ReviewDecision.VETO,
                reason_code=veto_reason,
                comment=comment,
                reviewed_result_sha256=reviewed_result_sha256,
                reviewed_provenance_sha256=reviewed_provenance_sha256,
            )

        self._reviews[run_id] = review
        return review

    # -----------------------------------------------------------------------------------------
    # Evidence Reports
    # -----------------------------------------------------------------------------------------

    def create_report(self, run_id: str) -> EvidenceReport:
        review = self._reviews.get(run_id)
        if review is None or review.decision != ReviewDecision.APPROVE:
            raise ValueError("Report export is forbidden without an active approval.")

        run = self.get_reconstruction(run_id)
        if (
            run.result_png_sha256 != review.reviewed_result_sha256
            or run.provenance_sha256 != review.reviewed_provenance_sha256
        ):
            raise ValueError("Stale approval! Artifact hashes have changed since review.")

        report = self.fixture_report.revise(
            run_id=run.run_id,
            review_id=review.review_id,
            source_sha256_verified_at_export=self.source_video.sha256,
        )
        self._reports[report.report_id] = report
        return report

    def get_report(self, report_id: str) -> EvidenceReport | None:
        return self._reports.get(report_id, self.fixture_report)

    # -----------------------------------------------------------------------------------------
    # Artifact Assets
    # -----------------------------------------------------------------------------------------

    def get_asset_path(self, name: str) -> Path:
        if (self.artifacts_dir / name).exists():
            return self.artifacts_dir / name
        if (self.demo_source_dir / name).exists():
            return self.demo_source_dir / name
        return self.artifacts_dir / "result.png"
