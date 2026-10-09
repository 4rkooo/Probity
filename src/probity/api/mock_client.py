"""Mock API client for Person 3 UI development and smoke/E2E testing.

Loads frozen fixtures from ``fixtures/contracts/`` and ``fixtures/demo/`` to provide
a realistic, stateful, offline-first client that satisfies the API contracts without
requiring Person 1's backend worker or Person 2's live reconstructor.

Scenario knobs (ingest outcome, reconstruction outcome, source tampering, artifact
regeneration, hallucinated narrative) let the UI exercise every required state against
the same fixtures that drive the happy path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np

from probity.adapters.fixture.wandb import WandbFixtureAdapter
from probity.adapters.fixture.weave import WeaveFixtureAdapter
from probity.domain.enums import (
    ExplanationSource,
    JobKind,
    JobStage,
    JobState,
    ProvenanceClass,
    ReviewDecision,
    SearchStatus,
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
    TimeRangeUs,
    Track,
    VideoSegment,
)
from probity.ports import QueryPlanRequest, ReportFact, ReportFacts, ReportNarrative
from probity.reports.gate import approval_blockers, export_blockers
from probity.reports.validator import validate_or_fallback_narrative

IngestOutcome = Literal["SUCCEEDED", "PARTIAL", "FAILED", "SPONSOR_TIMEOUT"]
RunOutcome = Literal["SUCCEEDED", "REFUSED", "FAILED"]

INGEST_OUTCOMES: tuple[str, ...] = ("SUCCEEDED", "PARTIAL", "FAILED", "SPONSOR_TIMEOUT")
RUN_OUTCOMES: tuple[str, ...] = ("SUCCEEDED", "REFUSED", "FAILED")

# Ordered ingest stages with the cumulative units completed when each stage finishes.
INGEST_STAGE_PLAN: tuple[tuple[JobStage, int], ...] = (
    (JobStage.VALIDATE, 1),
    (JobStage.HASH, 2),
    (JobStage.EXTRACT, 4),
    (JobStage.DESCRIBE, 7),
    (JobStage.EMBED, 10),
    (JobStage.DETECT, 13),
    (JobStage.INDEX, 15),
)

# Vocabulary the fixture index can match; anything else yields NO_RESULTS.
OBSERVABLE_TERMS = (
    "sedan", "car", "vehicle", "plate", "sign", "blue", "rear", "license", "moving", "street",
)

HALLUCINATED_PARAGRAPH = (
    "The blue sedan was driven by the suspect at 00:42 and the plate reads ABC-1234 with "
    "99% confidence, which is conclusive proof of the vehicle's identity."
)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class MockApiClient:
    """Stateful mock API client for the analyst UI workflow."""

    def __init__(self, repo_root: Path | str | None = None) -> None:
        self.root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
        self.contracts_dir = self.root / "fixtures" / "contracts"
        self.artifacts_dir = self.contracts_dir / "artifacts"
        self.demo_source_dir = self.root / "fixtures" / "demo" / "source"

        self._load_fixtures()
        self.reasoner = WandbFixtureAdapter()
        self.reset()

    def reset(self) -> None:
        """Restore all mutable demo state (reviews, scenarios, traces)."""
        self._reviews: dict[str, HumanReview] = {}
        self._reports: dict[str, EvidenceReport] = {}
        self.trace = WeaveFixtureAdapter()
        self.correlation_id = new_uuid7()
        self.ingest_outcome: str = "SUCCEEDED"
        self.run_outcome: str = "SUCCEEDED"
        self.source_tampered = False
        self.hallucinate_narrative = False
        self._artifact_revision = 0
        self.run_succeeded = self._run_succeeded_canonical

    def _load_fixtures(self) -> None:
        def load(name: str) -> str:
            return (self.contracts_dir / name).read_text()

        self.case = CaseWorkspace.model_validate_json(load("case_workspace.json"))
        self.source_video = SourceVideo.model_validate_json(load("source_video.json"))
        self.source_video_partial = SourceVideo.model_validate_json(
            load("source_video_partial.json")
        )
        self.segments = [VideoSegment.model_validate(s) for s in json.loads(load("video_segments.json"))]
        self.search_evidence = SearchEvidence.model_validate_json(load("search_evidence.json"))
        self.search_needs_clarification = SearchEvidence.model_validate_json(
            load("search_evidence_needs_clarification.json")
        )
        self.track_confirmed = Track.model_validate_json(load("track_confirmed.json"))
        self.track_not_confirmed = Track.model_validate_json(load("track_not_confirmed.json"))
        self._run_succeeded_canonical = ReconstructionRun.model_validate_json(
            load("reconstruction_run_succeeded.json")
        )
        self.run_succeeded = self._run_succeeded_canonical
        self.run_refused = ReconstructionRun.model_validate_json(load("reconstruction_run_refused.json"))
        self.run_failed = ReconstructionRun.model_validate_json(load("reconstruction_run_failed.json"))
        self.decisions_succeeded = [
            PolicyDecision.model_validate(d) for d in json.loads(load("policy_decisions_succeeded.json"))
        ]
        self.decisions_refused = [
            PolicyDecision.model_validate(d) for d in json.loads(load("policy_decisions_refused.json"))
        ]
        self.job_views = [JobView.model_validate(j) for j in json.loads(load("job_views.json"))]
        self.fixture_report = EvidenceReport.model_validate_json(load("evidence_report.json"))

        manifest = json.loads((self.root / "fixtures" / "demo" / "manifest.json").read_text())
        self.manifest = manifest["fixtures"][0]
        self.manifest_created_at: str = self.manifest.get("created_at", "unknown")

        npz_path = self.artifacts_dir / "provenance.npz"
        self._npz_data = dict(np.load(npz_path)) if npz_path.exists() else None

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
    # Videos, source verification, and ingestion
    # -----------------------------------------------------------------------------------------

    def get_video(self, video_id: str | None = None) -> SourceVideo:
        return self.source_video

    def source_path(self) -> Path:
        return self.demo_source_dir / self.source_video.original_name

    def verify_source(self) -> tuple[bool, str]:
        """Stream-hash the canonical source now; returns (matches_ingest_hash, observed_sha256)."""
        path = self.source_path()
        if not path.exists():
            return False, "0" * 64
        observed = _sha256_file(path)
        if self.source_tampered:
            # Simulates altered bytes on disk without touching the real fixture.
            observed = hashlib.sha256(observed.encode() + b"tampered").hexdigest()
        return observed == self.source_video.sha256, observed

    def list_segments(self, video_id: str | None = None) -> list[VideoSegment]:
        return list(self.segments)

    def _job(self, kind: JobKind, state: JobState, code: str | None = None) -> JobView:
        for j in self.job_views:
            if j.kind is kind and j.state is state and (
                code is None or (j.error is not None and j.error.code == code)
            ):
                return j
        raise LookupError(f"no {kind} {state} {code or ''} job fixture")

    def ingest_job_sequence(self, outcome: str | None = None) -> list[JobView]:
        """The ordered JobView snapshots a poller observes for one ingest job."""
        outcome = outcome or self.ingest_outcome
        running = self._job(JobKind.INGEST, JobState.RUNNING)
        seq = [self._job(JobKind.INGEST, JobState.CREATED), self._job(JobKind.INGEST, JobState.QUEUED)]

        if outcome == "FAILED":
            stop_stage = JobStage.EXTRACT
        elif outcome == "SPONSOR_TIMEOUT":
            stop_stage = JobStage.DESCRIBE
        else:
            stop_stage = None

        done = 0
        for stage, units in INGEST_STAGE_PLAN:
            seq.append(running.revise(stage=stage, completed_units=done))
            if stage is stop_stage:
                break
            if outcome == "PARTIAL" and stage is JobStage.INDEX:
                break
            done = units

        if outcome == "FAILED":
            seq.append(self._job(JobKind.INGEST, JobState.FAILED, "WORKER_INTERRUPTED"))
        elif outcome == "SPONSOR_TIMEOUT":
            seq.append(self._job(JobKind.INGEST, JobState.FAILED, "SPONSOR_TIMEOUT"))
        elif outcome == "PARTIAL":
            seq.append(self._job(JobKind.INGEST, JobState.PARTIAL))
        else:
            seq.append(self._job(JobKind.INGEST, JobState.SUCCEEDED))
        return seq

    def cancel_job_sequence(self) -> list[JobView]:
        """CANCELLING ('stopping after current safe checkpoint') then CANCELLED."""
        return [
            self._job(JobKind.INGEST, JobState.CANCELLING),
            self._job(JobKind.INGEST, JobState.CANCELLED),
        ]

    def start_ingest_job(self, video_id: str) -> JobView:
        return self.ingest_job_sequence()[0]

    def get_job(self, job_id: str) -> JobView:
        for j in self.job_views:
            if j.job_id == job_id:
                return j
        return self._job(JobKind.INGEST, JobState.SUCCEEDED)

    def cancel_job(self, job_id: str) -> JobView:
        return self.cancel_job_sequence()[-1]

    def video_for_ingest_state(self, state: JobState | None) -> SourceVideo:
        return self.source_video_partial if state is JobState.PARTIAL else self.source_video

    # -----------------------------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------------------------

    def search(
        self, query: str, max_results: int = 5, indexed_range: TimeRangeUs | None = None
    ) -> SearchEvidence:
        cleaned = query.strip()
        if not cleaned:
            return self.search_needs_clarification

        plan = asyncio.run(
            self.reasoner.plan_query(
                QueryPlanRequest(
                    query=cleaned,
                    video_id=self.source_video.video_id,
                    allowed_subject_classes=("car", "license_plate", "sign"),
                    indexed_range=indexed_range,
                )
            )
        )
        with self.trace.span(
            "search.plan",
            {
                "correlation_id": self.correlation_id,
                "query_sha256": hashlib.sha256(cleaned.encode()).hexdigest(),
                "objective_term_count": len(plan.objective_terms),
                "filtered_term_count": len(plan.filtered_terms),
                "mode": "FIXTURE",
            },
        ):
            pass

        if not any(term in cleaned.lower() for term in OBSERVABLE_TERMS):
            return self.search_evidence.revise(
                query=cleaned,
                query_plan=plan,
                status=SearchStatus.NO_RESULTS,
                results=(),
                correlation_id=self.correlation_id,
            )

        results = list(self.search_evidence.results)
        if indexed_range is not None:
            results = [r for r in results if r.start_pts_us < indexed_range.end_pts_us]
            results = [
                r.model_copy(
                    update={
                        "rank": i,
                        "outside_indexed_range_warning": r.end_pts_us > indexed_range.end_pts_us,
                    }
                )
                for i, r in enumerate(results, start=1)
            ]
        results = results[:max_results]

        with self.trace.span(
            "wandb.explain",
            {
                "correlation_id": self.correlation_id,
                "result_count": len(results),
                "segment_ids": ",".join(r.segment_id for r in results),
                "mode": "FIXTURE",
            },
        ):
            pass

        return self.search_evidence.revise(
            query=cleaned,
            query_plan=plan.model_copy(update={"time_range": None}),
            status=SearchStatus.OK if results else SearchStatus.NO_RESULTS,
            results=tuple(results),
            indexed_range=indexed_range or self.search_evidence.indexed_range,
            correlation_id=self.correlation_id,
        )

    # -----------------------------------------------------------------------------------------
    # Tracking
    # -----------------------------------------------------------------------------------------

    def get_track(self, confirmed: bool = True) -> Track:
        return self.track_confirmed if confirmed else self.track_not_confirmed

    # -----------------------------------------------------------------------------------------
    # Reconstruction & Provenance
    # -----------------------------------------------------------------------------------------

    def get_reconstruction(
        self, run_id: str | None = None, refused: bool = False, outcome: str | None = None
    ) -> ReconstructionRun:
        if refused or outcome == "REFUSED":
            return self.run_refused
        if outcome == "FAILED":
            return self.run_failed
        if run_id is not None:
            for run in (self.run_succeeded, self.run_refused, self.run_failed):
                if run.run_id == run_id:
                    return run
        return self.run_succeeded

    def outcome_for_target(self, target_frame_number: int) -> str:
        """The fixture refusal run targets f414; f417 is the defensible target."""
        if self.run_outcome == "FAILED":
            return "FAILED"
        if self.run_outcome == "REFUSED" or target_frame_number == 414:
            return "REFUSED"
        return "SUCCEEDED"

    def reconstruction_job_sequence(self, outcome: str) -> list[JobView]:
        running = self._job(JobKind.RECONSTRUCT, JobState.RUNNING)
        seq = [
            running.revise(stage=JobStage.ALIGN, completed_units=0),
            running.revise(stage=JobStage.ALIGN, completed_units=1),
            running,  # ALIGN 2 of 4
        ]
        if outcome == "FAILED":
            seq.append(self._job(JobKind.RECONSTRUCT, JobState.FAILED))
        elif outcome == "REFUSED":
            seq.append(self._job(JobKind.RECONSTRUCT, JobState.REFUSED))
        else:
            seq.append(running.revise(stage=JobStage.FUSE, completed_units=3))
            seq.append(self._job(JobKind.RECONSTRUCT, JobState.SUCCEEDED))
        return seq

    def regenerate_artifacts(self) -> ReconstructionRun:
        """Simulate a re-render that changes artifact hashes and so invalidates approval."""
        self._artifact_revision += 1
        base = self._run_succeeded_canonical
        bump = str(self._artifact_revision).encode()
        self.run_succeeded = base.revise(
            result_png_sha256=hashlib.sha256(base.result_png_sha256.encode() + bump).hexdigest(),
            provenance_sha256=hashlib.sha256(base.provenance_sha256.encode() + bump).hexdigest(),
        )
        return self.run_succeeded

    def list_decisions(self, run_id: str | None = None, refused: bool = False) -> list[PolicyDecision]:
        if refused or run_id == self.run_refused.run_id:
            return self.decisions_refused
        if run_id == self.run_failed.run_id:
            return []
        return self.decisions_succeeded

    def provenance_exceptions(self) -> list[list[float]]:
        """Pixels whose origin is not the identity mapping onto the target frame.

        Returned as ``[x, y, class, source_index, source_x, source_y]`` rows. Every other
        pixel is ORIGINAL, source index 0, at its own coordinate; the canvas uses this
        compact form to resolve any hovered pixel exactly as the NPZ does.
        """
        if self._npz_data is None:
            return []
        cls = self._npz_data["class"]
        idx = self._npz_data["source_index"]
        sx = self._npz_data["source_x"]
        sy = self._npz_data["source_y"]
        h, w = cls.shape
        yy, xx = np.mgrid[0:h, 0:w]
        mask = (cls != 0) | (idx != 0) | (sx != xx) | (sy != yy)
        ys, xs = np.nonzero(mask)
        return [
            [int(x), int(y), int(cls[y, x]), int(idx[y, x]), float(sx[y, x]), float(sy[y, x])]
            for y, x in zip(ys, xs, strict=True)
        ]

    def provenance_shape(self) -> tuple[int, int]:
        """(width, height) of the provenance arrays."""
        if self._npz_data is None:
            return (self.source_video.width_px, self.source_video.height_px)
        h, w = self._npz_data["class"].shape
        return (int(w), int(h))

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

        run = self.run_succeeded
        lut = run.provenance.source_lut if run.provenance else ()
        entry = lut[src_idx] if src_idx < len(lut) else None
        if entry is None:
            return None

        return PixelOrigin(
            run_id=run.run_id,
            x=x,
            y=y,
            provenance_class=ProvenanceClass(cls_val).name,  # type: ignore[arg-type]
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
        if run.run_id != run_id:
            raise ValueError(f"Unknown run {run_id}.")

        if decision == "APPROVE":
            verified, _ = self.verify_source()
            blockers = approval_blockers(run, source_verified=verified)
            if blockers:
                raise ValueError("Approval blocked: " + "; ".join(b.message for b in blockers))
            if run.result_png_sha256 != reviewed_result_sha256:
                raise ValueError(
                    f"Result hash mismatch! Expected {run.result_png_sha256}, got {reviewed_result_sha256}"
                )
            if run.provenance_sha256 != reviewed_provenance_sha256:
                raise ValueError(
                    f"Provenance hash mismatch! Expected {run.provenance_sha256}, got {reviewed_provenance_sha256}"
                )
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

        with self.trace.span(
            "review.record",
            {
                "correlation_id": self.correlation_id,
                "run_id": run.run_id,
                "review_id": review.review_id,
                "decision": str(review.decision),
                "reason_code": str(review.reason_code) if review.reason_code else None,
                "result_sha256_prefix": reviewed_result_sha256[:12],
                "provenance_sha256_prefix": reviewed_provenance_sha256[:12],
                # Analyst comment is deliberately not traced.
            },
        ):
            pass

        self._reviews[run_id] = review
        return review

    # -----------------------------------------------------------------------------------------
    # Evidence Reports
    # -----------------------------------------------------------------------------------------

    def report_facts(self, run: ReconstructionRun, review: HumanReview) -> ReportFacts:
        integrity = run.integrity
        return ReportFacts(
            case_id=self.case.case_id,
            run_id=run.run_id,
            review_id=review.review_id,
            facts=(
                ReportFact(key="target_frame_id", value=run.target_frame_id),
                ReportFact(key="target_pts_us", value=run.target_pts_us),
                ReportFact(key="policy_profile", value=run.policy_profile),
                ReportFact(key="algorithm_version", value=run.algorithm_version),
                ReportFact(
                    key="accepted_donor_frame_ids", value=", ".join(run.accepted_donor_frame_ids)
                ),
                ReportFact(key="integrity_score", value=integrity.score_0_100 if integrity else 0),
                ReportFact(
                    key="supported_coverage_pct",
                    value=integrity.supported_coverage * 100 if integrity else 0.0,
                ),
            ),
        )

    def draft_narrative(self, facts: ReportFacts) -> tuple[ReportNarrative, bool]:
        """Ask the W&B reasoner for prose, validate it, and fall back if ungrounded.

        Returns ``(narrative, fell_back)``.
        """
        drafted = asyncio.run(self.reasoner.draft_report(facts))
        if self.hallucinate_narrative:
            drafted = ReportNarrative(
                paragraphs=(HALLUCINATED_PARAGRAPH, *drafted.paragraphs),
                model_id=self.reasoner.model_id,
                source=ExplanationSource.REASONER,
            )
        validated = validate_or_fallback_narrative(drafted, facts)
        return validated, validated is not drafted

    def create_report(self, run_id: str) -> EvidenceReport:
        run = self.get_reconstruction(run_id)
        review = self._reviews.get(run_id)
        verified, observed = self.verify_source()
        blockers = export_blockers(run, review, source_verified=verified)
        if blockers or review is None:
            raise ValueError("Export blocked: " + "; ".join(b.message for b in blockers))

        report = self.fixture_report.revise(
            run_id=run.run_id,
            review_id=review.review_id,
            source_sha256_verified_at_export=observed,
        )
        self._reports[report.report_id] = report
        return report

    def get_report(self, report_id: str) -> EvidenceReport | None:
        return self._reports.get(report_id, self.fixture_report)

    # -----------------------------------------------------------------------------------------
    # Artifact Assets
    # -----------------------------------------------------------------------------------------

    def get_asset_path(self, name: str) -> Path:
        for base in (self.artifacts_dir, self.demo_source_dir):
            if (base / name).exists():
                return base / name
        raise FileNotFoundError(name)

    def frame_asset_name(self, frame_number: int) -> str | None:
        """Lossless still for a frame number, if the fixture bundle has one."""
        for name in (f"target_f{frame_number}.png", f"donor_f{frame_number}.png"):
            if (self.artifacts_dir / name).exists():
                return name
        return None
