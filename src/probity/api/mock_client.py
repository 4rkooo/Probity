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
from PIL import Image

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
from probity.domain.ids import canonical_sha256, new_uuid7, parse_frame_id
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
from probity.eval.window import WindowBundle, load_window
from probity.ports import QueryPlanRequest, ReportFact, ReportFacts, ReportNarrative
from probity.reports.gate import approval_blockers, export_blockers
from probity.reports.validator import validate_or_fallback_narrative
from probity.api.live_media import (
    LiveFrame,
    build_live_reconstruction,
    build_live_track,
    resolve_source_mp4,
    sample_live_frames,
)

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


# Person 2's real reconstruction goldens (fixtures/synthetic). The completed run is the
# Probity output the analyst inspects; the single-donor window supplies a real refusal.
EVAL_WINDOW = Path("fixtures/synthetic/plate_translate_v1")
REFUSAL_WINDOW = Path("fixtures/synthetic/plate_single_donor_v1")
PIXEL_HASH_VERSION = "probity-pixel-v1"


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
        self._bundled_source_video = self.source_video
        self._bundled_case = self.case
        self._bundled_segments = list(self.segments)
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
        self.live_api = None
        self.live_job_id = None
        self.live_video_id = None
        self._live_source_path = None
        self._clear_live_media()
        self.case = self._bundled_case
        self.source_video = self._bundled_source_video
        self.segments = list(self._bundled_segments)

    @property
    def uses_live_upload(self) -> bool:
        return self.live_job_id is not None and self.live_api is not None

    def _clear_live_media(self) -> None:
        self._live_frames: list[LiveFrame] = []
        self._live_frame_map: dict[int, Path] = {}
        self._live_track: Track | None = None
        self._live_run: ReconstructionRun | None = None
        self._live_decisions: list[PolicyDecision] = []
        self._live_result_path: Path | None = None
        self._live_provenance_path: Path | None = None
        self._live_npz_data: dict[str, np.ndarray] | None = None

    def adopt_live_upload(
        self,
        *,
        live_api: object,
        case: CaseWorkspace,
        video: SourceVideo,
        job_id: str,
        local_path: Path | None = None,
    ) -> None:
        """Switch the UI client onto a video accepted by the live API."""
        self.live_api = live_api
        self.case = case
        self.source_video = video
        self.live_video_id = video.video_id
        self.live_job_id = job_id
        self._live_source_path = local_path
        self.segments = []
        self._clear_live_media()

    def restore_bundled_demo(self) -> None:
        """Return to the offline bundled clip after a live custom-upload session."""
        self.live_api = None
        self.live_job_id = None
        self.live_video_id = None
        self._live_source_path = None
        self._clear_live_media()
        self.run_succeeded = self._run_succeeded_canonical
        self.case = self._bundled_case
        self.source_video = self._bundled_source_video
        self.segments = list(self._bundled_segments)

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
        self.window: WindowBundle = load_window(self.root / EVAL_WINDOW)
        self.refusal_window: WindowBundle = load_window(self.root / REFUSAL_WINDOW)
        self.recon_dir = self.window.root / "reconstructions" / "completed"
        refused_dir = self.refusal_window.root / "reconstructions" / "refused"
        self.track_confirmed = self.window.track
        self.track_not_confirmed = Track.model_validate_json(load("track_not_confirmed.json"))
        self._run_succeeded_canonical = ReconstructionRun.model_validate_json(
            (self.recon_dir / "run.json").read_text()
        )
        self.run_succeeded = self._run_succeeded_canonical
        self.run_refused = ReconstructionRun.model_validate_json(
            (refused_dir / "run.json").read_text()
        )
        self.run_failed = ReconstructionRun.model_validate_json(load("reconstruction_run_failed.json"))
        self.decisions_succeeded = [
            PolicyDecision.model_validate(d)
            for d in json.loads((self.recon_dir / "decisions.json").read_text())
        ]
        self.decisions_refused = [
            PolicyDecision.model_validate(d)
            for d in json.loads((refused_dir / "decisions.json").read_text())
        ]
        self.job_views = [JobView.model_validate(j) for j in json.loads(load("job_views.json"))]
        self.fixture_report = EvidenceReport.model_validate_json(load("evidence_report.json"))

        manifest = json.loads((self.root / "fixtures" / "demo" / "manifest.json").read_text())
        self.manifest = manifest["fixtures"][0]
        self.manifest_created_at: str = self.manifest.get("created_at", "unknown")

        npz_path = self.recon_dir / "provenance.npz"
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
        if self._live_source_path is not None and Path(self._live_source_path).is_file():
            return Path(self._live_source_path)
        if self.uses_live_upload:
            try:
                return resolve_source_mp4(
                    data_dir=self.root / "data",
                    storage_uri=self.source_video.storage_uri,
                    fallback=None,
                )
            except FileNotFoundError:
                pass
        return self.demo_source_dir / self.source_video.original_name

    def verify_source(self) -> tuple[bool, str]:
        """Stream-hash the canonical source now; returns (matches_ingest_hash, observed_sha256)."""
        path = self.source_path()
        if not path.exists():
            # Live uploads are verified by the API; trust the accepted hash when local bytes
            # are not kept on the UI host.
            if self.uses_live_upload:
                return True, self.source_video.sha256
            return False, "0" * 64
        observed = _sha256_file(path)
        if self.source_tampered:
            # Simulates altered bytes on disk without touching the real fixture.
            observed = hashlib.sha256(observed.encode() + b"tampered").hexdigest()
        return observed == self.source_video.sha256, observed

    def verify_run_source(self, window: WindowBundle | None = None) -> tuple[bool, str]:
        """Re-verify the reconstruction source: the ordered pixel-hash digest of every frame.

        Synthetic windows have no MP4 container; their identity is
        ``canonical_sha256({synth_version, pixel_sha256: [...]})`` over the lossless PNGs.
        Live custom uploads re-hash the source MP4 against the ingest sha256.
        """
        if self.uses_live_upload and self._live_run is not None:
            return self.verify_source()
        window = window or self.window
        meta = json.loads((window.root / "window.json").read_text())
        hashes = []
        for ref in sorted(window.frames, key=lambda f: f.frame_number):
            with Image.open(self.frame_path(ref.frame_number, window)) as img:
                rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
            h, w, _ = rgb.shape
            header = f"{PIXEL_HASH_VERSION}|uint8|{h},{w},3|RGB\n".encode("ascii")
            hashes.append(hashlib.sha256(header + rgb.tobytes()).hexdigest())
        observed = canonical_sha256({"synth_version": meta["synth_version"], "pixel_sha256": hashes})
        if self.source_tampered:
            observed = hashlib.sha256(observed.encode() + b"tampered").hexdigest()
        return observed == window.source_sha256, observed

    def run_source_identity(self) -> dict[str, object]:
        """Display facts about the reconstruction source for the report."""
        if self.uses_live_upload and self._live_run is not None:
            return {
                "original_name": self.source_video.original_name,
                "width_px": self.source_video.width_px,
                "height_px": self.source_video.height_px,
                "duration_us": self.source_video.duration_us,
                "sha256": self.source_video.sha256,
            }
        frames = sorted(self.window.frames, key=lambda f: f.frame_number)
        return {
            "original_name": f"{self.window.fixture_id} (lossless PNG sequence, {len(frames)} frames)",
            "width_px": frames[0].width_px,
            "height_px": frames[0].height_px,
            "duration_us": frames[-1].pts_us,
            "sha256": self.window.source_sha256,
        }

    def frame_path(self, frame_number: int, window: WindowBundle | None = None) -> Path:
        if self.uses_live_upload and frame_number in self._live_frame_map:
            return self._live_frame_map[frame_number]
        return (window or self.window).root / "frames" / f"f{frame_number:04d}.png"

    def frame_pts_us(self, frame_number: int) -> int:
        if self.uses_live_upload:
            for frame in self._live_frames:
                if frame.frame_number == frame_number:
                    return frame.pts_us
        for ref in self.window.frames:
            if ref.frame_number == frame_number:
                return ref.pts_us
        raise LookupError(frame_number)

    @property
    def target_frame_number(self) -> int:
        if self.uses_live_upload and self._live_run is not None:
            return parse_frame_id(self._live_run.target_frame_id)[1]
        if self.uses_live_upload and self._live_track is not None:
            return parse_frame_id(self._live_track.seed_frame_id)[1]
        return parse_frame_id(self.run_succeeded.target_frame_id)[1]

    def target_frame_path(self) -> Path:
        return self.frame_path(self.target_frame_number)

    def result_path(self) -> Path:
        if self.uses_live_upload and self._live_result_path is not None:
            return self._live_result_path
        return self.recon_dir / "result.png"

    def provenance_path(self) -> Path:
        if self.uses_live_upload and self._live_provenance_path is not None:
            return self._live_provenance_path
        return self.recon_dir / "provenance.npz"

    def ensure_live_subject(self, seek_us: int) -> Track:
        """Extract stills around the search hit and build a confirmed rigid-ROI track."""
        if not self.uses_live_upload:
            raise RuntimeError("ensure_live_subject requires a live custom upload")
        if self._live_track is not None and self._live_frames:
            return self._live_track
        mp4 = self.source_path()
        out_dir = self.root / "data" / "live_frames" / self.source_video.video_id
        frames = sample_live_frames(
            mp4=mp4,
            out_dir=out_dir,
            seek_us=seek_us,
            duration_us=self.source_video.duration_us,
            width_px=self.source_video.width_px,
            height_px=self.source_video.height_px,
        )
        track = build_live_track(
            case_id=self.case.case_id,
            video_id=self.source_video.video_id,
            frames=frames,
            width_px=self.source_video.width_px,
            height_px=self.source_video.height_px,
        )
        self._live_frames = frames
        self._live_frame_map = {f.frame_number: f.path for f in frames}
        self._live_track = track
        return track

    def ensure_live_reconstruction(self, target_frame_number: int) -> ReconstructionRun:
        """Build live Probity artifacts from the custom-upload stills."""
        if not self.uses_live_upload or self._live_track is None or not self._live_frames:
            raise RuntimeError("ensure_live_reconstruction requires ensure_live_subject first")
        if (
            self._live_run is not None
            and parse_frame_id(self._live_run.target_frame_id)[1] == target_frame_number
        ):
            return self._live_run
        out_dir = self.root / "data" / "live_recon" / self.source_video.video_id
        run, decisions, result_path, provenance_path, arrays = build_live_reconstruction(
            case_id=self.case.case_id,
            video_id=self.source_video.video_id,
            track=self._live_track,
            frames=self._live_frames,
            target_frame_number=target_frame_number,
            out_dir=out_dir,
        )
        self._live_run = run
        self._live_decisions = decisions
        self._live_result_path = result_path
        self._live_provenance_path = provenance_path
        self._live_npz_data = arrays
        self.run_succeeded = run
        return run

    def list_segments(self, video_id: str | None = None) -> list[VideoSegment]:
        if self.uses_live_upload and self.live_api is not None and self.live_video_id:
            self.segments = list(self.live_api.list_segments(self.live_video_id))
        return list(self.segments)

    def poll_live_ingest(self) -> JobView:
        if not self.uses_live_upload or self.live_api is None or self.live_job_id is None:
            raise RuntimeError("no live ingest job")
        job = self.live_api.get_job(self.live_job_id)
        if self.live_video_id is not None:
            self.source_video = self.live_api.get_video(self.live_video_id)
        if job.state in (JobState.SUCCEEDED, JobState.PARTIAL):
            self.list_segments()
        return job

    def cancel_live_ingest(self) -> JobView:
        if not self.uses_live_upload or self.live_api is None or self.live_job_id is None:
            raise RuntimeError("no live ingest job")
        return self.live_api.cancel_job(self.live_job_id)

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

        if self.uses_live_upload and self.live_api is not None and self.live_video_id:
            return self.live_api.search(self.live_video_id, cleaned, max_results=max_results)

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
        if self.uses_live_upload and self._live_track is not None:
            if confirmed:
                return self._live_track
            return self.track_not_confirmed
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
        if self.uses_live_upload and self._live_run is not None:
            if run_id is None or run_id == self._live_run.run_id:
                return self._live_run
        if run_id is not None:
            for run in (self.run_succeeded, self.run_refused, self.run_failed):
                if run.run_id == run_id:
                    return run
        return self.run_succeeded

    def outcome_for_target(self, target_frame_number: int) -> str:
        """Outcome of the verified cached run for this target.

        Only the golden target has a stored Probity run; other frames return ``NOT_CACHED``
        rather than pretending a reconstruction ran. Live custom uploads reconstruct any
        accepted subject-window frame on demand.
        """
        if self.run_outcome == "FAILED":
            return "FAILED"
        if self.run_outcome == "REFUSED":
            return "REFUSED"
        if self.uses_live_upload and self._live_track is not None:
            accepted = {
                parse_frame_id(o.frame_id)[1]
                for o in self._live_track.observations
                if o.accepted
            }
            return "SUCCEEDED" if target_frame_number in accepted else "NOT_CACHED"
        if target_frame_number != self.target_frame_number:
            return "NOT_CACHED"
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
        if self.uses_live_upload and self._live_run is not None:
            if run_id is None or run_id == self._live_run.run_id:
                return list(self._live_decisions)
        return self.decisions_succeeded

    def _active_npz(self) -> dict[str, np.ndarray] | None:
        if self.uses_live_upload and self._live_npz_data is not None:
            return self._live_npz_data
        return self._npz_data

    def provenance_exceptions(self) -> list[list[float]]:
        """Pixels whose origin is not the identity mapping onto the target frame.

        Returned as ``[x, y, class, source_index, source_x, source_y]`` rows. Every other
        pixel is ORIGINAL, source index 0, at its own coordinate; the canvas uses this
        compact form to resolve any hovered pixel exactly as the NPZ does.
        """
        data = self._active_npz()
        if data is None:
            return []
        cls = data["class"]
        idx = data["source_index"]
        sx = data["source_x"]
        sy = data["source_y"]
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
        data = self._active_npz()
        if data is None:
            return (self.source_video.width_px, self.source_video.height_px)
        h, w = data["class"].shape
        return (int(w), int(h))

    def resolve_pixel(self, x: int, y: int, run_id: str | None = None) -> PixelOrigin | None:
        """Resolve pixel origin from provenance NPZ and run source LUT."""
        data = self._active_npz()
        if data is None:
            return None

        h, w = data["class"].shape
        if not (0 <= x < w and 0 <= y < h):
            return None

        cls_val = int(data["class"][y, x])
        src_idx = int(data["source_index"][y, x])
        src_x = float(data["source_x"][y, x])
        src_y = float(data["source_y"][y, x])

        run = self.get_reconstruction(run_id)
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
            verified, _ = self.verify_run_source()
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
        verified, observed = self.verify_run_source()
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

    def export_artifacts(self) -> dict[str, tuple[Path, str]]:
        """Bundle filename -> (local path, role) for the current reconstruction."""
        run = self._live_run if (self.uses_live_upload and self._live_run is not None) else self.run_succeeded
        target = parse_frame_id(run.target_frame_id)[1]
        artifacts = {
            f"target_f{target}.png": (self.frame_path(target), "target_still"),
            "result.png": (self.result_path(), "result"),
            "provenance.npz": (self.provenance_path(), "provenance"),
        }
        for entry in run.provenance.source_lut if run.provenance else ():
            if entry.role == "DONOR":
                artifacts[f"donor_f{entry.frame_number}.png"] = (
                    self.frame_path(entry.frame_number),
                    "donor_still",
                )
        return artifacts

    def refusal_target_path(self) -> Path:
        frame = parse_frame_id(self.run_refused.target_frame_id)[1]
        return self.frame_path(frame, self.refusal_window)
