"""In-memory fakes for ports consumed by the API. Other agents' impls do not exist here."""

from __future__ import annotations

import hashlib
import io
import shutil
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

from probity.domain.enums import (
    ALLOWED_JOB_TRANSITIONS,
    TERMINAL_JOB_STATES,
    InferenceMode,
    JobStage,
    JobState,
    ReasonCode,
)
from probity.domain.errors import (
    FixtureNotFound,
    IdempotencyConflict,
    InvalidStateTransition,
    NotFound,
    PayloadTooLarge,
    SourcePathViolation,
    UnsupportedMedia,
    ValidationFailed,
)
from probity.domain.ids import canonical_sha256, utc_now
from probity.domain.models import (
    AssetRef,
    CaseWorkspace,
    Detection,
    Embedding,
    EvidenceReport,
    FrameManifest,
    HumanReview,
    JobEvent,
    JobView,
    LineageRecord,
    PixelProvenance,
    PolicyDecision,
    ReconstructionRun,
    SearchEvidence,
    SearchPlan,
    SourceVideo,
    TimeRangeUs,
    Track,
    VideoSegment,
)
from probity.ports import (
    ClaimedJob,
    IdempotencyRecord,
    JobOutcome,
    JobPayload,
    JobSpec,
    ProbeResult,
    SearchQuery,
    SourceVerification,
    StagedUpload,
    StoredSource,
)

DEMO_SHA256 = "9806895754d3af16f50ba66a99514f4b7c1b6659b278fdf7f8260cfd12bf05e5"
DEMO_PROBE_SHA256 = "15fa28ba80537578e3f624b43fac525cc19e0b665ba7f3718ba4f119aebd2b4a"


class FakeRepository:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cases: dict[str, CaseWorkspace] = {}
        self.videos: dict[str, SourceVideo] = {}
        self.manifests: dict[str, FrameManifest] = {}
        self.segments: dict[str, list[VideoSegment]] = {}
        self.embeddings: dict[str, list[Embedding]] = {}
        self.detections: dict[str, list[Detection]] = {}
        self.searches: dict[str, SearchEvidence] = {}
        self.tracks: dict[str, Track] = {}
        self.runs: dict[str, ReconstructionRun] = {}
        self.decisions: dict[str, list[PolicyDecision]] = {}
        self.provenances: dict[str, PixelProvenance] = {}
        self.reviews: dict[str, list[HumanReview]] = {}
        self.reports: dict[str, EvidenceReport] = {}
        self.assets: dict[str, AssetRef] = {}
        self.lineage: dict[str, LineageRecord] = {}

    def put_case(self, case: CaseWorkspace) -> None:
        with self._lock:
            self.cases[case.case_id] = case

    def get_case(self, case_id: str) -> CaseWorkspace:
        with self._lock:
            try:
                return self.cases[case_id]
            except KeyError as exc:
                raise NotFound("Case not found.") from exc

    def put_video(self, video: SourceVideo) -> None:
        with self._lock:
            self.videos[video.video_id] = video

    def get_video(self, video_id: str) -> SourceVideo:
        with self._lock:
            try:
                return self.videos[video_id]
            except KeyError as exc:
                raise NotFound("Video not found.") from exc

    def list_videos(self, case_id: str) -> Sequence[SourceVideo]:
        with self._lock:
            return tuple(v for v in self.videos.values() if v.case_id == case_id)

    def put_manifest(self, manifest: FrameManifest) -> None:
        with self._lock:
            self.manifests[manifest.video_id] = manifest

    def get_manifest(self, video_id: str) -> FrameManifest:
        with self._lock:
            try:
                return self.manifests[video_id]
            except KeyError as exc:
                raise NotFound("Frame manifest not found.") from exc

    def put_segments(self, segments: Sequence[VideoSegment]) -> None:
        with self._lock:
            for segment in segments:
                bucket = self.segments.setdefault(segment.video_id, [])
                bucket[:] = [s for s in bucket if s.segment_id != segment.segment_id]
                bucket.append(segment)

    def list_segments(self, video_id: str) -> Sequence[VideoSegment]:
        with self._lock:
            return tuple(sorted(self.segments.get(video_id, ()), key=lambda s: s.ordinal))

    def put_embeddings(self, video_id: str, embeddings: Sequence[Embedding]) -> None:
        with self._lock:
            self.embeddings[video_id] = list(embeddings)

    def list_embeddings(self, video_id: str) -> Sequence[Embedding]:
        with self._lock:
            return tuple(self.embeddings.get(video_id, ()))

    def put_detections(self, detections: Sequence[Detection]) -> None:
        with self._lock:
            for det in detections:
                bucket = self.detections.setdefault(det.video_id, [])
                bucket.append(det)

    def list_detections(self, video_id: str) -> Sequence[Detection]:
        with self._lock:
            return tuple(self.detections.get(video_id, ()))

    def put_search(self, search: SearchEvidence) -> None:
        with self._lock:
            self.searches[search.search_id] = search

    def get_search(self, search_id: str) -> SearchEvidence:
        with self._lock:
            try:
                return self.searches[search_id]
            except KeyError as exc:
                raise NotFound("Search not found.") from exc

    def put_track(self, track: Track) -> None:
        with self._lock:
            self.tracks[track.track_id] = track

    def get_track(self, track_id: str) -> Track:
        with self._lock:
            try:
                return self.tracks[track_id]
            except KeyError as exc:
                raise NotFound("Track not found.") from exc

    def put_run(self, run: ReconstructionRun) -> None:
        with self._lock:
            self.runs[run.run_id] = run

    def get_run(self, run_id: str) -> ReconstructionRun:
        with self._lock:
            try:
                return self.runs[run_id]
            except KeyError as exc:
                raise NotFound("Reconstruction run not found.") from exc

    def put_decisions(self, decisions: Sequence[PolicyDecision]) -> None:
        with self._lock:
            for item in decisions:
                self.decisions.setdefault(item.run_id, []).append(item)

    def list_decisions(self, run_id: str) -> Sequence[PolicyDecision]:
        with self._lock:
            rows = list(self.decisions.get(run_id, ()))
            rows.sort(key=lambda d: d.sequence)
            return tuple(rows)

    def put_provenance(self, provenance: PixelProvenance) -> None:
        with self._lock:
            self.provenances[provenance.run_id] = provenance

    def get_provenance(self, run_id: str) -> PixelProvenance:
        with self._lock:
            try:
                return self.provenances[run_id]
            except KeyError as exc:
                raise NotFound("Provenance not found.") from exc

    def put_review(self, review: HumanReview) -> None:
        with self._lock:
            self.reviews.setdefault(review.run_id, []).append(review)

    def latest_review(self, run_id: str) -> HumanReview | None:
        with self._lock:
            rows = self.reviews.get(run_id, ())
            return rows[-1] if rows else None

    def put_report(self, report: EvidenceReport) -> None:
        with self._lock:
            self.reports[report.report_id] = report

    def get_report(self, report_id: str) -> EvidenceReport:
        with self._lock:
            try:
                return self.reports[report_id]
            except KeyError as exc:
                raise NotFound("Report not found.") from exc

    def put_asset(self, asset: AssetRef) -> None:
        with self._lock:
            self.assets[asset.asset_id] = asset

    def get_asset(self, asset_id: str) -> AssetRef:
        with self._lock:
            try:
                return self.assets[asset_id]
            except KeyError as exc:
                raise NotFound("Asset not found.") from exc

    def put_lineage(self, record: LineageRecord) -> None:
        with self._lock:
            self.lineage[record.lineage_id] = record


class FakeJobStore:
    def __init__(self, clock=utc_now, new_id=None) -> None:
        self.clock = clock
        self.new_id = new_id
        self._lock = threading.Lock()
        self.jobs: dict[str, JobView] = {}
        self.payloads: dict[str, JobPayload] = {}
        self._events: dict[str, list[JobEvent]] = {}
        self._leases: dict[str, str] = {}

    def _transition(self, job: JobView, to_state: JobState, **changes: object) -> JobView:
        allowed = ALLOWED_JOB_TRANSITIONS[job.state]
        if to_state not in allowed:
            raise InvalidStateTransition(f"{job.state} cannot transition to {to_state}.")
        return job.revise(state=to_state, updated_at=self.clock(), **changes)

    def create(self, spec: JobSpec) -> JobView:
        now = self.clock()
        created = JobView.create(
            job_id=spec.job_id,
            kind=spec.payload.kind,
            case_id=spec.case_id,
            subject_id=spec.subject_id,
            state=JobState.CREATED,
            total_units=spec.total_units,
            mode=InferenceMode.FIXTURE,
            correlation_id=spec.correlation_id,
            updated_at=now,
        )
        queued = self._transition(created, JobState.QUEUED)
        with self._lock:
            self.jobs[queued.job_id] = queued
            self.payloads[queued.job_id] = spec.payload
        return queued

    def get(self, job_id: str) -> JobView:
        with self._lock:
            try:
                return self.jobs[job_id]
            except KeyError as exc:
                raise NotFound("Job not found.") from exc

    def events(self, job_id: str) -> Sequence[JobEvent]:
        with self._lock:
            return tuple(self._events.get(job_id, ()))

    def request_cancel(self, job_id: str) -> JobView:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise NotFound("Job not found.")
            if job.state in {JobState.CANCELLING, *TERMINAL_JOB_STATES}:
                return job
            updated = self._transition(job, JobState.CANCELLING, cancel_requested_at=self.clock())
            self.jobs[job_id] = updated
            return updated

    def _claim_locked(self, job: JobView, owner: str) -> ClaimedJob:
        running = self._transition(
            job,
            JobState.RUNNING,
            stage=JobStage.VALIDATE,
            started_at=self.clock(),
        )
        self.jobs[job.job_id] = running
        self._leases[job.job_id] = owner
        return ClaimedJob(
            job=running,
            payload=self.payloads[job.job_id],
            lease_owner=owner,
            lease_expires_at=self.clock(),
        )

    def claim(self, job_id: str, owner: str) -> ClaimedJob:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise NotFound("Job not found.")
            if job.state is JobState.RUNNING and self._leases.get(job_id) == owner:
                return ClaimedJob(
                    job=job,
                    payload=self.payloads[job_id],
                    lease_owner=owner,
                    lease_expires_at=self.clock(),
                )
            return self._claim_locked(job, owner)

    def claim_next(self, owner: str, lease_seconds: int) -> ClaimedJob | None:
        del lease_seconds
        with self._lock:
            queued = [j for j in self.jobs.values() if j.state is JobState.QUEUED]
            if not queued:
                return None
            queued.sort(key=lambda j: j.created_at)
            return self._claim_locked(queued[0], owner)

    def renew_lease(self, job_id: str, owner: str, lease_seconds: int) -> bool:
        del lease_seconds
        with self._lock:
            return self._leases.get(job_id) == owner

    def report_progress(
        self, job_id: str, owner: str, stage: JobStage, completed: int, total: int
    ) -> JobView:
        del owner
        with self._lock:
            job = self.jobs[job_id]
            updated = job.revise(
                stage=stage,
                completed_units=completed,
                total_units=total,
                updated_at=self.clock(),
            )
            self.jobs[job_id] = updated
            return updated

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self.jobs[job_id]
            return job.state is JobState.CANCELLING or job.cancel_requested_at is not None

    def complete(self, job_id: str, owner: str, outcome: JobOutcome) -> JobView:
        del owner
        with self._lock:
            job = self.jobs[job_id]
            changes: dict[str, object] = {
                "error": outcome.error,
                "refusal_reasons": outcome.refusal_reasons,
                "partial": outcome.partial,
                "finished_at": self.clock(),
            }
            updated = self._transition(job, outcome.state, **changes)
            self.jobs[job_id] = updated
            self._leases.pop(job_id, None)
            return updated

    def mark_cancelled(self, job_id: str, owner: str, partial: object) -> JobView:
        del owner
        with self._lock:
            job = self.jobs[job_id]
            updated = self._transition(
                job, JobState.CANCELLED, partial=partial, finished_at=self.clock()
            )
            self.jobs[job_id] = updated
            return updated

    def recover_expired(self) -> Sequence[JobView]:
        return ()


class FakeIdempotencyStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str | None, str], IdempotencyRecord] = {}

    def lookup(self, route: str, case_id: str | None, key: str) -> IdempotencyRecord | None:
        with self._lock:
            return self._records.get((route, case_id, key))

    def save(self, record: IdempotencyRecord) -> IdempotencyRecord:
        key = (record.route, record.case_id, record.key)
        with self._lock:
            existing = self._records.get(key)
            if existing is None:
                self._records[key] = record
                return record
            if existing.request_sha256 != record.request_sha256:
                raise IdempotencyConflict("Idempotency-Key was reused with a different request.")
            return existing

    def purge_expired(self) -> int:
        return 0


class FakeSourceStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.staging_dir = data_dir / "staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.bytes_read = 0
        self.stage_calls = 0

    def stage_upload(self, stream: BinaryIO, max_bytes: int) -> StagedUpload:
        self.stage_calls += 1
        hasher = hashlib.sha256()
        total = 0
        path = self.staging_dir / f"upload-{id(stream)}.bin"
        with path.open("wb") as handle:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                if total + len(chunk) > max_bytes:
                    allowed = max_bytes - total
                    if allowed > 0:
                        handle.write(chunk[:allowed])
                        hasher.update(chunk[:allowed])
                        self.bytes_read += allowed
                        total += allowed
                    raise PayloadTooLarge("Upload exceeds the configured maximum size.")
                handle.write(chunk)
                hasher.update(chunk)
                self.bytes_read += len(chunk)
                total += len(chunk)
        if total <= 0:
            path.unlink(missing_ok=True)
            raise ValidationFailed("Upload body is empty.")
        return StagedUpload(staging_path=str(path), sha256=hasher.hexdigest(), byte_length=total)

    def commit(self, staged: StagedUpload) -> StoredSource:
        dest_rel = f"source/sha256/{staged.sha256[:2]}/{staged.sha256}/original.mp4"
        dest = self.data_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = Path(staged.staging_path)
        deduplicated = dest.is_file()
        if not deduplicated:
            shutil.copy2(src, dest)
            dest.chmod(0o444)
        if src.exists():
            src.unlink()
        observed = hashlib.sha256(dest.read_bytes()).hexdigest()
        if observed != staged.sha256:
            raise SourcePathViolation("staged hash did not match committed bytes")
        return StoredSource(
            sha256=staged.sha256,
            byte_length=staged.byte_length,
            storage_uri=dest_rel,
            deduplicated=deduplicated,
            verified_at=utc_now(),
        )

    def discard(self, staged: StagedUpload) -> None:
        Path(staged.staging_path).unlink(missing_ok=True)

    def verify(self, storage_uri: str, expected_sha256: str) -> SourceVerification:
        path = Path(self.resolve_read_path(storage_uri))
        observed = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        ok = observed == expected_sha256
        return SourceVerification(
            storage_uri=storage_uri,  # type: ignore[arg-type]
            expected_sha256=expected_sha256,  # type: ignore[arg-type]
            observed_sha256=observed,  # type: ignore[arg-type]
            verified=ok,
            checked_at=utc_now(),
            reason_code=ReasonCode.SOURCE_HASH_VERIFIED if ok else ReasonCode.SOURCE_HASH_MISMATCH,
        )

    def resolve_read_path(self, storage_uri: str) -> str:
        if ".." in storage_uri.split("/") or not storage_uri.startswith("source/sha256/"):
            raise SourcePathViolation("invalid source uri")
        path = (self.data_dir / storage_uri).resolve()
        if not path.is_relative_to(self.data_dir.resolve()):
            raise SourcePathViolation("source path escaped the data root")
        return str(path)


class FakeMediaProber:
    def probe(self, path: str) -> ProbeResult:
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if data[:8] == b"not-mp4!" or len(data) < 32:
            raise UnsupportedMedia("Not a supported MP4.")
        if digest == DEMO_SHA256:
            fields = {
                "container": "mp4",
                "video_codec": "h264",
                "width_px": 1280,
                "height_px": 720,
                "duration_us": 90_000_000,
                "time_base": "1/15360",
                "nominal_fps": "30/1",
                "frame_count": 2700,
                "has_audio": False,
                "extractor": "fake-prober",
            }
            return ProbeResult(
                **fields,
                probe_sha256=DEMO_PROBE_SHA256,  # type: ignore[arg-type]
            )
        fields = {
            "container": "mp4",
            "video_codec": "h264",
            "width_px": 1280,
            "height_px": 720,
            "duration_us": 1_000_000,
            "time_base": "1/30",
            "nominal_fps": "30/1",
            "frame_count": 30,
            "has_audio": False,
            "extractor": "fake-prober",
        }
        return ProbeResult(**fields, probe_sha256=canonical_sha256(fields))  # type: ignore[arg-type]

    def frame_manifest(self, path: str, video_id: str, source_sha256: str) -> FrameManifest:
        raise FixtureNotFound("frame manifests are produced by the worker")


class FakeSearchService:
    def __init__(self, new_id, clock=utc_now) -> None:
        self.new_id = new_id
        self.clock = clock

    async def search(
        self, video: SourceVideo, query: SearchQuery, correlation_id: str
    ) -> SearchEvidence:
        lowered = query.query.lower()
        needs = any(term in lowered for term in ("guilty", "who is", "flee", "motive"))
        if needs:
            return SearchEvidence.create(
                search_id=self.new_id(),
                case_id=video.case_id,
                video_id=video.video_id,
                query=query.query,
                query_plan=SearchPlan(
                    semantic_query="",
                    needs_clarification=True,
                    filtered_terms=("who", "guilty"),
                    policy_reason_codes=(ReasonCode.QUERY_POLICY_FILTERED,),
                ),
                planner_model_id="fixture/wandb-planner-v1",
                embedding_model_id="fixture/cosmos-embed-v1",
                reasoner_model_id=None,
                status="NEEDS_CLARIFICATION",
                mode=InferenceMode.FIXTURE,
                indexed_range=video.indexed_range
                or TimeRangeUs(start_pts_us=0, end_pts_us=max(video.duration_us, 1)),
                candidates=(),
                results=(),
                latency_ms=1,
                correlation_id=correlation_id,
            )
        from probity.domain.enums import SearchStatus
        from probity.domain.ids import segment_id
        from probity.domain.models import ScoreComponents, SearchCandidate, SearchResult

        sid = segment_id(video.video_id, 2)
        components = ScoreComponents(
            cosine=0.91, lexical_overlap=0.75, detection_match=1.0, visibility_match=1.0
        )
        candidate = SearchCandidate(
            segment_id=sid,
            vector_rank=1,
            similarity=0.91,
            components=components,
            score=0.90,
            final_rank=1,
        )
        result = SearchResult(
            rank=1,
            segment_id=sid,
            video_id=video.video_id,
            source_sha256=video.sha256,
            score=0.90,
            components=components,
            start_pts_us=12_000_000,
            end_pts_us=20_000_000,
            start_frame=360,
            end_frame=600,
            thumbnail_uri=None,
            description="A blue sedan; rear plate region is most visible.",
            explanation="A blue sedan crosses; a rear plate region is detected near 00:13.9.",
            explanation_source="TEMPLATE",
            explanation_input_hash="a" * 64,
        )
        return SearchEvidence.create(
            search_id=self.new_id(),
            case_id=video.case_id,
            video_id=video.video_id,
            query=query.query,
            query_plan=SearchPlan(
                semantic_query="blue sedan rear license plate most visible",
                objective_terms=("blue sedan", "rear plate"),
                subject_classes=("car", "license_plate"),
            ),
            planner_model_id="fixture/wandb-planner-v1",
            embedding_model_id="fixture/cosmos-embed-v1",
            reasoner_model_id=None,
            status=SearchStatus.OK,
            mode=InferenceMode.FIXTURE,
            indexed_range=video.indexed_range
            or TimeRangeUs(start_pts_us=0, end_pts_us=max(video.duration_us, 1)),
            candidates=(candidate,),
            results=(result,),
            latency_ms=2,
            correlation_id=correlation_id,
        )


class CountingStream(io.RawIOBase):
    """BinaryIO that records how many bytes were actually read."""

    def __init__(self, payload: bytes) -> None:
        self._buf = io.BytesIO(payload)
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        chunk = self._buf.read(size)
        self.bytes_read += len(chunk)
        return chunk


class ExplodingRepository(FakeRepository):
    """Raises an exception containing a secret path on get_case."""

    def get_case(self, case_id: str) -> CaseWorkspace:
        raise RuntimeError("failed opening /secret/keys/prod.token for case " + case_id)
