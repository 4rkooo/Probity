"""Probity-owned Protocol contracts and their typed request/response objects.

These are not claims about vendor SDKs. Live adapters translate them inside ``adapters/live/`` only;
sponsor-specific types never appear here or in domain/service modules.

Every adapter exposes ``adapter_name``, ``model_id``, ``mode``, ``schema_version`` and
``async health() -> AdapterHealth``. All returned values are frozen Pydantic models.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Annotated, BinaryIO, Literal, Protocol, runtime_checkable

from pydantic import Field

from probity.domain.enums import (
    AdapterMode,
    ExplanationSource,
    JobKind,
    JobStage,
    JobState,
    ReasonCode,
    SubjectType,
)
from probity.domain.models import (
    AdapterHealth,
    AssetRef,
    BBoxPx,
    CaseWorkspace,
    Detection,
    Embedding,
    EvidenceReport,
    FrameId,
    FrameManifest,
    FrameReference,
    FrozenModel,
    GroundedExplanation,
    HumanReview,
    JobError,
    JobEvent,
    JobView,
    LineageRecord,
    ModelId,
    PartialCoverage,
    PixelProvenance,
    PolicyDecision,
    PtsUs,
    Rational,
    ReconstructionRun,
    RetrievedSegment,
    ScalarValue,
    SearchEvidence,
    SearchPlan,
    SegmentDescription,
    SegmentId,
    Sha256Hex,
    SourceStorageUri,
    SourceVideo,
    TimeRangeUs,
    Track,
    UtcTimestamp,
    Uuid7,
    VideoSegment,
)

# --------------------------------------------------------------------------------------------
# Adapter metadata
# --------------------------------------------------------------------------------------------


class AdapterMetadata(FrozenModel):
    adapter_name: Annotated[str, Field(min_length=1, max_length=64)]
    model_id: ModelId | None
    mode: AdapterMode
    schema_version: Literal["1.0"] = "1.0"


@runtime_checkable
class Adapter(Protocol):
    @property
    def adapter_name(self) -> str: ...
    @property
    def model_id(self) -> str | None: ...
    @property
    def mode(self) -> AdapterMode: ...
    @property
    def schema_version(self) -> str: ...
    async def health(self) -> AdapterHealth:
        """Bounded by ``adapters.health_budget_s``; never raises, reports DISABLED/UNAVAILABLE."""
        ...


# --------------------------------------------------------------------------------------------
# Sponsor-port DTOs
# --------------------------------------------------------------------------------------------


class ClipReference(FrozenModel):
    video_id: Uuid7
    segment_id: SegmentId
    ordinal: Annotated[int, Field(ge=0)]
    source_sha256: Sha256Hex
    storage_uri: SourceStorageUri
    start_pts_us: PtsUs
    end_pts_us: PtsUs


class SearchRequest(FrozenModel):
    video_id: Uuid7
    source_sha256: Sha256Hex
    query_embedding: Embedding
    top_k: Annotated[int, Field(gt=0, le=50)] = 12
    time_range: TimeRangeUs | None = None
    subject_classes: tuple[str, ...] = ()


class QueryPlanRequest(FrozenModel):
    query: Annotated[str, Field(min_length=1, max_length=500)]
    video_id: Uuid7
    allowed_subject_classes: tuple[str, ...]
    indexed_range: TimeRangeUs | None = None


class TrackRequest(FrozenModel):
    track_id: Uuid7
    case_id: Uuid7
    video_id: Uuid7
    source_sha256: Sha256Hex
    subject_type: SubjectType
    seed_frame_id: FrameId
    seed_bbox_px: BBoxPx
    seed_detection_id: Uuid7 | None = None
    window_radius_us: Annotated[int, Field(gt=0)] = 2_000_000


class ReconstructionRequest(FrozenModel):
    run_id: Uuid7
    case_id: Uuid7
    video_id: Uuid7
    source_sha256: Sha256Hex
    source_storage_uri: SourceStorageUri
    track: Track
    target_frame_id: FrameId
    target_bbox_px: BBoxPx
    policy_profile: Literal["demo-conservative-v1"] = "demo-conservative-v1"
    config_sha256: Sha256Hex


class ReconstructionResult(FrozenModel):
    run: ReconstructionRun
    decisions: tuple[PolicyDecision, ...]
    provenance: PixelProvenance | None


class ReportFact(FrozenModel):
    key: Annotated[str, Field(min_length=1, max_length=80)]
    value: ScalarValue
    citations: tuple[Annotated[str, Field(max_length=120)], ...] = ()


class ReportFacts(FrozenModel):
    case_id: Uuid7
    run_id: Uuid7
    review_id: Uuid7
    facts: tuple[ReportFact, ...]


class ReportNarrative(FrozenModel):
    paragraphs: tuple[Annotated[str, Field(min_length=1, max_length=2000)], ...]
    model_id: ModelId | None
    source: ExplanationSource


class EvaluationMetric(FrozenModel):
    name: Annotated[str, Field(min_length=1, max_length=80)]
    value: float


class EvaluationRow(FrozenModel):
    name: Annotated[str, Field(min_length=1, max_length=80)]
    correlation_id: Uuid7
    metrics: tuple[EvaluationMetric, ...]


# --------------------------------------------------------------------------------------------
# Sponsor protocols (section 7)
# --------------------------------------------------------------------------------------------


class EvidenceStore(Adapter, Protocol):
    async def put_source(self, video: SourceVideo, local_path: str) -> str: ...
    async def upsert_segments(
        self, segments: Sequence[VideoSegment], embeddings: Sequence[Embedding]
    ) -> None: ...
    async def upsert_detections(self, detections: Sequence[Detection]) -> None: ...
    async def search(self, request: SearchRequest) -> Sequence[RetrievedSegment]: ...
    async def put_lineage(self, record: LineageRecord) -> None: ...
    async def put_report(self, report: EvidenceReport) -> None: ...


class VideoUnderstanding(Adapter, Protocol):
    async def describe(self, clip: ClipReference, prompt: str) -> SegmentDescription: ...
    async def embed_segments(self, items: Sequence[SegmentDescription]) -> Sequence[Embedding]: ...
    async def embed_query(self, query: str) -> Embedding: ...


class DetectorTracker(Adapter, Protocol):
    async def detect(
        self, frames: Sequence[FrameReference], classes: set[str]
    ) -> Sequence[Detection]: ...
    async def track(self, request: TrackRequest) -> Track: ...


class EvidenceReasoner(Adapter, Protocol):
    async def plan_query(self, request: QueryPlanRequest) -> SearchPlan: ...
    async def explain_results(
        self, query: str, evidence: Sequence[RetrievedSegment]
    ) -> Sequence[GroundedExplanation]: ...
    async def draft_report(self, facts: ReportFacts) -> ReportNarrative: ...


class SpanContext(AbstractContextManager["SpanContext"], Protocol):
    def set_attribute(self, key: str, value: ScalarValue) -> None: ...


class TraceSink(Protocol):
    def span(self, name: str, safe_attributes: dict[str, object]) -> SpanContext: ...
    async def log_evaluation(self, row: EvaluationRow) -> None: ...


class CancelToken(Protocol):
    def is_cancelled(self) -> bool: ...
    def checkpoint(self) -> None:
        """Raise ``probity.domain.errors.JobCancelled`` if cancellation was requested."""
        ...


class Reconstructor(Protocol):
    def reconstruct(
        self, request: ReconstructionRequest, cancel: CancelToken
    ) -> ReconstructionResult: ...


# --------------------------------------------------------------------------------------------
# Media (Agent A)
# --------------------------------------------------------------------------------------------


class StagedUpload(FrozenModel):
    """Bytes streamed to a private temp file under ``{data}/staging``; not yet accepted."""

    staging_path: Annotated[str, Field(min_length=1)]
    sha256: Sha256Hex
    byte_length: Annotated[int, Field(gt=0)]


class StoredSource(FrozenModel):
    sha256: Sha256Hex
    byte_length: Annotated[int, Field(gt=0)]
    storage_uri: SourceStorageUri
    deduplicated: bool
    verified_at: UtcTimestamp


class SourceVerification(FrozenModel):
    storage_uri: SourceStorageUri
    expected_sha256: Sha256Hex
    observed_sha256: Sha256Hex | None
    verified: bool
    checked_at: UtcTimestamp
    reason_code: Literal[ReasonCode.SOURCE_HASH_VERIFIED, ReasonCode.SOURCE_HASH_MISMATCH]


class ProbeResult(FrozenModel):
    container: Literal["mp4"]
    video_codec: Literal["h264", "hevc"]
    width_px: Annotated[int, Field(gt=0, le=1920)]
    height_px: Annotated[int, Field(gt=0, le=1080)]
    duration_us: Annotated[int, Field(gt=0, le=600_000_000)]
    time_base: Rational
    nominal_fps: Rational
    frame_count: Annotated[int, Field(gt=0)]
    has_audio: bool
    probe_sha256: Sha256Hex
    extractor: Annotated[str, Field(min_length=1, max_length=200)]


class SegmentWindow(FrozenModel):
    ordinal: Annotated[int, Field(ge=0)]
    start_pts_us: PtsUs
    end_pts_us: PtsUs
    start_frame: Annotated[int, Field(ge=0)]
    end_frame: Annotated[int, Field(ge=0)]


class SourceStore(Protocol):
    def stage_upload(self, stream: BinaryIO, max_bytes: int) -> StagedUpload:
        """Stream to temp while hashing; raise PayloadTooLarge without reading past the limit."""
        ...

    def commit(self, staged: StagedUpload) -> StoredSource:
        """Atomic content-addressed placement, fsync file+dir, read-only, immediate re-hash."""
        ...

    def discard(self, staged: StagedUpload) -> None: ...
    def verify(self, storage_uri: str, expected_sha256: str) -> SourceVerification: ...
    def resolve_read_path(self, storage_uri: str) -> str:
        """Absolute path for read-only use; rejects traversal and non-source URIs."""
        ...


class DerivedStore(Protocol):
    def derived_path(self, video_id: str, kind: str, entity_id: str, filename: str) -> str:
        """Absolute path under ``derived/``; raises SourcePathViolation for source/ paths."""
        ...


class MediaProber(Protocol):
    def probe(self, path: str) -> ProbeResult:
        """ffprobe with a 10 s timeout; raises UnsupportedMedia (415) on any constraint failure."""
        ...

    def frame_manifest(self, path: str, video_id: str, source_sha256: str) -> FrameManifest: ...


class Segmenter(Protocol):
    def plan(self, manifest: FrameManifest, duration_us: int) -> Sequence[SegmentWindow]: ...


class ThumbnailExtractor(Protocol):
    def extract_thumbnail(
        self, source_path: str, window: SegmentWindow, manifest: FrameManifest, out_path: str
    ) -> int:
        """Write a 480 px-wide non-canonical JPEG; return the chosen frame_number."""
        ...

    def extract_frame_png(
        self, source_path: str, manifest: FrameManifest, frame_number: int, out_path: str
    ) -> str:
        """Write one lossless PNG for an explicitly requested frame; return the pixel SHA-256."""
        ...


# --------------------------------------------------------------------------------------------
# Jobs and persistence (Agent B)
# --------------------------------------------------------------------------------------------


class IngestPayload(FrozenModel):
    kind: Literal[JobKind.INGEST] = JobKind.INGEST
    video_id: Uuid7
    fixture_id: str | None = None


class TrackPayload(FrozenModel):
    kind: Literal[JobKind.TRACK] = JobKind.TRACK
    request: TrackRequest


class ReconstructPayload(FrozenModel):
    kind: Literal[JobKind.RECONSTRUCT] = JobKind.RECONSTRUCT
    request: ReconstructionRequest


class ReportPayload(FrozenModel):
    kind: Literal[JobKind.REPORT] = JobKind.REPORT
    report_id: Uuid7
    run_id: Uuid7
    review_id: Uuid7


JobPayload = Annotated[
    IngestPayload | TrackPayload | ReconstructPayload | ReportPayload, Field(discriminator="kind")
]


class JobSpec(FrozenModel):
    job_id: Uuid7
    case_id: Uuid7
    subject_id: Uuid7
    payload: JobPayload
    total_units: Annotated[int, Field(ge=0)] = 0
    correlation_id: Uuid7


class ClaimedJob(FrozenModel):
    job: JobView
    payload: JobPayload
    lease_owner: Annotated[str, Field(min_length=1, max_length=128)]
    lease_expires_at: UtcTimestamp


class JobOutcome(FrozenModel):
    state: Literal[JobState.SUCCEEDED, JobState.PARTIAL, JobState.REFUSED, JobState.FAILED]
    error: JobError | None = None
    refusal_reasons: tuple[ReasonCode, ...] = ()
    partial: PartialCoverage | None = None


class IdempotencyRecord(FrozenModel):
    route: Annotated[str, Field(min_length=1, max_length=200)]
    case_id: Uuid7 | None
    key: Annotated[str, Field(min_length=8, max_length=128)]
    request_sha256: Sha256Hex
    status_code: Annotated[int, Field(ge=100, le=599)]
    response_body: Annotated[str, Field(description="Canonical JSON of the original response body")]
    created_at: UtcTimestamp
    expires_at: UtcTimestamp


class JobStore(Protocol):
    def create(self, spec: JobSpec) -> JobView:
        """Insert CREATED and transition to QUEUED in one transaction."""
        ...

    def get(self, job_id: str) -> JobView: ...
    def events(self, job_id: str) -> Sequence[JobEvent]: ...
    def request_cancel(self, job_id: str) -> JobView:
        """QUEUED/RUNNING -> CANCELLING; idempotent on CANCELLING/terminal (returns the view)."""
        ...

    def claim_next(self, owner: str, lease_seconds: int) -> ClaimedJob | None:
        """Atomically lease the oldest QUEUED job and move it to RUNNING."""
        ...

    def renew_lease(self, job_id: str, owner: str, lease_seconds: int) -> bool: ...
    def abandon_lease(self, job_id: str, owner: str) -> None:
        """Expire a RUNNING lease without a state change, for process shutdown."""
        ...

    def report_progress(
        self, job_id: str, owner: str, stage: JobStage, completed: int, total: int
    ) -> JobView: ...
    def is_cancel_requested(self, job_id: str) -> bool: ...
    def complete(self, job_id: str, owner: str, outcome: JobOutcome) -> JobView: ...
    def mark_cancelled(
        self, job_id: str, owner: str, partial: PartialCoverage | None
    ) -> JobView: ...
    def recover_expired(self) -> Sequence[JobView]:
        """Startup recovery: expired RUNNING leases -> FAILED(WORKER_INTERRUPTED)."""
        ...


class IdempotencyStore(Protocol):
    def lookup(self, route: str, case_id: str | None, key: str) -> IdempotencyRecord | None: ...
    def save(self, record: IdempotencyRecord) -> IdempotencyRecord:
        """Insert-or-return-existing; raises IdempotencyConflict when request_sha256 differs."""
        ...

    def purge_expired(self) -> int: ...


class Repository(Protocol):
    def put_case(self, case: CaseWorkspace) -> None: ...
    def get_case(self, case_id: str) -> CaseWorkspace: ...
    def put_video(self, video: SourceVideo) -> None: ...
    def get_video(self, video_id: str) -> SourceVideo: ...
    def list_videos(self, case_id: str) -> Sequence[SourceVideo]: ...
    def put_manifest(self, manifest: FrameManifest) -> None: ...
    def get_manifest(self, video_id: str) -> FrameManifest: ...
    def put_segments(self, segments: Sequence[VideoSegment]) -> None: ...
    def list_segments(self, video_id: str) -> Sequence[VideoSegment]: ...
    def put_embeddings(self, video_id: str, embeddings: Sequence[Embedding]) -> None: ...
    def list_embeddings(self, video_id: str) -> Sequence[Embedding]: ...
    def put_detections(self, detections: Sequence[Detection]) -> None: ...
    def list_detections(self, video_id: str) -> Sequence[Detection]: ...
    def put_search(self, search: SearchEvidence) -> None: ...
    def get_search(self, search_id: str) -> SearchEvidence: ...
    def put_track(self, track: Track) -> None: ...
    def get_track(self, track_id: str) -> Track: ...
    def put_run(self, run: ReconstructionRun) -> None: ...
    def get_run(self, run_id: str) -> ReconstructionRun: ...
    def put_decisions(self, decisions: Sequence[PolicyDecision]) -> None: ...
    def list_decisions(self, run_id: str) -> Sequence[PolicyDecision]: ...
    def put_provenance(self, provenance: PixelProvenance) -> None: ...
    def get_provenance(self, run_id: str) -> PixelProvenance: ...
    def put_review(self, review: HumanReview) -> None: ...
    def latest_review(self, run_id: str) -> HumanReview | None: ...
    def put_report(self, report: EvidenceReport) -> None: ...
    def get_report(self, report_id: str) -> EvidenceReport: ...
    def put_asset(self, asset: AssetRef) -> None: ...
    def get_asset(self, asset_id: str) -> AssetRef: ...
    def put_lineage(self, record: LineageRecord) -> None: ...


class JobContext(Protocol):
    @property
    def cancel(self) -> CancelToken: ...
    def progress(self, stage: JobStage, completed: int, total: int) -> None: ...


class JobHandler(Protocol):
    kind: JobKind

    async def run(self, claimed: ClaimedJob, ctx: JobContext) -> JobOutcome: ...


# --------------------------------------------------------------------------------------------
# Search service (Agent C) consumed by the API (Agent D)
# --------------------------------------------------------------------------------------------


class SearchQuery(FrozenModel):
    query: Annotated[str, Field(min_length=1, max_length=500)]
    max_results: Annotated[int, Field(ge=1, le=5)] = 5
    time_range: TimeRangeUs | None = None


class SearchService(Protocol):
    async def search(
        self, video: SourceVideo, query: SearchQuery, correlation_id: str
    ) -> SearchEvidence: ...
