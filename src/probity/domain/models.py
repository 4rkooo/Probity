"""Frozen Pydantic v2 domain contracts (schema_version 1.0).

Rules (section 6):
- Every model is ``frozen=True, extra="forbid"``; collections are tuples, never lists/dicts.
- Mutable business entities use lowercase UUIDv7 strings. Frame and segment IDs are deterministic.
- Wall-clock values are RFC 3339 UTC strings. Video time is integer microseconds (``*_pts_us``).
- Confidence and normalized coordinates are in ``[0, 1]``. Pixel boxes are half-open integers.
- Every ``Record`` carries ``schema_version``, ``created_at`` and ``content_sha256``; the hash is
  the SHA-256 of canonical JSON of the record with the top-level ``content_sha256`` field
  excluded and is verified on every validation. Build records with ``Model.create(...)``.

Changing any field here is a contract change and requires the team interface review.
"""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    model_validator,
)

from probity.domain.enums import (
    AdapterMode,
    AlignmentMethod,
    AssetKind,
    CasePurpose,
    CaseStatus,
    ErrorCode,
    ExplanationSource,
    HealthStatus,
    IndexState,
    InferenceMode,
    IngestState,
    Interpolation,
    JobKind,
    JobStage,
    JobState,
    Legibility,
    LineageKind,
    ObservationSource,
    OperatingMode,
    PolicyOutcome,
    PolicyStage,
    ProvenanceClass,
    ReasonCode,
    ReconstructionState,
    ReviewDecision,
    RigidSubjectKind,
    SearchStatus,
    SourceRole,
    SubjectType,
    TrackState,
    VetoReason,
    VisibilityKind,
)
from probity.domain.ids import (
    FRAME_ID_RE,
    RFC3339_UTC_RE,
    SEGMENT_ID_RE,
    UUID7_RE,
    canonical_sha256,
    is_uuid7,
    parse_frame_id,
    parse_segment_id,
    utc_now,
)

SCHEMA_VERSION = "1.0"

# --------------------------------------------------------------------------------------------
# Scalar types
# --------------------------------------------------------------------------------------------

Uuid7 = Annotated[
    str, Field(pattern=UUID7_RE.pattern, examples=["0199a51e-43bf-7aa2-86e1-b2a0bb287492"])
]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
UtcTimestamp = Annotated[
    str, Field(pattern=RFC3339_UTC_RE.pattern, examples=["2026-10-09T16:00:00.000000Z"])
]
PtsUs = Annotated[int, Field(ge=0, description="Presentation timestamp in integer microseconds")]
DurationUs = Annotated[int, Field(gt=0, description="Duration in integer microseconds")]
Confidence = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
UnitScore = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
Percent = Annotated[float, Field(ge=0.0, le=100.0, allow_inf_nan=False)]
NonNegInt = Annotated[int, Field(ge=0)]
PosInt = Annotated[int, Field(gt=0)]
ModelId = Annotated[str, Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:/@+-]+$")]
ShortText = Annotated[str, Field(min_length=1, max_length=500)]
Rational = Annotated[str, Field(pattern=r"^[1-9]\d*/[1-9]\d*$", examples=["30000/1001"])]
AssetUri = Annotated[
    str,
    Field(
        pattern=r"^asset://[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        description="Opaque, case-scoped asset reference served by GET /v1/assets/{asset_id}",
    ),
]
SourceStorageUri = Annotated[
    str,
    Field(
        pattern=r"^source/sha256/[0-9a-f]{2}/[0-9a-f]{64}/original\.mp4$",
        description="Content-addressed, write-once source path relative to the data root",
    ),
]
DerivedStorageUri = Annotated[
    str,
    Field(
        pattern=r"^derived/[0-9a-f-]{36}/[a-z_]+/[A-Za-z0-9:._-]+/[A-Za-z0-9._-]+$",
        description="derived/{video_id}/{kind}/{entity_id}/{file}; never beneath source/",
    ),
]


def _check_frame_id(value: str) -> str:
    parse_frame_id(value)
    return value


def _check_segment_id(value: str) -> str:
    parse_segment_id(value)
    return value


FrameId = Annotated[
    str,
    Field(pattern=FRAME_ID_RE.pattern, examples=["0199a51e-43bf-7aa2-86e1-b2a0bb287492:f417"]),
    AfterValidator(_check_frame_id),
]
SegmentId = Annotated[
    str,
    Field(pattern=SEGMENT_ID_RE.pattern, examples=["0199a51e-43bf-7aa2-86e1-b2a0bb287492:s0004"]),
    AfterValidator(_check_segment_id),
]

ScalarValue = float | int | str | bool | None


def _check_bbox_px(value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = value
    if min(value) < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError(
            "bbox_px must be half-open [x1,y1,x2,y2) with 0 <= x1 < x2 and 0 <= y1 < y2"
        )
    return value


def _check_bbox_norm(value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = value
    if any(not math.isfinite(v) or v < 0.0 or v > 1.0 for v in value) or x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_norm must be [x1,y1,x2,y2] within [0,1] with x1 < x2 and y1 < y2")
    return value


BBoxPx = Annotated[tuple[int, int, int, int], AfterValidator(_check_bbox_px)]
BBoxNorm = Annotated[tuple[float, float, float, float], AfterValidator(_check_bbox_norm)]

# --------------------------------------------------------------------------------------------
# Base classes
# --------------------------------------------------------------------------------------------


class FrozenModel(BaseModel):
    """Immutable value object."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Record(FrozenModel):
    """Serialized evidence/business entity with a verified content hash."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    created_at: UtcTimestamp
    content_sha256: Sha256Hex

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"content_sha256"})

    def compute_content_sha256(self) -> str:
        return canonical_sha256(self.canonical_payload())

    @model_validator(mode="after")
    def _verify_content_sha256(self, info: ValidationInfo) -> Self:
        if info.context and info.context.get("_probity_skip_content_hash"):
            return self
        expected = self.compute_content_sha256()
        if self.content_sha256 != expected:
            raise ValueError(
                f"content_sha256 mismatch for {type(self).__name__}: "
                f"declared {self.content_sha256}, computed {expected}"
            )
        return self

    @classmethod
    def create(cls, **fields: Any) -> Self:
        """Validate ``fields``, fill ``schema_version``/``created_at`` and compute the hash."""
        data = dict(fields)
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("created_at", utc_now())
        data["content_sha256"] = "0" * 64
        draft = cls.model_validate(data, context={"_probity_skip_content_hash": True})
        data["content_sha256"] = draft.compute_content_sha256()
        return cls.model_validate(data)

    def revise(self, **changes: Any) -> Self:
        """Return a new validated revision with ``changes`` applied and a recomputed hash."""
        data = self.model_dump(mode="python", exclude={"content_sha256"})
        data.update(changes)
        return type(self).create(**data)


def _video_of_frame(frame: str) -> str:
    return parse_frame_id(frame)[0]


# --------------------------------------------------------------------------------------------
# Shared value objects
# --------------------------------------------------------------------------------------------


class TimeRangeUs(FrozenModel):
    start_pts_us: PtsUs
    end_pts_us: PtsUs

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end_pts_us <= self.start_pts_us:
            raise ValueError("end_pts_us must be greater than start_pts_us")
        return self


class ClassConfidence(FrozenModel):
    class_name: Annotated[str, Field(min_length=1, max_length=64)]
    max_confidence: Confidence
    detection_count: NonNegInt


class AdapterHealth(FrozenModel):
    adapter_name: Annotated[str, Field(min_length=1, max_length=64)]
    model_id: ModelId | None
    mode: AdapterMode
    status: HealthStatus
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    checked_at: UtcTimestamp
    latency_ms: NonNegInt
    detail: Annotated[str, Field(max_length=500)] | None = None


class AssetRef(FrozenModel):
    """Registered servable artifact. ``non_evidentiary`` assets are never reconstruction input."""

    asset_id: Uuid7
    case_id: Uuid7
    kind: AssetKind
    storage_uri: Annotated[str, Field(min_length=1, max_length=400)]
    media_type: Annotated[str, Field(pattern=r"^[a-z]+/[a-z0-9.+-]+$")]
    byte_length: NonNegInt
    sha256: Sha256Hex
    non_evidentiary: bool
    created_at: UtcTimestamp

    @model_validator(mode="after")
    def _paths(self) -> Self:
        if self.kind is AssetKind.SOURCE_VIDEO:
            if not re.match(
                r"^source/sha256/[0-9a-f]{2}/[0-9a-f]{64}/original\.mp4$", self.storage_uri
            ):
                raise ValueError("SOURCE_VIDEO asset must use the content-addressed source path")
        elif not self.storage_uri.startswith("derived/") and not self.storage_uri.startswith(
            "fixtures/"
        ):
            raise ValueError("derived assets must live under derived/ (or committed fixtures/)")
        if ".." in self.storage_uri.split("/"):
            raise ValueError("path traversal is not allowed")
        if self.kind in {AssetKind.BASELINE, AssetKind.THUMBNAIL, AssetKind.INSPECTION_CLIP}:
            if not self.non_evidentiary:
                raise ValueError(f"{self.kind} assets must be marked non_evidentiary")
        return self


# --------------------------------------------------------------------------------------------
# Case, source, frames, segments
# --------------------------------------------------------------------------------------------


class CaseWorkspace(Record):
    case_id: Uuid7
    display_name: Annotated[str, Field(min_length=1, max_length=120)]
    purpose: Literal[CasePurpose.DEMO_RESEARCH] = CasePurpose.DEMO_RESEARCH
    mode: OperatingMode
    owner_alias: Annotated[str, Field(min_length=1, max_length=64)]
    status: CaseStatus


class SourceVideo(Record):
    video_id: Uuid7
    case_id: Uuid7
    original_name: Annotated[
        str,
        Field(min_length=1, max_length=255, description="Client-supplied; display only, untrusted"),
    ]
    sha256: Sha256Hex
    byte_length: PosInt
    mime: Literal["video/mp4"] = Field(description="Detected by probe, never taken from the client")
    container: Literal["mp4"]
    video_codec: Literal["h264", "hevc"]
    width_px: Annotated[int, Field(gt=0, le=1920)]
    height_px: Annotated[int, Field(gt=0, le=1080)]
    duration_us: Annotated[int, Field(gt=0, le=600_000_000)]
    time_base: Rational
    nominal_fps: Rational = Field(description="Display only; never used to derive frame identity")
    frame_count: PosInt
    has_audio: bool
    storage_uri: SourceStorageUri
    source_asset_uri: AssetUri
    probe_sha256: Sha256Hex = Field(description="SHA-256 of canonical JSON of the ffprobe result")
    ingest_state: IngestState
    indexed_range: TimeRangeUs | None = None
    license_note: Annotated[str, Field(max_length=500)] | None = None
    fixture_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")] | None = None

    @model_validator(mode="after")
    def _storage_matches_hash(self) -> Self:
        expected = f"source/sha256/{self.sha256[:2]}/{self.sha256}/original.mp4"
        if self.storage_uri != expected:
            raise ValueError(f"storage_uri must be {expected}")
        if self.ingest_state is IngestState.PARTIAL and self.indexed_range is None:
            raise ValueError("PARTIAL ingest requires indexed_range")
        return self


class FrameManifestEntry(FrozenModel):
    frame_number: NonNegInt = Field(description="Decoder-output (presentation) order index")
    pts_us: PtsUs
    pts: int = Field(description="Raw stream PTS in time_base units")
    is_keyframe: bool
    width_px: PosInt
    height_px: PosInt


class FrameManifest(Record):
    """Canonical exact-PTS frame manifest.

    ``pts_us = (pts * tb_num * 1_000_000) // tb_den`` in exact integer arithmetic. Video duration is
    computed in stream ticks first: ``end_ticks = last.pts + frame_duration_ticks`` (packet
    duration, else the last PTS delta), then converted with the same floor rule.
    Never ``fps * seconds``.
    """

    video_id: Uuid7
    source_sha256: Sha256Hex
    time_base: Rational
    extractor: Annotated[
        str, Field(min_length=1, max_length=200, description="ffprobe version string")
    ]
    frames: Annotated[tuple[FrameManifestEntry, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _monotonic(self) -> Self:
        prev: FrameManifestEntry | None = None
        for i, entry in enumerate(self.frames):
            if entry.frame_number != i:
                raise ValueError("frame_number must be contiguous from 0")
            if prev is not None and (entry.pts_us <= prev.pts_us or entry.pts <= prev.pts):
                raise ValueError("presentation timestamps must be strictly increasing")
            prev = entry
        return self


class FrameReference(Record):
    frame_id: FrameId
    video_id: Uuid7
    frame_number: NonNegInt
    pts_us: PtsUs
    is_keyframe: bool
    width_px: PosInt
    height_px: PosInt
    lossless_png_uri: DerivedStorageUri | None = None
    pixel_sha256: Sha256Hex | None = None
    decoder: Annotated[str, Field(max_length=200)] | None = None
    pixel_format: Annotated[str, Field(max_length=32)] | None = None

    @model_validator(mode="after")
    def _identity(self) -> Self:
        vid, n = parse_frame_id(self.frame_id)
        if vid != self.video_id or n != self.frame_number:
            raise ValueError("frame_id must equal {video_id}:f{frame_number}")
        if self.lossless_png_uri is not None:
            parts = self.lossless_png_uri.split("/")
            if parts[1] != self.video_id or parts[2] != "frames":
                raise ValueError("lossless_png_uri must be derived/{video_id}/frames/...")
            if not self.lossless_png_uri.endswith(".png"):
                raise ValueError("canonical frames are lossless PNG")
            if self.pixel_sha256 is None:
                raise ValueError("pixel_sha256 is required when a PNG is extracted")
        return self


class RigidSubjectObservation(FrozenModel):
    kind: RigidSubjectKind
    color: Annotated[str, Field(max_length=40)] | None = None
    location: Annotated[str, Field(min_length=1, max_length=200)]
    approximate_size: Annotated[str, Field(max_length=100)] | None = None
    orientation: Annotated[str, Field(max_length=100)] | None = None
    legibility: Legibility
    quoted_text: Annotated[str, Field(max_length=32)] | None = Field(
        default=None, description="Only when every quoted character is visibly supported"
    )
    uncertainty: Annotated[str, Field(max_length=300)] | None = None
    has_multiple_clearer_observations: bool | None = None
    confidence: Confidence
    time_range: TimeRangeUs

    @model_validator(mode="after")
    def _quote_rule(self) -> Self:
        if self.quoted_text is not None and self.legibility is not Legibility.CLEAR:
            raise ValueError("quoted_text is only allowed when legibility is CLEAR")
        return self


class VisibilityLimit(FrozenModel):
    kind: VisibilityKind
    observable_source: Annotated[str, Field(max_length=200)] | None = None
    confidence: Confidence
    time_range: TimeRangeUs | None = None


class ObservedAction(FrozenModel):
    description: Annotated[str, Field(min_length=1, max_length=200)]
    direction: Annotated[str, Field(max_length=60)] | None = None
    confidence: Confidence
    time_range: TimeRangeUs


class SegmentDescription(FrozenModel):
    """Validated structured Cosmos output for one segment (section 8 ingestion prompt)."""

    segment_id: SegmentId
    video_id: Uuid7
    source_sha256: Sha256Hex
    time_range: TimeRangeUs
    model_id: ModelId
    prompt_sha256: Sha256Hex
    raw_output_sha256: Sha256Hex
    summary: Annotated[str, Field(min_length=1, max_length=1000)]
    rigid_subjects: tuple[RigidSubjectObservation, ...] = ()
    actions: tuple[ObservedAction, ...] = ()
    visibility: tuple[VisibilityLimit, ...] = ()
    scene_changes: tuple[Annotated[str, Field(max_length=200)], ...] = ()
    search_terms: tuple[Annotated[str, Field(min_length=1, max_length=40)], ...] = ()
    uncertainty: tuple[Annotated[str, Field(max_length=300)], ...] = ()
    confidence: Confidence
    mode: InferenceMode

    @model_validator(mode="after")
    def _within_segment(self) -> Self:
        if parse_segment_id(self.segment_id)[0] != self.video_id:
            raise ValueError("segment_id must belong to video_id")
        lo, hi = self.time_range.start_pts_us, self.time_range.end_pts_us
        ranges = [s.time_range for s in self.rigid_subjects] + [a.time_range for a in self.actions]
        ranges += [v.time_range for v in self.visibility if v.time_range is not None]
        for r in ranges:
            if r.start_pts_us < lo or r.end_pts_us > hi:
                raise ValueError("observation time ranges must lie within the segment")
        return self


class Embedding(FrozenModel):
    embedding_ref: Annotated[str, Field(min_length=1, max_length=200)]
    model_id: ModelId
    dimension: PosInt
    vector: tuple[float, ...]
    normalized: bool
    input_sha256: Sha256Hex
    mode: InferenceMode

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if len(self.vector) != self.dimension:
            raise ValueError("vector length must equal dimension")
        if any(not math.isfinite(v) for v in self.vector):
            raise ValueError("vector must be finite")
        norm = math.sqrt(sum(v * v for v in self.vector))
        if norm == 0.0:
            raise ValueError("zero vector is not a valid embedding")
        if self.normalized and abs(norm - 1.0) > 1e-4:
            raise ValueError("normalized embedding must have unit L2 norm")
        return self


class VideoSegment(Record):
    segment_id: SegmentId
    video_id: Uuid7
    source_sha256: Sha256Hex
    ordinal: NonNegInt
    start_pts_us: PtsUs
    end_pts_us: PtsUs
    start_frame: NonNegInt
    end_frame: NonNegInt = Field(description="Inclusive last frame with pts_us < end_pts_us")
    clip_uri: AssetUri | None = None
    thumbnail_uri: AssetUri | None = Field(
        default=None, description="480 px-wide JPEG, non-canonical"
    )
    thumbnail_frame: NonNegInt | None = None
    description: Annotated[str, Field(max_length=1000)] | None = None
    description_model_id: ModelId | None = None
    search_terms: tuple[Annotated[str, Field(min_length=1, max_length=40)], ...] = ()
    detected_classes: tuple[ClassConfidence, ...] = ()
    visibility_tags: tuple[VisibilityKind, ...] = ()
    evidence_detection_ids: tuple[Uuid7, ...] = ()
    uncertainty: tuple[Annotated[str, Field(max_length=300)], ...] = ()
    embedding_ref: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    embedding_model_id: ModelId | None = None
    embedding_dimension: PosInt | None = None
    index_state: IndexState
    mode: InferenceMode

    @model_validator(mode="after")
    def _identity(self) -> Self:
        vid, n = parse_segment_id(self.segment_id)
        if vid != self.video_id or n != self.ordinal:
            raise ValueError("segment_id must equal {video_id}:s{ordinal:04d}")
        if self.end_pts_us <= self.start_pts_us:
            raise ValueError("end_pts_us must be greater than start_pts_us")
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be >= start_frame")
        if self.thumbnail_frame is not None and not (
            self.start_frame <= self.thumbnail_frame <= self.end_frame
        ):
            raise ValueError("thumbnail_frame must lie within the segment")
        if self.index_state in {IndexState.DESCRIBED, IndexState.EMBEDDED, IndexState.INDEXED}:
            if self.description is None or self.description_model_id is None:
                raise ValueError("described segments require description and description_model_id")
        if self.index_state in {IndexState.EMBEDDED, IndexState.INDEXED}:
            if None in (self.embedding_ref, self.embedding_model_id, self.embedding_dimension):
                raise ValueError("embedded segments require embedding_ref/model_id/dimension")
        return self

    @property
    def time_range(self) -> TimeRangeUs:
        return TimeRangeUs(start_pts_us=self.start_pts_us, end_pts_us=self.end_pts_us)


# --------------------------------------------------------------------------------------------
# Detection and tracking (Person 2 consumes/produces)
# --------------------------------------------------------------------------------------------


class Detection(Record):
    detection_id: Uuid7
    video_id: Uuid7
    frame_id: FrameId
    pts_us: PtsUs
    model_id: ModelId
    model_sha256: Sha256Hex | None
    class_id: NonNegInt
    class_name: Annotated[str, Field(min_length=1, max_length=64)]
    confidence: Confidence
    bbox_px: BBoxPx
    bbox_norm: BBoxNorm
    occlusion_score: Confidence | None = None
    inference_mode: InferenceMode

    @model_validator(mode="after")
    def _frame_video(self) -> Self:
        if _video_of_frame(self.frame_id) != self.video_id:
            raise ValueError("frame_id must belong to video_id")
        return self


class TrackObservation(FrozenModel):
    frame_id: FrameId
    pts_us: PtsUs
    bbox_px: BBoxPx
    source: ObservationSource
    detection_id: Uuid7 | None
    confidence: Confidence
    accepted: bool
    reason_code: ReasonCode | None = None

    @model_validator(mode="after")
    def _source_rules(self) -> Self:
        if self.source is ObservationSource.DETECTOR and self.detection_id is None:
            raise ValueError("DETECTOR observations must reference a detection_id")
        if self.source is not ObservationSource.DETECTOR and self.detection_id is not None:
            raise ValueError("only DETECTOR observations may reference a detection_id")
        if not self.accepted and self.reason_code is None:
            raise ValueError("rejected observations require a reason_code")
        return self


class Track(Record):
    track_id: Uuid7
    case_id: Uuid7
    video_id: Uuid7
    subject_type: SubjectType
    seed_frame_id: FrameId
    seed_bbox_px: BBoxPx
    seed_detection_id: Uuid7 | None = None
    tracker: Literal["bytetrack"] = "bytetrack"
    tracker_version: Annotated[str, Field(min_length=1, max_length=64)]
    detector_model_id: ModelId | None = None
    window_start_us: PtsUs
    window_end_us: PtsUs
    state: TrackState
    mean_confidence: Confidence
    continuity_score: UnitScore
    confirmed: bool
    detector_observation_count: NonNegInt
    observations: tuple[TrackObservation, ...]
    reason_codes: tuple[ReasonCode, ...] = ()
    mode: InferenceMode

    @model_validator(mode="after")
    def _consistency(self) -> Self:
        if self.window_end_us <= self.window_start_us:
            raise ValueError("window_end_us must be greater than window_start_us")
        if _video_of_frame(self.seed_frame_id) != self.video_id:
            raise ValueError("seed_frame_id must belong to video_id")
        for obs in self.observations:
            if _video_of_frame(obs.frame_id) != self.video_id:
                raise ValueError("observations must belong to the track's video")
            if not (self.window_start_us <= obs.pts_us <= self.window_end_us):
                raise ValueError("observation outside track window")
        counted = sum(
            1 for o in self.observations if o.accepted and o.source is ObservationSource.DETECTOR
        )
        if counted != self.detector_observation_count:
            raise ValueError("detector_observation_count must equal accepted DETECTOR observations")
        if self.confirmed != (self.state is TrackState.CONFIRMED):
            raise ValueError("confirmed must be true exactly when state is CONFIRMED")
        if self.state is TrackState.NOT_CONFIRMED and not self.reason_codes:
            raise ValueError("NOT_CONFIRMED tracks require reason_codes")
        return self


# --------------------------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------------------------


class SearchPlan(FrozenModel):
    semantic_query: Annotated[str, Field(min_length=0, max_length=300)]
    objective_terms: tuple[Annotated[str, Field(min_length=1, max_length=40)], ...] = ()
    subject_classes: tuple[Annotated[str, Field(min_length=1, max_length=64)], ...] = ()
    time_range: TimeRangeUs | None = None
    needs_clarification: bool = False
    filtered_terms: tuple[Annotated[str, Field(min_length=1, max_length=60)], ...] = ()
    policy_reason_codes: tuple[ReasonCode, ...] = ()

    @model_validator(mode="after")
    def _rules(self) -> Self:
        if not self.semantic_query.strip() and not self.needs_clarification:
            raise ValueError("an empty semantic_query requires needs_clarification=true")
        if self.filtered_terms and ReasonCode.QUERY_POLICY_FILTERED not in self.policy_reason_codes:
            raise ValueError("filtered_terms require QUERY_POLICY_FILTERED")
        return self


class ScoreComponents(FrozenModel):
    """Components clamped to [0,1]; unavailable optional evidence contributes 0 and is listed."""

    cosine: UnitScore
    lexical_overlap: UnitScore
    detection_match: UnitScore
    visibility_match: UnitScore
    unavailable: tuple[Literal["lexical_overlap", "detection_match", "visibility_match"], ...] = ()


class SearchCandidate(FrozenModel):
    segment_id: SegmentId
    vector_rank: PosInt
    similarity: Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]
    components: ScoreComponents
    score: UnitScore
    final_rank: PosInt | None = None
    collapsed_into: SegmentId | None = None


class SearchResult(FrozenModel):
    rank: Annotated[int, Field(ge=1, le=5)]
    segment_id: SegmentId
    video_id: Uuid7
    source_sha256: Sha256Hex
    score: UnitScore
    components: ScoreComponents
    start_pts_us: PtsUs
    end_pts_us: PtsUs
    start_frame: NonNegInt
    end_frame: NonNegInt
    thumbnail_uri: AssetUri | None
    description: Annotated[str, Field(max_length=1000)]
    detected_classes: tuple[ClassConfidence, ...] = ()
    evidence_detection_ids: tuple[Uuid7, ...] = ()
    explanation: Annotated[str, Field(min_length=1, max_length=600)]
    explanation_source: ExplanationSource
    explanation_input_hash: Sha256Hex
    outside_indexed_range_warning: bool = False

    @model_validator(mode="after")
    def _identity(self) -> Self:
        if parse_segment_id(self.segment_id)[0] != self.video_id:
            raise ValueError("segment_id must belong to video_id")
        if self.end_pts_us <= self.start_pts_us:
            raise ValueError("end_pts_us must be greater than start_pts_us")
        return self


class SearchEvidence(Record):
    search_id: Uuid7
    case_id: Uuid7
    video_id: Uuid7
    query: Annotated[str, Field(min_length=1, max_length=500)]
    query_plan: SearchPlan
    planner_model_id: ModelId
    embedding_model_id: ModelId
    reasoner_model_id: ModelId | None
    status: SearchStatus
    mode: InferenceMode
    indexed_range: TimeRangeUs | None
    candidates: tuple[SearchCandidate, ...]
    results: Annotated[tuple[SearchResult, ...], Field(max_length=5)]
    latency_ms: NonNegInt
    correlation_id: Uuid7

    @model_validator(mode="after")
    def _ranks(self) -> Self:
        ranks = [r.rank for r in self.results]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("results must be ranked 1..n without gaps")
        if len({r.segment_id for r in self.results}) != len(self.results):
            raise ValueError("duplicate segment in results")
        scores = [r.score for r in self.results]
        if scores != sorted(scores, reverse=True):
            raise ValueError("results must be ordered by descending score")
        for r in self.results:
            if r.video_id != self.video_id:
                raise ValueError("results must belong to the searched video")
        candidate_ids = {c.segment_id for c in self.candidates}
        if not {r.segment_id for r in self.results} <= candidate_ids:
            raise ValueError("every result must come from the persisted candidates")
        if self.status is SearchStatus.OK and not self.results:
            raise ValueError("status OK requires at least one result")
        if self.status is not SearchStatus.OK and self.results:
            raise ValueError("only status OK may carry results")
        if (
            self.status is SearchStatus.NEEDS_CLARIFICATION
            and not self.query_plan.needs_clarification
        ):
            raise ValueError("NEEDS_CLARIFICATION status requires a plan that needs clarification")
        return self


class RetrievedSegment(FrozenModel):
    """EvidenceStore.search output: one candidate with its raw vector similarity."""

    segment: VideoSegment
    similarity: Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]
    vector_rank: PosInt


class GroundedExplanation(FrozenModel):
    segment_id: SegmentId
    text: Annotated[str, Field(min_length=1, max_length=600)]
    cited_detection_ids: tuple[Uuid7, ...] = ()
    cited_pts_us: tuple[PtsUs, ...] = ()
    source: ExplanationSource
    model_id: ModelId | None
    input_sha256: Sha256Hex


# --------------------------------------------------------------------------------------------
# Reconstruction, policy, provenance, integrity (Person 2 produces; Person 3 consumes)
# --------------------------------------------------------------------------------------------


class PolicyDecision(Record):
    decision_id: Uuid7
    run_id: Uuid7
    sequence: NonNegInt
    rule_code: ReasonCode
    stage: PolicyStage
    subject_ref: Annotated[str, Field(min_length=1, max_length=120)]
    outcome: PolicyOutcome
    reason: Annotated[str, Field(min_length=1, max_length=300)]
    observed: ScalarValue
    operator: Literal["<", "<=", ">", ">=", "==", "!=", "in", "none"]
    threshold: ScalarValue
    units: Annotated[str, Field(max_length=32)] | None
    policy_key: Annotated[str, Field(max_length=80)] | None = None


class SourceLutEntry(FrozenModel):
    index: NonNegInt
    frame_id: FrameId
    frame_number: NonNegInt
    pts_us: PtsUs
    role: SourceRole
    transform_id: Annotated[str, Field(max_length=64)] | None = None
    alignment_method: AlignmentMethod
    matrix: tuple[float, ...] | None = Field(
        default=None, description="Row-major 3x3 homography (9) or 2x3 affine (6), donor->target"
    )
    interpolation: Interpolation
    color_gain: tuple[float, float, float] | None = None
    color_bias: tuple[float, float, float] | None = None
    decision_ids: tuple[Uuid7, ...] = ()

    @model_validator(mode="after")
    def _roles(self) -> Self:
        if self.index == 0 and self.role is not SourceRole.TARGET:
            raise ValueError("source_lut[0] must be the TARGET frame")
        if self.role is SourceRole.TARGET:
            if self.alignment_method is not AlignmentMethod.IDENTITY or self.matrix is not None:
                raise ValueError("TARGET entry uses the identity transform")
        else:
            expected = {AlignmentMethod.AKAZE_HOMOGRAPHY: 9, AlignmentMethod.ECC_AFFINE: 6}
            if self.alignment_method not in expected or self.matrix is None:
                raise ValueError("DONOR entries require an AKAZE homography or ECC affine matrix")
            if len(self.matrix) != expected[self.alignment_method]:
                raise ValueError("matrix length does not match alignment_method")
        return self


class CoverageCounts(FrozenModel):
    ORIGINAL: NonNegInt
    BORROWED: NonNegInt
    GENERATED_BLEND: NonNegInt


class CoveragePct(FrozenModel):
    ORIGINAL: Percent
    BORROWED: Percent
    GENERATED_BLEND: Percent


class PixelProvenance(Record):
    """Authority is provenance.npz (class/source_index/source_x/source_y arrays) plus source_lut.

    ``coverage_counts``/``coverage_pct`` cover the full output frame (sum = width*height);
    ``subject_coverage_pct`` covers only ``subject_bbox_px``.
    """

    run_id: Uuid7
    width_px: PosInt
    height_px: PosInt
    tile_size_px: Literal[8] = 8
    subject_bbox_px: BBoxPx
    class_map_uri: DerivedStorageUri
    source_index_uri: DerivedStorageUri
    source_xy_uri: DerivedStorageUri
    source_lut: Annotated[tuple[SourceLutEntry, ...], Field(min_length=1)]
    coverage_counts: CoverageCounts
    coverage_pct: CoveragePct
    subject_coverage_pct: CoveragePct
    coverage_complete: bool
    encoding_version: Literal["npz-pixel-v1"] = "npz-pixel-v1"
    artifact_sha256: Sha256Hex

    @model_validator(mode="after")
    def _coverage(self) -> Self:
        c = self.coverage_counts
        total = c.ORIGINAL + c.BORROWED + c.GENERATED_BLEND
        frame_px = self.width_px * self.height_px
        if self.coverage_complete != (total == frame_px):
            raise ValueError("coverage_complete must equal (counts total == width*height)")
        for name in ("ORIGINAL", "BORROWED", "GENERATED_BLEND"):
            expected = 100.0 * getattr(c, name) / frame_px
            if abs(getattr(self.coverage_pct, name) - expected) > 0.005:
                raise ValueError(f"coverage_pct.{name} must equal its count / (width*height)")
        for i, entry in enumerate(self.source_lut):
            if entry.index != i:
                raise ValueError("source_lut indices must be contiguous from 0")
        x1, y1, x2, y2 = self.subject_bbox_px
        if x2 > self.width_px or y2 > self.height_px:
            raise ValueError("subject_bbox_px must lie within the frame")
        return self


class ProvenanceSummary(FrozenModel):
    """Embedded in ReconstructionRun; PixelProvenance + provenance.npz are the authority."""

    encoding_version: Literal["npz-pixel-v1"] = "npz-pixel-v1"
    tile_size_px: Literal[8] = 8
    coverage_pct: CoveragePct
    subject_coverage_pct: CoveragePct
    coverage_complete: bool
    source_lut: Annotated[tuple[SourceLutEntry, ...], Field(min_length=1)]


def integrity_score_0_100(
    supported_coverage: float,
    mean_source_confidence: float,
    mean_alignment_confidence: float,
    track_continuity: float,
    audit_completeness: float,
    semantic_generated_pct: float,
) -> int:
    """integrity-v1: round_half_up(100*(0.30C+0.25Qd+0.20Ad+0.15Tc+0.10Au)*(1-G))."""
    weighted = (
        Decimal("0.30") * Decimal(repr(supported_coverage))
        + Decimal("0.25") * Decimal(repr(mean_source_confidence))
        + Decimal("0.20") * Decimal(repr(mean_alignment_confidence))
        + Decimal("0.15") * Decimal(repr(track_continuity))
        + Decimal("0.10") * Decimal(repr(audit_completeness))
    )
    g = Decimal(repr(semantic_generated_pct)) / Decimal(100)
    return int((Decimal(100) * weighted * (1 - g)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


class IntegrityScore(FrozenModel):
    score_0_100: Annotated[int, Field(ge=0, le=100)]
    supported_coverage: UnitScore
    mean_source_confidence: UnitScore
    mean_alignment_confidence: UnitScore
    track_continuity: UnitScore
    audit_completeness: UnitScore
    semantic_generated_pct: Percent
    formula_version: Literal["integrity-v1"] = "integrity-v1"

    @model_validator(mode="after")
    def _formula(self) -> Self:
        expected = integrity_score_0_100(
            self.supported_coverage,
            self.mean_source_confidence,
            self.mean_alignment_confidence,
            self.track_continuity,
            self.audit_completeness,
            self.semantic_generated_pct,
        )
        if self.score_0_100 != expected:
            raise ValueError(f"score_0_100 must equal the integrity-v1 formula result {expected}")
        return self


class ReconstructionRun(Record):
    run_id: Uuid7
    case_id: Uuid7
    video_id: Uuid7
    track_id: Uuid7
    target_frame_id: FrameId
    target_pts_us: PtsUs
    target_bbox_px: BBoxPx
    state: ReconstructionState
    policy_profile: Literal["demo-conservative-v1"] = "demo-conservative-v1"
    config_sha256: Sha256Hex
    algorithm_version: Literal["probity-tile-v1"] = "probity-tile-v1"
    iteration_count: Annotated[int, Field(ge=0, le=2)]
    accepted_donor_frame_ids: Annotated[tuple[FrameId, ...], Field(max_length=5)]
    result_png_uri: AssetUri | None = None
    inspection_clip_uri: AssetUri | None = None
    provenance_uri: AssetUri | None = None
    provenance: ProvenanceSummary | None = None
    result_png_sha256: Sha256Hex | None = None
    provenance_sha256: Sha256Hex | None = None
    policy_decision_ids: tuple[Uuid7, ...]
    integrity: IntegrityScore | None = None
    refusal_reasons: tuple[ReasonCode, ...] = ()
    uncertainty: tuple[Annotated[str, Field(min_length=1, max_length=300)], ...]
    mode: InferenceMode
    started_at: UtcTimestamp | None
    finished_at: UtcTimestamp | None

    @model_validator(mode="after")
    def _state_rules(self) -> Self:
        if _video_of_frame(self.target_frame_id) != self.video_id:
            raise ValueError("target_frame_id must belong to video_id")
        for fid in self.accepted_donor_frame_ids:
            if _video_of_frame(fid) != self.video_id:
                raise ValueError("donor frames must belong to the run's video")
            if fid == self.target_frame_id:
                raise ValueError("the target frame cannot be its own donor")
        artifacts = (
            self.result_png_uri,
            self.provenance_uri,
            self.provenance,
            self.result_png_sha256,
            self.provenance_sha256,
            self.integrity,
        )
        if self.state is ReconstructionState.SUCCEEDED:
            if any(a is None for a in artifacts):
                raise ValueError("SUCCEEDED runs require result, provenance, hashes, and integrity")
            assert self.provenance is not None
            assert self.integrity is not None
            if not self.provenance.coverage_complete:
                raise ValueError("SUCCEEDED runs require complete provenance")
            if self.provenance.coverage_pct.GENERATED_BLEND != 0.0:
                raise ValueError("default MVP never emits GENERATED_BLEND pixels")
            if self.integrity.semantic_generated_pct != 0.0:
                raise ValueError("semantic_generated_pct must be 0 in the default MVP")
            if len(self.accepted_donor_frame_ids) < 2:
                raise ValueError("SUCCEEDED runs require at least two accepted donors")
            if self.refusal_reasons:
                raise ValueError("SUCCEEDED runs carry no refusal_reasons")
            if self.finished_at is None:
                raise ValueError("terminal runs require finished_at")
        if self.state is ReconstructionState.REFUSED:
            if not self.refusal_reasons:
                raise ValueError("REFUSED runs require rule-coded refusal_reasons")
            if self.result_png_uri is not None or self.result_png_sha256 is not None:
                raise ValueError("REFUSED runs expose no result image")
            if self.finished_at is None:
                raise ValueError("terminal runs require finished_at")
        return self


class PixelOrigin(FrozenModel):
    run_id: Uuid7
    x: NonNegInt
    y: NonNegInt
    provenance_class: Literal["ORIGINAL", "BORROWED", "GENERATED_BLEND"]
    source_index: NonNegInt
    source_frame_id: FrameId
    source_pts_us: PtsUs
    source_x: float
    source_y: float
    transform_id: Annotated[str, Field(max_length=64)] | None
    alignment_method: AlignmentMethod
    color_gain: tuple[float, float, float] | None = None
    color_bias: tuple[float, float, float] | None = None
    decision_ids: tuple[Uuid7, ...] = ()

    @staticmethod
    def class_name(value: int) -> str:
        return ProvenanceClass(value).name


class HumanReview(Record):
    review_id: Uuid7
    run_id: Uuid7
    reviewer_alias: Annotated[str, Field(min_length=1, max_length=64)]
    decision: ReviewDecision
    reason_code: VetoReason | None
    comment: Annotated[str, Field(max_length=1000)]
    reviewed_result_sha256: Sha256Hex
    reviewed_provenance_sha256: Sha256Hex

    @model_validator(mode="after")
    def _veto_rules(self) -> Self:
        if self.decision is ReviewDecision.VETO:
            if self.reason_code is None or not self.comment.strip():
                raise ValueError("VETO requires a reason_code and a comment")
        elif self.reason_code is not None:
            raise ValueError("APPROVE carries no veto reason_code")
        return self


class EvidenceReport(Record):
    report_id: Uuid7
    case_id: Uuid7
    run_id: Uuid7
    review_id: Uuid7
    source_sha256: Sha256Hex
    source_sha256_verified_at_export: Sha256Hex
    source_verified: bool
    html_uri: AssetUri
    json_uri: AssetUri
    bundle_sha256: Sha256Hex
    generated_at: UtcTimestamp
    draft_model_id: ModelId | None
    mode: InferenceMode
    limitations: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=500)], ...], Field(min_length=1)
    ]

    @model_validator(mode="after")
    def _verified(self) -> Self:
        if self.source_verified != (self.source_sha256 == self.source_sha256_verified_at_export):
            raise ValueError("source_verified must reflect the export-time re-hash comparison")
        if not self.source_verified:
            raise ValueError("a report cannot be created when the source hash does not verify")
        return self


class LineageRecord(Record):
    lineage_id: Uuid7
    kind: LineageKind
    entity_id: Annotated[str, Field(min_length=1, max_length=120)]
    video_id: Uuid7
    parent_sha256: Sha256Hex
    artifact_uri: Annotated[str, Field(min_length=1, max_length=400)]
    artifact_sha256: Sha256Hex
    algorithm_version: Annotated[str, Field(min_length=1, max_length=64)]
    config_sha256: Sha256Hex | None = None
    model_ids: tuple[ModelId, ...] = ()
    frame_ids: tuple[FrameId, ...] = ()
    mode: InferenceMode


# --------------------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------------------


class JobError(FrozenModel):
    code: ErrorCode
    message: Annotated[str, Field(min_length=1, max_length=500)]
    retryable: bool


class PartialCoverage(FrozenModel):
    committed_segment_ids: Annotated[tuple[SegmentId, ...], Field(min_length=1)]
    indexed_ranges: Annotated[tuple[TimeRangeUs, ...], Field(min_length=1)]
    failed_segment_ids: tuple[SegmentId, ...] = ()


class JobView(Record):
    job_id: Uuid7
    kind: JobKind
    case_id: Uuid7
    subject_id: Uuid7 = Field(description="video_id | track_id | run_id | report_id by kind")
    state: JobState
    stage: JobStage | None = None
    completed_units: NonNegInt = 0
    total_units: NonNegInt = 0
    attempt: PosInt = 1
    error: JobError | None = None
    refusal_reasons: tuple[ReasonCode, ...] = ()
    partial: PartialCoverage | None = None
    mode: InferenceMode
    correlation_id: Uuid7
    updated_at: UtcTimestamp
    started_at: UtcTimestamp | None = None
    finished_at: UtcTimestamp | None = None
    cancel_requested_at: UtcTimestamp | None = None

    @model_validator(mode="after")
    def _state_rules(self) -> Self:
        if self.completed_units > self.total_units:
            raise ValueError("completed_units cannot exceed total_units")
        if self.state is JobState.RUNNING and self.stage is None:
            raise ValueError("RUNNING jobs carry a stage")
        if self.state is JobState.PARTIAL:
            if self.kind is not JobKind.INGEST:
                raise ValueError("PARTIAL is valid only for INGEST jobs")
            if self.partial is None:
                raise ValueError("PARTIAL requires at least one committed searchable segment")
        elif self.partial is not None and self.state is not JobState.CANCELLED:
            raise ValueError("partial coverage is only reported for PARTIAL or CANCELLED jobs")
        if self.state is JobState.REFUSED:
            if self.kind not in {JobKind.TRACK, JobKind.RECONSTRUCT, JobKind.REPORT}:
                raise ValueError("REFUSED is reserved for policy/evidence insufficiency")
            if not self.refusal_reasons:
                raise ValueError("REFUSED requires rule-coded refusal_reasons")
        elif self.refusal_reasons:
            raise ValueError("only REFUSED jobs carry refusal_reasons")
        if self.state is JobState.FAILED and self.error is None:
            raise ValueError("FAILED jobs require an error")
        if (
            self.state in {JobState.CANCELLING, JobState.CANCELLED}
            and self.cancel_requested_at is None
        ):
            raise ValueError("cancellation states require cancel_requested_at")
        terminal = {
            JobState.SUCCEEDED,
            JobState.PARTIAL,
            JobState.REFUSED,
            JobState.FAILED,
            JobState.CANCELLED,
        }
        if (self.state in terminal) != (self.finished_at is not None):
            raise ValueError("finished_at is set exactly for terminal states")
        return self


class JobEvent(Record):
    event_id: Uuid7
    job_id: Uuid7
    sequence: NonNegInt
    from_state: JobState | None
    to_state: JobState
    stage: JobStage | None = None
    completed_units: NonNegInt = 0
    total_units: NonNegInt = 0
    error_code: ErrorCode | None = None
    reason_codes: tuple[ReasonCode, ...] = ()
    actor: Annotated[str, Field(min_length=1, max_length=64)]
    correlation_id: Uuid7


def assert_uuid7(value: str) -> str:
    if not is_uuid7(value):
        raise ValueError(f"not a lowercase UUIDv7: {value!r}")
    return value


__all__ = [
    "SCHEMA_VERSION",
    "AdapterHealth",
    "AssetRef",
    "BBoxNorm",
    "BBoxPx",
    "CaseWorkspace",
    "ClassConfidence",
    "Confidence",
    "CoverageCounts",
    "CoveragePct",
    "Detection",
    "Embedding",
    "EvidenceReport",
    "FrameId",
    "FrameManifest",
    "FrameManifestEntry",
    "FrameReference",
    "FrozenModel",
    "GroundedExplanation",
    "HumanReview",
    "IntegrityScore",
    "JobError",
    "JobEvent",
    "JobView",
    "LineageRecord",
    "ObservedAction",
    "PartialCoverage",
    "PixelOrigin",
    "PixelProvenance",
    "PolicyDecision",
    "ProvenanceSummary",
    "PtsUs",
    "ReconstructionRun",
    "Record",
    "RetrievedSegment",
    "RigidSubjectObservation",
    "ScoreComponents",
    "SearchCandidate",
    "SearchEvidence",
    "SearchPlan",
    "SearchResult",
    "SegmentDescription",
    "SegmentId",
    "Sha256Hex",
    "SourceLutEntry",
    "SourceVideo",
    "TimeRangeUs",
    "Track",
    "TrackObservation",
    "Uuid7",
    "VideoSegment",
    "VisibilityLimit",
    "integrity_score_0_100",
]
