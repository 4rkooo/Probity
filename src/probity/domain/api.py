"""Frozen HTTP request/response models (section 7). Domain records are returned as-is."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from probity.domain.enums import ErrorCode, OperatingMode, ReviewDecision, SubjectType, VetoReason
from probity.domain.models import (
    AdapterHealth,
    BBoxPx,
    FrameId,
    FrozenModel,
    Sha256Hex,
    TimeRangeUs,
    Uuid7,
)

IDEMPOTENCY_HEADER = "Idempotency-Key"
CORRELATION_HEADER = "X-Correlation-ID"
IdempotencyKey = Annotated[str, Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")]


class CreateCaseRequest(FrozenModel):
    display_name: Annotated[str, Field(min_length=1, max_length=120)]
    owner_alias: Annotated[str, Field(min_length=1, max_length=64)] = "analyst"


class VideoAccepted(FrozenModel):
    video_id: Uuid7
    job_id: Uuid7
    sha256: Sha256Hex
    deduplicated: bool


class SearchCreateRequest(FrozenModel):
    query: Annotated[str, Field(min_length=1, max_length=500)]
    max_results: Annotated[int, Field(ge=1, le=5)] = 5
    time_range: TimeRangeUs | None = None


class TrackCreateRequest(FrozenModel):
    subject_type: SubjectType
    seed_frame_id: FrameId
    seed_bbox_px: BBoxPx
    seed_detection_id: Uuid7 | None = None


class TrackAccepted(FrozenModel):
    track_id: Uuid7
    job_id: Uuid7


class ReconstructionCreateRequest(FrozenModel):
    track_id: Uuid7
    target_frame_id: FrameId
    target_bbox_px: BBoxPx
    policy_profile: Literal["demo-conservative-v1"] = "demo-conservative-v1"


class RunAccepted(FrozenModel):
    run_id: Uuid7
    job_id: Uuid7


class ReviewCreateRequest(FrozenModel):
    reviewer_alias: Annotated[str, Field(min_length=1, max_length=64)]
    decision: ReviewDecision
    reason_code: VetoReason | None = None
    comment: Annotated[str, Field(max_length=1000)] = ""
    reviewed_result_sha256: Sha256Hex
    reviewed_provenance_sha256: Sha256Hex


class ReportCreateRequest(FrozenModel):
    run_id: Uuid7
    review_id: Uuid7


class ReportAccepted(FrozenModel):
    report_id: Uuid7
    job_id: Uuid7


class ErrorBody(FrozenModel):
    code: ErrorCode
    message: Annotated[str, Field(min_length=1, max_length=500)]
    retryable: bool
    correlation_id: Uuid7
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(FrozenModel):
    error: ErrorBody


class FixtureInfo(FrozenModel):
    fixture_id: str
    display_name: str
    source_sha256: Sha256Hex
    duration_us: int
    license_note: str
    synthetic: bool


class HealthResponse(FrozenModel):
    status: Literal["ok", "degraded"]
    version: str
    mode: OperatingMode
    process: Literal["api", "worker"]
    database: Literal["ok", "unavailable"]
    adapters: tuple[AdapterHealth, ...]
    fixtures: tuple[FixtureInfo, ...] = ()
