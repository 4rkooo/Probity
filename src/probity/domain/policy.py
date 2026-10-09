"""Typed policy configuration loaded from ``config/policy.demo.yaml``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from probity.domain.ids import canonical_sha256

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[3] / "config" / "policy.demo.yaml"

Pos = Annotated[float, Field(gt=0)]
Unit = Annotated[float, Field(ge=0, le=1)]


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class VideoPolicy(_Section):
    max_duration_s: Annotated[int, Field(gt=0)]
    max_bytes: Annotated[int, Field(gt=0)]
    max_width: Annotated[int, Field(gt=0)]
    max_height: Annotated[int, Field(gt=0)]
    probe_timeout_s: Pos
    allowed_codecs: tuple[Literal["h264", "hevc"], ...]


class SegmentPolicy(_Section):
    duration_s: Annotated[int, Field(gt=0)]
    overlap_s: Annotated[int, Field(ge=0)]
    min_final_duration_s: Annotated[int, Field(gt=0)]
    thumbnail_width_px: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def _step(self) -> Self:
        if self.overlap_s >= self.duration_s:
            raise ValueError("overlap must be smaller than duration")
        return self

    @property
    def duration_us(self) -> int:
        return self.duration_s * 1_000_000

    @property
    def step_us(self) -> int:
        return (self.duration_s - self.overlap_s) * 1_000_000


class DetectPolicy(_Section):
    max_sample_fps: Pos


class TrackPolicy(_Section):
    window_radius_s: Pos
    min_detector_observations: Annotated[int, Field(gt=0)]
    min_mean_confidence: Unit
    max_bridge_gap_frames: Annotated[int, Field(ge=0)]
    min_iou_for_association: Unit
    max_center_step_diag: Unit
    box_scale_ratio_min: Pos
    box_scale_ratio_max: Pos
    max_aspect_change: Unit
    min_hsv_hist_correlation: Annotated[float, Field(ge=-1, le=1)]
    max_competing_box_iou: Unit


class CropPolicy(_Section):
    alignment_context_expand: Unit


class QualityPolicy(_Section):
    resize_long_edge_px: Annotated[int, Field(gt=0)]
    laplacian_variance_normalizer: Pos
    subject_size_normalizer_px: Pos
    black_luma_8bit: Annotated[int, Field(ge=0, le=255)]
    white_luma_8bit: Annotated[int, Field(ge=0, le=255)]
    max_clipped_fraction: Unit
    pose_aspect_ratio_limit: Pos
    analyst_seed_confidence: Unit


class TargetPolicy(_Section):
    min_width_px: Annotated[int, Field(gt=0)]
    min_height_px: Annotated[int, Field(gt=0)]
    min_quality: Unit


class DonorPolicy(_Section):
    min_quality: Unit
    min_quality_gain: Unit
    max_occluded_fraction: Unit
    max_exposure_delta_stops: Pos
    temporal_decay_s: Pos
    max_count: Annotated[int, Field(gt=0)]
    min_count: Annotated[int, Field(gt=0)]


class AkazePolicy(_Section):
    min_keypoints: Annotated[int, Field(gt=0)]
    ratio_test: Unit
    min_good_matches: Annotated[int, Field(gt=0)]


class RansacPolicy(_Section):
    reprojection_threshold_px: Pos


class AlignmentPolicy(_Section):
    min_inlier_ratio: Unit
    max_median_reprojection_px: Pos
    min_valid_coverage: Unit
    max_corner_outside_fraction: Unit


class EccPolicy(_Section):
    max_iterations: Annotated[int, Field(gt=0)]
    epsilon: Pos
    min_correlation: Unit


class ColorPolicy(_Section):
    gain_min: Pos
    gain_max: Pos
    bias_min_8bit: int
    bias_max_8bit: int
    max_context_gradient_8bit: Annotated[int, Field(ge=0, le=255)]
    min_context_samples: Annotated[int, Field(gt=0)]
    max_mean_abs_residual_8bit: Annotated[float, Field(ge=0, le=255)]


class FusionPolicy(_Section):
    tile_px: Literal[8]
    keep_original_sharpness: Unit
    min_sharpness_improvement: Unit
    sharpness_ratio_epsilon: Pos
    required_valid_tile_coverage: Unit


class ValidationPolicy(_Section):
    max_boundary_discontinuity_8bit: Annotated[float, Field(ge=0, le=255)]


class IterationPolicy(_Section):
    max_passes: Annotated[int, Field(ge=1, le=2)]
    convergence_changed_fraction: Unit


class IntegrityPolicy(_Section):
    min_score: Annotated[int, Field(ge=0, le=100)]
    required_provenance_coverage: Unit
    max_semantic_generated_fraction: Unit


class SearchWeights(_Section):
    cosine: Unit
    lexical_overlap: Unit
    detection_match: Unit
    visibility_match: Unit

    @model_validator(mode="after")
    def _sum(self) -> Self:
        total = self.cosine + self.lexical_overlap + self.detection_match + self.visibility_match
        if abs(total - 1.0) > 1e-9:
            raise ValueError("search weights must sum to 1")
        return self


class DisallowedTerms(_Section):
    identity: tuple[str, ...]
    legal: tuple[str, ...]
    intent: tuple[str, ...]


class SearchPolicy(_Section):
    top_k: Annotated[int, Field(gt=0)]
    max_results: Annotated[int, Field(gt=0, le=5)]
    overlap_collapse_time_iou: Unit
    weights: SearchWeights
    disallowed_terms: DisallowedTerms


class AdapterTimeouts(_Section):
    vast: Pos
    wandb: Pos
    cosmos_per_segment: Pos
    yolo_per_window: Pos


class AdapterPolicy(_Section):
    health_budget_s: Pos
    read_retry_delays_s: tuple[float, ...]
    timeouts_s: AdapterTimeouts


class IdempotencyPolicy(_Section):
    ttl_hours: Annotated[int, Field(gt=0)]


class JobsPolicy(_Section):
    lease_seconds: Annotated[int, Field(gt=0)]
    lease_renew_seconds: Annotated[int, Field(gt=0)]


class PolicyConfig(_Section):
    profile: Literal["demo-conservative-v1"]
    schema_version: Literal["1.0"]
    video: VideoPolicy
    segment: SegmentPolicy
    detect: DetectPolicy
    track: TrackPolicy
    crop: CropPolicy
    quality: QualityPolicy
    target: TargetPolicy
    donor: DonorPolicy
    akaze: AkazePolicy
    ransac: RansacPolicy
    alignment: AlignmentPolicy
    ecc: EccPolicy
    color: ColorPolicy
    fusion: FusionPolicy
    validation: ValidationPolicy
    iteration: IterationPolicy
    integrity: IntegrityPolicy
    search: SearchPolicy
    adapters: AdapterPolicy
    idempotency: IdempotencyPolicy
    jobs: JobsPolicy

    @property
    def config_sha256(self) -> str:
        """Canonical hash of the parsed configuration; bound into every reconstruction/approval."""
        return canonical_sha256(self.model_dump(mode="json"))


def load_policy(path: Path | str | None = None) -> PolicyConfig:
    source = Path(path) if path is not None else DEFAULT_POLICY_PATH
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return PolicyConfig.model_validate(raw)


@lru_cache(maxsize=1)
def default_policy() -> PolicyConfig:
    return load_policy()
