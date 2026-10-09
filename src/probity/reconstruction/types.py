"""Internal TrueFrame interfaces shared by every lane. FROZEN since "p2: parallel scaffolding".

Change requests go to ``notes/lane-<x>-requests.md``; only the integrator edits this file, and any
edit regenerates ``scripts/frozen_hashes.json``.

Conventions for every type below
--------------------------------
Images
    ``numpy.ndarray`` of dtype ``uint8``, shape ``(H, W, 3)``, channel order **BGR** (OpenCV), and
    C-contiguous. Luma is ``uint8 (H, W)`` from ``cv2.COLOR_BGR2GRAY``. Masks are ``bool (H, W)``.
Coordinates
    Pixel ``(x, y)`` with x to the right and y down; pixel centres sit on integer coordinates.
Boxes
    ``BBox = (x1, y1, x2, y2)`` of ints, half-open: a pixel is inside iff ``x1 <= x < x2`` and
    ``y1 <= y < y2``. Boxes are in FULL-FRAME coordinates unless documented otherwise.
Target-crop frame
    Arrays named ``*_crop`` (and the masks and source maps that travel with them) are indexed in
    the target's expanded-box frame: crop pixel ``[r, c]`` is full-frame ``(ex1 + c, ey1 + r)``
    where ``(ex1, ey1, ex2, ey2) = target.expanded_box``. Their shape is
    ``(ey2 - ey1, ex2 - ex1[, 3])``.
Matrices
    ``float64 (3, 3)``, row-major, mapping DONOR full-frame homogeneous ``(x, y, 1)`` to TARGET
    full-frame coordinates. An ECC affine is stored as 3x3 with last row ``(0, 0, 1)``.
Source maps
    ``source_x`` / ``source_y`` are ``float32`` donor full-frame coordinates for each target-crop
    pixel, i.e. the inverse transform. Warped pixels are exactly
    ``cv2.remap(donor_frame, source_x, source_y, cv2.INTER_LANCZOS4,
    borderMode=cv2.BORDER_REFLECT_101)`` before color normalization.
Arrays are not copied. A producer must not mutate an array after building the dataclass and a
consumer must never mutate one it received.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from probity.domain.enums import (
    AlignmentMethod,
    InferenceMode,
    ObservationSource,
    PolicyStage,
    ReasonCode,
)
from probity.domain.ids import frame_id as make_frame_id
from probity.domain.models import (
    Detection,
    FrameReference,
    PolicyDecision,
    ScalarValue,
    SourceLutEntry,
    Track,
    TrackObservation,
)

BBox = tuple[int, int, int, int]
FrameLoader = Callable[[FrameReference], np.ndarray]
Operator = Literal["<", "<=", ">", ">=", "==", "!=", "in", "none"]


# ------------------------------------------------------------------------------------------------
# Gates and decisions
# ------------------------------------------------------------------------------------------------


def passes(observed: ScalarValue, operator: Operator, threshold: ScalarValue) -> bool:
    if operator == "==":
        return observed == threshold
    if operator == "!=":
        return observed != threshold
    if not isinstance(observed, int | float) or not isinstance(threshold, int | float):
        raise ValueError(f"operator {operator!r} needs numeric operands")
    if operator == "<":
        return observed < threshold
    if operator == "<=":
        return observed <= threshold
    if operator == ">":
        return observed > threshold
    if operator == ">=":
        return observed >= threshold
    raise ValueError(f"operator {operator!r} is not a comparison")


@dataclass(frozen=True, slots=True)
class Gate:
    """One material comparison, not yet logged. ``reject_code`` defaults to ``rule_code``.

    Lanes return gates; the integrator records them with ``DecisionLog.check`` so every material
    accept/reject becomes a ``PolicyDecision`` with rule code, observed, operator, threshold, units.
    """

    rule_code: ReasonCode
    stage: PolicyStage
    observed: ScalarValue
    operator: Operator
    threshold: ScalarValue
    units: str | None
    policy_key: str | None
    accept_reason: str
    reject_reason: str
    reject_code: ReasonCode | None = None

    @property
    def ok(self) -> bool:
        return passes(self.observed, self.operator, self.threshold)

    @property
    def failure_code(self) -> ReasonCode:
        return self.reject_code or self.rule_code


@dataclass(frozen=True, slots=True)
class GateResult:
    """Outcome of recording a gate sequence: ``decisions`` are the rows written, in order."""

    accepted: bool
    decisions: tuple[PolicyDecision, ...]


# ------------------------------------------------------------------------------------------------
# Observations and quality (section 9)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QualityScores:
    """Section 9 components, each clamped to [0, 1]; ``Q`` is the weighted observation quality."""

    D: float
    S: float
    Z: float
    E: float
    O: float  # noqa: E741 - section 9 symbol
    P: float
    Q: float


@dataclass(frozen=True, slots=True, eq=False)
class Obs:
    """One track observation with its pixels, ready for alignment and fusion.

    ``bbox_px`` and ``expanded_box`` are full-frame and half-open; ``expanded_box`` is
    ``bbox_px`` grown by ``crop.alignment_context_expand`` (half per side) and clipped to the
    frame. ``crop`` is ``frame[y1:y2, x1:x2]`` of ``bbox_px``; ``expanded_crop`` is the same for
    ``expanded_box``. Both are BGR uint8 copies. ``frame_size`` is the full frame ``(W, H)``.
    ``detector_backed`` is True only for ``ObservationSource.DETECTOR``; ``bridged`` is True for
    ``CSRT_BRIDGE``. ``occluded_fraction`` is the preflight O_i input in [0, 1].
    """

    frame_number: int
    pts_us: int
    video_id: str
    track_id: str
    bbox_px: BBox
    detector_backed: bool
    bridged: bool
    confidence: float
    occluded_fraction: float
    crop: np.ndarray
    expanded_crop: np.ndarray
    expanded_box: BBox
    frame_size: tuple[int, int]

    @property
    def frame_id(self) -> str:
        return make_frame_id(self.video_id, self.frame_number)


# ------------------------------------------------------------------------------------------------
# Alignment and color (lane B)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class Alignment:
    """Donor-to-target geometry and its section 9 confidence ``A``.

    ``method`` is ``AKAZE_HOMOGRAPHY`` or ``ECC_AFFINE`` (the method of the final attempt).
    ``matrix`` is ``None`` when no estimate exists. Metrics that were never measured are 0.0.
    ``valid_coverage`` is the fraction of the target expanded box whose source coordinates fall
    inside the donor frame; ``corner_outside_fraction`` is the fraction of the four projected
    target-expanded-box corners outside the donor frame. ``reason`` is
    ``ALIGNMENT_AKAZE_ACCEPTED``, ``ALIGNMENT_ECC_FALLBACK_ACCEPTED`` or ``ALIGNMENT_FAILED``.
    ``gates`` is the ordered audit trail of every attempt (an AKAZE failure followed by the ECC
    gates when the fallback ran); ``accepted`` is authoritative.
    """

    method: AlignmentMethod
    matrix: np.ndarray | None
    inlier_ratio: float
    median_reproj_px: float
    valid_coverage: float
    corner_outside_fraction: float
    A: float
    accepted: bool
    reason: ReasonCode
    gates: tuple[Gate, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class Warp:
    """A donor resampled into the target-crop frame (before color normalization).

    ``crop`` is BGR uint8, ``valid_mask`` is True where ``source_x/source_y`` lie inside the donor
    frame ``[0, W-1] x [0, H-1]``; ``source_x/source_y`` are float32 donor full-frame coordinates.
    """

    crop: np.ndarray
    valid_mask: np.ndarray
    source_x: np.ndarray
    source_y: np.ndarray


@dataclass(frozen=True, slots=True)
class ColorFit:
    """Per-channel ``target = gain * donor + bias`` in 8-bit units, channel order B, G, R.

    Fit on stable context-ring pixels (expanded box minus subject box, Sobel <= limit, valid warp).
    ``mean_abs_residual`` is in 8-bit units after applying the fit; ``samples`` is the pixel count.
    ``reason`` is ``PHOTOMETRIC_INCOMPATIBLE`` or ``LIGHTING_OUT_OF_RANGE`` on rejection and
    ``PHOTOMETRIC_INCOMPATIBLE`` (as an ACCEPT rule code) on acceptance.
    """

    gain: tuple[float, float, float]
    bias: tuple[float, float, float]
    mean_abs_residual: float
    samples: int
    accepted: bool
    reason: ReasonCode
    gates: tuple[Gate, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class AlignedDonor:
    """A donor that passed preflight, alignment, and color, ranked for fusion.

    ``warped_crop`` is the color-normalized warp in the target-crop frame (BGR uint8);
    ``valid_mask``, ``source_x`` and ``source_y`` come unchanged from its ``Warp``.
    ``T = exp(-|dt_s| / donor.temporal_decay_s)`` and ``R = 0.45 Q + 0.35 A + 0.20 T``.
    """

    obs: Obs
    quality: QualityScores
    alignment: Alignment
    color: ColorFit
    warped_crop: np.ndarray
    valid_mask: np.ndarray
    source_x: np.ndarray
    source_y: np.ndarray
    T: float
    R: float


# ------------------------------------------------------------------------------------------------
# Fusion (lane C)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TileDecision:
    """One 8x8 subject tile (edge tiles clipped). ``box`` is full-frame and half-open.

    ``tile_index`` is row-major over ``quality.tiles(subject_box, 8)``. ``source_index`` is 0 for
    the original target or k >= 1 for ``FusionResult.donor_lut_order[k - 1]``. ``score`` is the
    selection score (0.0 when kept). ``reason_code`` is ``BORROW_TILE_ACCEPTED``,
    ``KEEP_ORIGINAL_CLEAR`` (already sharp), ``NO_TILE_IMPROVED`` (no donor gain),
    ``PHOTOMETRIC_INCOMPATIBLE`` (gain but residual/coverage failed), or
    ``REVERT_TILE_VALIDATION`` (reverted by the validation pass).
    """

    tile_index: int
    box: BBox
    source_index: int
    score: float
    reason_code: ReasonCode
    improvement: float = 0.0
    residual_8bit: float = 0.0


@dataclass(frozen=True, slots=True, eq=False)
class FusionResult:
    """``result_frame`` is a full-frame BGR uint8 copy of the target with borrowed tiles pasted.

    ``donor_lut_order`` lists donor frame numbers in LUT order (LUT index k = position + 1),
    which is rank order: ``R`` descending, then frame number ascending.
    """

    result_frame: np.ndarray
    tile_decisions: tuple[TileDecision, ...]
    donor_lut_order: tuple[int, ...]


# ------------------------------------------------------------------------------------------------
# Provenance (lane D)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProvenanceArrays:
    """Full-frame npz-pixel-v1 arrays, each ``(H, W)``: ``cls`` uint8 (ProvenanceClass),
    ``source_index`` uint16 (LUT row), ``source_x`` / ``source_y`` float32 source coordinates.
    ``lut`` is the matching source LUT (row 0 is the target); it is not stored in the npz.
    """

    cls: np.ndarray
    source_index: np.ndarray
    source_x: np.ndarray
    source_y: np.ndarray
    lut: tuple[SourceLutEntry, ...] = ()

    @classmethod
    def identity(cls, width: int, height: int) -> ProvenanceArrays:
        """Every pixel ORIGINAL from LUT row 0 at its own coordinate."""
        xs, ys = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
        return cls(
            cls=np.zeros((height, width), dtype=np.uint8),
            source_index=np.zeros((height, width), dtype=np.uint16),
            source_x=xs,
            source_y=ys,
        )

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "class": self.cls,
            "source_index": self.source_index,
            "source_x": self.source_x,
            "source_y": self.source_y,
        }


# ------------------------------------------------------------------------------------------------
# Tracking (step 2, reconstruction/tracking.py)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    """A detector box plus the ByteTrack ID the tracker assigned to it (None if unassociated)."""

    detection: Detection
    tracker_id: int | None


class Bridger(Protocol):
    def start(self, frame_number: int, image: np.ndarray, box: BBox) -> None: ...
    def step(self, frame_number: int, image: np.ndarray) -> BBox | None: ...


BridgerFactory = Callable[[], Bridger]


@dataclass(frozen=True, slots=True)
class ObservationAudit:
    frame_number: int
    source: ObservationSource
    accepted: bool
    reason_code: ReasonCode | None
    gate: str
    observed: float | str | None
    operator: Operator
    threshold: float | str | None
    detail: str
    detection_id: str | None = None


@dataclass(frozen=True, slots=True)
class TrackingOutcome:
    track: Track
    audit: tuple[ObservationAudit, ...]
    sampled_frame_numbers: tuple[int, ...]
    tracker_id: int | None


@dataclass(frozen=True, slots=True)
class TrackMeta:
    tracker_version: str
    detector_model_id: str | None
    mode: InferenceMode
    created_at: str


@dataclass(frozen=True, slots=True)
class IdentityMetrics:
    association_iou: float
    center_step: float
    scale: float
    aspect: float
    appearance: float
    competing_iou: float


@dataclass(frozen=True, slots=True)
class GateFailure:
    gate: str
    observed: float
    operator: Operator
    threshold: float
    detail: str


# ------------------------------------------------------------------------------------------------
# Preflight (step 3, reconstruction/preflight.py)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SubjectMeasure:
    """``image`` is the full BGR frame; ``box`` the full-frame subject box."""

    obs: TrackObservation
    frame: FrameReference
    image: np.ndarray
    box: BBox
    occluded: float
    quality: QualityScores

    @property
    def frame_number(self) -> int:
        return self.frame.frame_number


@dataclass(frozen=True, slots=True)
class DonorMeasure(SubjectMeasure):
    dt_s: float
    scale: float
    aspect_change: float
    exposure_stops: float
    decision_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreflightResult:
    target: SubjectMeasure | None
    donors: tuple[DonorMeasure, ...]
    refusal: ReasonCode | None
    median_aspect: float
