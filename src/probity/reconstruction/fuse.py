"""Lane C: donor ranking and 8x8 winner-take-all fusion (section 9, steps 8-9). SIGNATURES FROZEN.

Keep the original tile when its normalized sharpness >= ``fusion.keep_original_sharpness`` or no
donor improves it by ``fusion.min_sharpness_improvement``. Otherwise consider only donors with full
valid coverage for the tile and residual <= ``color.max_mean_abs_residual_8bit``, and pick exactly
one by ``0.50 * improvement + 0.30 * R + 0.20 * (1 - residual / 18)``. Never average donors; never
read a previous result. Ties break by LUT order (rank desc, frame asc).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from probity.domain.enums import PolicyStage, ReasonCode
from probity.domain.errors import ValidationFailed
from probity.domain.policy import PolicyConfig
from probity.reconstruction.types import AlignedDonor, FusionResult, Gate, Obs, QualityScores

LANE = "lane C"
R_WEIGHTS = {"Q": 0.45, "A": 0.35, "T": 0.20}
SELECT_WEIGHTS = {"improvement": 0.50, "rank": 0.30, "residual": 0.20}
R_TOLERANCE = 1e-9


def _unit(name: str, value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite score in [0, 1], got {value!r}")
    return value


# ------------------------------------------------------------------------------------------------
# Ranking (step 8)
# ------------------------------------------------------------------------------------------------


def temporal_preference(dt_s: float, cfg: PolicyConfig) -> float:
    """T = exp(-|dt_s| / donor.temporal_decay_s)."""
    if not math.isfinite(dt_s):
        raise ValueError(f"dt_s must be finite, got {dt_s!r}")
    return math.exp(-abs(dt_s) / cfg.donor.temporal_decay_s)


def rank_score(quality: QualityScores, A: float, T: float) -> float:
    """R = 0.45 Q + 0.35 A + 0.20 T."""
    w = R_WEIGHTS
    return w["Q"] * _unit("Q", quality.Q) + w["A"] * _unit("A", A) + w["T"] * _unit("T", T)


def _same_subject(donors: Sequence[AlignedDonor], target: Obs | None = None) -> None:
    """Donors never cross videos or tracks, never repeat a frame, and never are bridged-only."""
    ref = target if target is not None else (donors[0].obs if donors else None)
    seen: set[int] = set()
    for d in donors:
        o = d.obs
        if ref is not None and (o.video_id, o.track_id) != (ref.video_id, ref.track_id):
            raise ValidationFailed(
                f"donor {o.frame_id} is from video/track {o.video_id}/{o.track_id}, "
                f"expected {ref.video_id}/{ref.track_id}"
            )
        if target is not None and o.frame_number == target.frame_number:
            raise ValidationFailed(f"donor {o.frame_id} is the target frame")
        if o.frame_number in seen:
            raise ValidationFailed(f"donor frame {o.frame_number} appears twice")
        seen.add(o.frame_number)
        if not o.detector_backed or o.bridged:
            raise ValidationFailed(f"donor {o.frame_id} is not a detector-backed observation")
        if not d.alignment.accepted or not d.color.accepted:
            raise ValidationFailed(f"donor {o.frame_id} did not pass alignment and color gates")
        expected = rank_score(d.quality, d.alignment.A, d.T)
        if not math.isclose(d.R, expected, rel_tol=R_TOLERANCE, abs_tol=R_TOLERANCE):
            raise ValidationFailed(f"donor {o.frame_id} has R={d.R!r}, formula gives {expected!r}")


def _lut_key(d: AlignedDonor) -> tuple[float, int]:
    return (-d.R, d.obs.frame_number)


def rank_donors(donors: Sequence[AlignedDonor], cfg: PolicyConfig) -> tuple[AlignedDonor, ...]:
    """Sort by (R desc, frame_number asc), keep at most ``donor.max_count``; this is LUT order."""
    _same_subject(donors)
    return tuple(sorted(donors, key=_lut_key)[: cfg.donor.max_count])


def donor_count_gate(ranked: Sequence[AlignedDonor], cfg: PolicyConfig) -> Gate:
    """``INSUFFICIENT_COMPATIBLE_DONORS``: at least ``donor.min_count`` ranked donors, even if one
    donor would supply every chosen tile. Log it at ``PolicyStage.RANK`` on subject ``donors``."""
    return Gate(
        ReasonCode.INSUFFICIENT_COMPATIBLE_DONORS,
        PolicyStage.RANK,
        len(ranked),
        ">=",
        cfg.donor.min_count,
        "donors",
        "donor.min_count",
        "Enough compatible donors",
        "Fewer compatible donors than required",
    )


# ------------------------------------------------------------------------------------------------
# Tile selection (step 9)
# ------------------------------------------------------------------------------------------------


def relative_improvement(s_donor: float, s_target: float, cfg: PolicyConfig) -> float:
    """(S_donor - S_target) / max(S_target, fusion.sharpness_ratio_epsilon)."""
    raise NotImplementedError(LANE)


def selection_score(improvement: float, R: float, residual_8bit: float, cfg: PolicyConfig) -> float:
    raise NotImplementedError(LANE)


def fuse_tiles(
    target_frame: np.ndarray, target: Obs, donors: Sequence[AlignedDonor], cfg: PolicyConfig
) -> FusionResult:
    """``target_frame`` is the full BGR target; ``donors`` are already in LUT order.

    Tiles are ``quality.tiles(target.bbox_px, 8)``; sharpness is
    ``quality.tile_sharpness_scores`` on the subject luma.
    """
    raise NotImplementedError(LANE)


def tile_gates(result: FusionResult, cfg: PolicyConfig) -> tuple[tuple[str, Gate], ...]:
    """``(subject_ref, gate)`` rows to log: one ``BORROW_TILE_ACCEPTED`` per borrowed tile and one
    aggregate ``KEEP_ORIGINAL_CLEAR`` row with counts per keep reason."""
    raise NotImplementedError(LANE)
