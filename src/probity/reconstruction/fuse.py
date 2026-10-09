"""Lane C: donor ranking and 8x8 winner-take-all fusion (section 9, steps 8-9). SIGNATURES FROZEN.

Keep the original tile when its normalized sharpness >= ``fusion.keep_original_sharpness`` or no
donor improves it by ``fusion.min_sharpness_improvement``. Otherwise consider only donors with full
valid coverage for the tile and residual <= ``color.max_mean_abs_residual_8bit``, and pick exactly
one by ``0.50 * improvement + 0.30 * R + 0.20 * (1 - residual / 18)``. Never average donors; never
read a previous result. Ties break by LUT order (rank desc, frame asc).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from probity.domain.policy import PolicyConfig
from probity.reconstruction.types import AlignedDonor, FusionResult, Gate, Obs, QualityScores

LANE = "lane C"
R_WEIGHTS = {"Q": 0.45, "A": 0.35, "T": 0.20}
SELECT_WEIGHTS = {"improvement": 0.50, "rank": 0.30, "residual": 0.20}


def temporal_preference(dt_s: float, cfg: PolicyConfig) -> float:
    """T = exp(-|dt_s| / donor.temporal_decay_s)."""
    raise NotImplementedError(LANE)


def rank_score(quality: QualityScores, A: float, T: float) -> float:
    """R = 0.45 Q + 0.35 A + 0.20 T."""
    raise NotImplementedError(LANE)


def rank_donors(donors: Sequence[AlignedDonor], cfg: PolicyConfig) -> tuple[AlignedDonor, ...]:
    """Sort by (R desc, frame_number asc), keep at most ``donor.max_count``; this is LUT order."""
    raise NotImplementedError(LANE)


def relative_improvement(s_donor: float, s_target: float, cfg: PolicyConfig) -> float:
    """(S_donor - S_target) / max(S_target, fusion.sharpness_ratio_epsilon)."""
    raise NotImplementedError(LANE)


def selection_score(improvement: float, R: float, residual_8bit: float,
                    cfg: PolicyConfig) -> float:
    raise NotImplementedError(LANE)


def fuse_tiles(target_frame: np.ndarray, target: Obs, donors: Sequence[AlignedDonor],
               cfg: PolicyConfig) -> FusionResult:
    """``target_frame`` is the full BGR target; ``donors`` are already in LUT order.

    Tiles are ``quality.tiles(target.bbox_px, 8)``; sharpness is
    ``quality.tile_sharpness_scores`` on the subject luma.
    """
    raise NotImplementedError(LANE)


def tile_gates(result: FusionResult, cfg: PolicyConfig) -> tuple[tuple[str, Gate], ...]:
    """``(subject_ref, gate)`` rows to log: one ``BORROW_TILE_ACCEPTED`` per borrowed tile and one
    aggregate ``KEEP_ORIGINAL_CLEAR`` row with counts per keep reason."""
    raise NotImplementedError(LANE)
