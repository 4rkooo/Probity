"""Lane B: bounded per-channel color normalization (section 9, algorithm step 7). SIGNATURES FROZEN.

Fit ``target = gain * donor + bias`` per BGR channel on aligned, valid context-ring pixels outside
the subject box (ring from the expanded alignment crop), using only pixels with Sobel magnitude
``<= color.max_context_gradient_8bit / 255``. Robust fit, deterministic. No histogram matching, no
learned colorization. Gates: sample count, gain bounds, bias bounds, mean absolute residual.
"""

from __future__ import annotations

import numpy as np

from probity.domain.policy import PolicyConfig
from probity.reconstruction.types import ColorFit, Obs, Warp

LANE = "lane B"


def context_ring_mask(target: Obs) -> np.ndarray:
    """bool, target-crop frame: True inside ``expanded_box`` and outside ``bbox_px``."""
    raise NotImplementedError(LANE)


def stable_context_mask(target: Obs, warp: Warp, cfg: PolicyConfig) -> np.ndarray:
    """Ring pixels with low target gradient and a valid warp sample (bool, target-crop frame)."""
    raise NotImplementedError(LANE)


def fit_color(target: Obs, warp: Warp, cfg: PolicyConfig) -> ColorFit:
    """Fit and gate the color transform; the residual is measured after applying the fit."""
    raise NotImplementedError(LANE)


def apply_color(crop: np.ndarray, fit: ColorFit) -> np.ndarray:
    """uint8 BGR: ``clip(rint(gain * crop + bias), 0, 255)`` per channel (float64 internally)."""
    raise NotImplementedError(LANE)
