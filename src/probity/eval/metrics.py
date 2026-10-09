"""Lane E: evaluation metrics (section 12 table). SIGNATURES FROZEN.

Ground-truth metrics (PSNR/SSIM/OCR accuracy) are valid only where the synthetic generator's clean
target exists. OCR character accuracy compares strings supplied by an eval harness; no OCR runs in
reconstruction and no plate characters are emitted by it. Provenance metrics are reported in a
separate group and methods are never ranked by sharpness alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from probity.domain.models import PolicyDecision
from probity.ports import EvaluationRow
from probity.reconstruction.types import BBox, ProvenanceArrays

LANE = "lane E"


def linear_luma(bgr: np.ndarray) -> np.ndarray:
    """float64 (H, W) linearized luminance in [0, 1] (sRGB decode, Rec. 709 weights)."""
    raise NotImplementedError(LANE)


def psnr(result_bgr: np.ndarray, truth_bgr: np.ndarray, mask: np.ndarray | None = None) -> float:
    """dB on linearized luminance; ``mask`` (bool) restricts to registered ground truth."""
    raise NotImplementedError(LANE)


def ssim(result_bgr: np.ndarray, truth_bgr: np.ndarray) -> float:
    """Mean SSIM on linearized luminance (Gaussian window 11, sigma 1.5, K1 0.01, K2 0.03)."""
    raise NotImplementedError(LANE)


def provenance_percentages(arrays: ProvenanceArrays, subject_box: BBox | None = None
                           ) -> dict[str, float]:
    """ORIGINAL / BORROWED / GENERATED_BLEND percent of the frame (or of ``subject_box``)."""
    raise NotImplementedError(LANE)


def supported_changed_pixel_rate(result: np.ndarray, target: np.ndarray,
                                 arrays: ProvenanceArrays) -> float:
    """Changed pixels with a BORROWED label resolving to a donor LUT row / changed (1.0 if none)."""
    raise NotImplementedError(LANE)


def ocr_character_accuracy(predicted: str, truth: str) -> float:
    """Exact per-position character accuracy over ``len(truth)``; evaluation data only."""
    raise NotImplementedError(LANE)


def alignment_rejection_rate(decisions: Sequence[PolicyDecision]) -> dict[str, float]:
    """Rejected / considered donors overall (``"total"``) and per rejecting rule code."""
    raise NotImplementedError(LANE)


def evaluation_row(name: str, correlation_id: str, metrics: Mapping[str, float]
                   ) -> EvaluationRow:
    """Deterministic row: metrics sorted by name, non-finite values rejected."""
    raise NotImplementedError(LANE)
