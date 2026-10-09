"""Lane E: evaluation metrics (section 12 table). SIGNATURES FROZEN.

Ground-truth metrics (PSNR/SSIM/OCR accuracy) are valid only where the synthetic generator's clean
target exists. OCR character accuracy compares strings supplied by an eval harness; no OCR runs in
reconstruction and no plate characters are emitted by it. Provenance metrics are reported in a
separate group and methods are never ranked by sharpness alone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from probity.domain.enums import PolicyOutcome, PolicyStage, ProvenanceClass, SourceRole
from probity.domain.ids import is_uuid7
from probity.domain.models import PolicyDecision
from probity.ports import EvaluationMetric, EvaluationRow
from probity.reconstruction.types import BBox, ProvenanceArrays

LANE = "lane E"

_SRGB_KNEE = 0.04045
_REC709 = (0.2126, 0.7152, 0.0722)
_SSIM_WIN = 11
_SSIM_SIGMA = 1.5
_SSIM_K1 = 0.01
_SSIM_K2 = 0.03
_SSIM_L = 1.0
_CLASS_KEYS = (
    ProvenanceClass.ORIGINAL.name,
    ProvenanceClass.BORROWED.name,
    ProvenanceClass.GENERATED_BLEND.name,
)


def _check_bgr(image: object, what: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{what} must be a numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{what} must be BGR uint8 (H, W, 3), got {image.dtype} {image.shape}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"{what} is empty")
    return image


def linear_luma(bgr: np.ndarray) -> np.ndarray:
    """float64 (H, W) linearized luminance in [0, 1] (sRGB decode, Rec. 709 weights)."""
    rgb = _check_bgr(bgr, "image").astype(np.float64) / 255.0
    rgb = rgb[:, :, ::-1]
    linear = np.where(rgb <= _SRGB_KNEE, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = _REC709[0] * linear[:, :, 0] + _REC709[1] * linear[:, :, 1] + _REC709[2] * linear[:, :, 2]
    return np.clip(y, 0.0, 1.0)


def _pair_luma(result_bgr: np.ndarray, truth_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = _check_bgr(result_bgr, "result")
    truth = _check_bgr(truth_bgr, "truth")
    if result.shape != truth.shape:
        raise ValueError(f"result shape {result.shape} != truth shape {truth.shape}")
    return linear_luma(result), linear_luma(truth)


def psnr(result_bgr: np.ndarray, truth_bgr: np.ndarray, mask: np.ndarray | None = None) -> float:
    """dB on linearized luminance; ``mask`` (bool) restricts to registered ground truth."""
    result_y, truth_y = _pair_luma(result_bgr, truth_bgr)
    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.shape != result_y.shape or mask.dtype != bool:
            raise ValueError("mask must be bool (H, W) matching the images")
        if not mask.any():
            raise ValueError("mask selects no registered ground-truth pixels")
        result_y = result_y[mask]
        truth_y = truth_y[mask]
    mse = float(np.mean((result_y - truth_y) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def _gauss11(plane: np.ndarray) -> np.ndarray:
    kernel = cv2.getGaussianKernel(_SSIM_WIN, _SSIM_SIGMA)
    return cv2.sepFilter2D(plane, cv2.CV_64F, kernel, kernel, borderType=cv2.BORDER_REFLECT_101)


def ssim(result_bgr: np.ndarray, truth_bgr: np.ndarray) -> float:
    """Mean SSIM on linearized luminance (Gaussian window 11, sigma 1.5, K1 0.01, K2 0.03)."""
    x, y = _pair_luma(result_bgr, truth_bgr)
    if x.shape[0] < _SSIM_WIN or x.shape[1] < _SSIM_WIN:
        raise ValueError(f"SSIM needs images at least {_SSIM_WIN}x{_SSIM_WIN}")
    c1 = (_SSIM_K1 * _SSIM_L) ** 2
    c2 = (_SSIM_K2 * _SSIM_L) ** 2
    mu_x = _gauss11(x)
    mu_y = _gauss11(y)
    sigma_x2 = np.maximum(_gauss11(x * x) - mu_x * mu_x, 0.0)
    sigma_y2 = np.maximum(_gauss11(y * y) - mu_y * mu_y, 0.0)
    sigma_xy = _gauss11(x * y) - mu_x * mu_y
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x2 + sigma_y2 + c2)
    return float(np.mean(num / den))


def _class_plane(arrays: ProvenanceArrays) -> np.ndarray:
    plane = np.asarray(arrays.cls)
    if plane.ndim != 2 or plane.size == 0:
        raise ValueError("provenance class map must be a non-empty (H, W) array")
    return plane


def _box_slice(box: BBox, height: int, width: int) -> tuple[slice, slice]:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"subject_box {box} is empty")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(f"subject_box {box} is outside the {width}x{height} frame")
    return slice(y1, y2), slice(x1, x2)


def provenance_percentages(arrays: ProvenanceArrays, subject_box: BBox | None = None
                           ) -> dict[str, float]:
    """ORIGINAL / BORROWED / GENERATED_BLEND percent of the frame (or of ``subject_box``)."""
    plane = _class_plane(arrays)
    if subject_box is not None:
        rows, cols = _box_slice(subject_box, plane.shape[0], plane.shape[1])
        plane = plane[rows, cols]
    total = int(plane.size)
    return {
        name: round(100.0 * int((plane == getattr(ProvenanceClass, name)).sum()) / total, 4)
        for name in _CLASS_KEYS
    }


def _lut_donor_rows(arrays: ProvenanceArrays) -> set[int]:
    return {int(entry.index) for entry in arrays.lut
            if entry.role is SourceRole.DONOR and int(entry.index) >= 1}


def supported_changed_pixel_rate(result: np.ndarray, target: np.ndarray,
                                 arrays: ProvenanceArrays) -> float:
    """Changed pixels with a BORROWED label resolving to a donor LUT row / changed (1.0 if none)."""
    result_img = _check_bgr(result, "result")
    target_img = _check_bgr(target, "target")
    if result_img.shape != target_img.shape:
        raise ValueError(f"result shape {result_img.shape} != target shape {target_img.shape}")
    plane = _class_plane(arrays)
    if plane.shape != result_img.shape[:2]:
        raise ValueError("provenance class map does not match the image shape")
    index = np.asarray(arrays.source_index)
    if index.shape != plane.shape:
        raise ValueError("provenance source_index does not match the class map")
    changed = np.any(result_img != target_img, axis=2)
    n_changed = int(changed.sum())
    if n_changed == 0:
        return 1.0
    donors = _lut_donor_rows(arrays)
    borrowed = plane == int(ProvenanceClass.BORROWED)
    supported = changed & borrowed & np.isin(index, list(donors) if donors else [-1])
    return int(supported.sum()) / n_changed


def ocr_character_accuracy(predicted: str, truth: str) -> float:
    """Exact per-position character accuracy over ``len(truth)``; evaluation data only."""
    if not isinstance(predicted, str) or not isinstance(truth, str):
        raise TypeError("OCR accuracy compares strings supplied by the eval harness")
    if truth == "":
        raise ValueError("OCR character accuracy is undefined for empty truth")
    hits = sum(1 for i, ch in enumerate(truth) if i < len(predicted) and predicted[i] == ch)
    return hits / len(truth)


def alignment_rejection_rate(decisions: Sequence[PolicyDecision]) -> dict[str, float]:
    """Rejected / considered donors overall (``"total"``) and per rejecting rule code."""
    first_reject: dict[str, str | None] = {}
    for row in decisions:
        if row.stage is not PolicyStage.ALIGN:
            continue
        first_reject.setdefault(row.subject_ref, None)
        if row.outcome is PolicyOutcome.REJECT and first_reject[row.subject_ref] is None:
            first_reject[row.subject_ref] = str(row.rule_code)
    n = len(first_reject)
    if n == 0:
        return {"total": 0.0}
    codes = [code for code in first_reject.values() if code is not None]
    out: dict[str, float] = {"total": len(codes) / n}
    counts: dict[str, int] = {}
    for code in codes:
        counts[code] = counts.get(code, 0) + 1
    for code, count in sorted(counts.items()):
        out[code] = count / n
    return out


def evaluation_row(name: str, correlation_id: str, metrics: Mapping[str, float]
                   ) -> EvaluationRow:
    """Deterministic row: metrics sorted by name, non-finite values rejected."""
    if not isinstance(correlation_id, str) or not is_uuid7(correlation_id):
        raise ValueError(f"correlation_id {correlation_id!r} is not a UUIDv7")
    ordered: list[EvaluationMetric] = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"metric {key!r} must be a finite float")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"metric {key!r} is not finite")
        ordered.append(EvaluationMetric(name=key, value=number))
    return EvaluationRow(name=name, correlation_id=correlation_id, metrics=tuple(ordered))
