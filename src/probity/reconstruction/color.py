"""Lane B: bounded per-channel color normalization (section 9, algorithm step 7). SIGNATURES FROZEN.

Fit ``target = gain * donor + bias`` per BGR channel on aligned, valid context-ring pixels outside
the subject box (ring from the expanded alignment crop), using only pixels with Sobel magnitude
``<= color.max_context_gradient_8bit / 255``. Robust fit, deterministic. No histogram matching, no
learned colorization. Gates: sample count, gain bounds, bias bounds, mean absolute residual.
"""

from __future__ import annotations

import cv2
import numpy as np

from probity.domain.enums import PolicyStage, ReasonCode
from probity.domain.policy import PolicyConfig
from probity.reconstruction.types import ColorFit, Gate, Obs, Operator, Warp

LANE = "lane B"
COLOR = PolicyStage.COLOR
PHOTO = ReasonCode.PHOTOMETRIC_INCOMPATIBLE
LIGHT = ReasonCode.LIGHTING_OUT_OF_RANGE
IRLS_ITERS = 3
HUBER_K = 1.345 * 1.4826  # ~2.0 MAD, Huber with Gaussian consistency constant
IDENTITY_GAIN = (1.0, 1.0, 1.0)
IDENTITY_BIAS = (0.0, 0.0, 0.0)


def context_ring_mask(target: Obs) -> np.ndarray:
    """bool, target-crop frame: True inside ``expanded_box`` and outside ``bbox_px``."""
    ex1, ey1, ex2, ey2 = target.expanded_box
    height, width = ey2 - ey1, ex2 - ex1
    ring = np.ones((height, width), dtype=bool)
    x1, y1, x2, y2 = target.bbox_px
    ring[y1 - ey1:y2 - ey1, x1 - ex1:x2 - ex1] = False
    return ring


def _sobel_magnitude_8bit(bgr: np.ndarray) -> np.ndarray:
    """Sobel magnitude of luma in 8-bit units (equivalent to Sobel(luma/255)*255)."""
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gx = cv2.Sobel(luma, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def stable_context_mask(target: Obs, warp: Warp, cfg: PolicyConfig) -> np.ndarray:
    """Ring pixels with low target gradient and a valid warp sample (bool, target-crop frame)."""
    ring = context_ring_mask(target)
    if warp.valid_mask.shape != ring.shape:
        raise ValueError("warp.valid_mask must match the target-crop frame")
    if target.expanded_crop.shape[:2] != ring.shape:
        raise ValueError("target.expanded_crop must match the target-crop frame")
    mag = _sobel_magnitude_8bit(target.expanded_crop)
    return ring & warp.valid_mask & (mag <= float(cfg.color.max_context_gradient_8bit))


def _gate(
    accept: ReasonCode,
    reject: ReasonCode,
    observed: float | int,
    operator: Operator,
    threshold: float | int,
    units: str | None,
    key: str,
    accept_reason: str,
    reject_reason: str,
) -> Gate:
    return Gate(
        accept, COLOR, observed, operator, threshold, units, key, accept_reason, reject_reason,
        reject,
    )


def samples_gate(count: int, cfg: PolicyConfig) -> Gate:
    return _gate(
        PHOTO, PHOTO, int(count), ">=", cfg.color.min_context_samples, "pixels",
        "color.min_context_samples",
        "Enough stable context-ring samples", "Too few stable context-ring samples",
    )


def gain_min_gate(gain: float, cfg: PolicyConfig) -> Gate:
    return _gate(
        PHOTO, LIGHT, gain, ">=", cfg.color.gain_min, "gain",
        "color.gain_min",
        "Per-channel gain above the minimum", "Per-channel gain below the minimum",
    )


def gain_max_gate(gain: float, cfg: PolicyConfig) -> Gate:
    return _gate(
        PHOTO, LIGHT, gain, "<=", cfg.color.gain_max, "gain",
        "color.gain_max",
        "Per-channel gain below the maximum", "Per-channel gain above the maximum",
    )


def bias_min_gate(bias: float, cfg: PolicyConfig) -> Gate:
    return _gate(
        PHOTO, LIGHT, bias, ">=", cfg.color.bias_min_8bit, "8bit",
        "color.bias_min_8bit",
        "Per-channel bias above the minimum", "Per-channel bias below the minimum",
    )


def bias_max_gate(bias: float, cfg: PolicyConfig) -> Gate:
    return _gate(
        PHOTO, LIGHT, bias, "<=", cfg.color.bias_max_8bit, "8bit",
        "color.bias_max_8bit",
        "Per-channel bias below the maximum", "Per-channel bias above the maximum",
    )


def residual_gate(residual: float, cfg: PolicyConfig) -> Gate:
    return _gate(
        PHOTO, PHOTO, residual, "<=", cfg.color.max_mean_abs_residual_8bit, "8bit",
        "color.max_mean_abs_residual_8bit",
        "Mean absolute residual inside the limit", "Mean absolute residual above the limit",
    )


def _fit_channel(donor: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Three-iteration Huber IRLS for ``target = gain * donor + bias``. Deterministic."""
    d = donor.astype(np.float64, copy=False).ravel()
    t = target.astype(np.float64, copy=False).ravel()
    weights = np.ones(d.shape[0], dtype=np.float64)
    gain, bias = 1.0, 0.0
    for _ in range(IRLS_ITERS):
        scale = np.sqrt(weights)
        design = np.column_stack((d * scale, scale))
        beta, _, _, _ = np.linalg.lstsq(design, t * scale, rcond=None)
        gain, bias = float(beta[0]), float(beta[1])
        residual = t - (gain * d + bias)
        mad = float(np.median(np.abs(residual - np.median(residual))))
        if mad < 1e-9:
            break
        thresh = HUBER_K * mad
        abs_r = np.abs(residual)
        weights = np.where(abs_r <= thresh, 1.0, thresh / np.maximum(abs_r, 1e-12))
    return gain, bias


def apply_color(crop: np.ndarray, fit: ColorFit) -> np.ndarray:
    """uint8 BGR: ``clip(rint(gain * crop + bias), 0, 255)`` per channel (float64 internally)."""
    if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError("crop must be a BGR uint8 (H, W, 3) image")
    scaled = np.empty(crop.shape, dtype=np.float64)
    for channel, (gain, bias) in enumerate(zip(fit.gain, fit.bias, strict=True)):
        scaled[..., channel] = gain * crop[..., channel].astype(np.float64) + bias
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def _color_result(
    gain: tuple[float, float, float],
    bias: tuple[float, float, float],
    residual: float,
    samples: int,
    gates: list[Gate],
) -> ColorFit:
    failed = next((gate for gate in gates if not gate.ok), None)
    return ColorFit(
        gain=gain, bias=bias, mean_abs_residual=float(residual), samples=int(samples),
        accepted=failed is None, reason=PHOTO if failed is None else failed.failure_code,
        gates=tuple(gates),
    )


def fit_color(target: Obs, warp: Warp, cfg: PolicyConfig) -> ColorFit:
    """Fit and gate the color transform; the residual is measured after applying the fit."""
    if warp.crop.shape != target.expanded_crop.shape:
        raise ValueError("warp.crop must match target.expanded_crop")
    mask = stable_context_mask(target, warp, cfg)
    samples = int(mask.sum())
    counted = samples_gate(samples, cfg)
    if not counted.ok:
        return _color_result(IDENTITY_GAIN, IDENTITY_BIAS, 0.0, samples, [counted])
    donor_px = warp.crop[mask]
    target_px = target.expanded_crop[mask]
    gains: list[float] = []
    biases: list[float] = []
    for channel in range(3):
        gain, bias = _fit_channel(donor_px[:, channel], target_px[:, channel])
        gains.append(gain)
        biases.append(bias)
    gain_t = (gains[0], gains[1], gains[2])
    bias_t = (biases[0], biases[1], biases[2])
    predicted = np.empty(target_px.shape, dtype=np.float64)
    for channel, (gain, bias) in enumerate(zip(gain_t, bias_t, strict=True)):
        predicted[:, channel] = gain * donor_px[:, channel].astype(np.float64) + bias
    residual = float(np.mean(np.abs(target_px.astype(np.float64) - predicted)))
    gates = [
        counted,
        gain_min_gate(min(gain_t), cfg),
        gain_max_gate(max(gain_t), cfg),
        bias_min_gate(min(bias_t), cfg),
        bias_max_gate(max(bias_t), cfg),
        residual_gate(residual, cfg),
    ]
    return _color_result(gain_t, bias_t, residual, samples, gates)
