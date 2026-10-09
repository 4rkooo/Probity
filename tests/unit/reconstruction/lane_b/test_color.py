"""Lane B: context-ring color fit, apply_color, and every color-gate boundary."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from probity.domain.enums import AlignmentMethod, PolicyStage, ReasonCode
from probity.domain.policy import PolicyConfig
from probity.reconstruction import align, color
from probity.reconstruction.types import Alignment, ColorFit, Obs, Warp

from .scenes import FRAME_H, FRAME_W, SUBJECT_BOX, flat_frame, textured_frame

PHOTO = ReasonCode.PHOTOMETRIC_INCOMPATIBLE
LIGHT = ReasonCode.LIGHTING_OUT_OF_RANGE


def _identity_alignment() -> Alignment:
    return Alignment(
        method=AlignmentMethod.AKAZE_HOMOGRAPHY, matrix=np.eye(3), inlier_ratio=1.0,
        median_reproj_px=0.0, valid_coverage=1.0, corner_outside_fraction=0.0, A=1.0,
        accepted=True, reason=ReasonCode.ALIGNMENT_AKAZE_ACCEPTED,
    )


def _warp(frame: np.ndarray, target: Obs, valid: np.ndarray | None = None) -> Warp:
    warp = align.warp_donor(frame, _identity_alignment(), target)
    if valid is None:
        return warp
    return Warp(crop=warp.crop, valid_mask=valid, source_x=warp.source_x, source_y=warp.source_y)


def _fit(gain: tuple[float, float, float], bias: tuple[float, float, float],
         residual: float = 0.0, samples: int = 200, accepted: bool = True) -> ColorFit:
    return ColorFit(
        gain=gain, bias=bias, mean_abs_residual=residual, samples=samples,
        accepted=accepted, reason=PHOTO,
    )


# ------------------------------------------------------------------------------------------------
# Masks
# ------------------------------------------------------------------------------------------------


def test_context_ring_is_expanded_minus_subject(
    obs_of: Callable[..., Obs],
) -> None:
    frame = flat_frame(80)
    target = obs_of(frame, SUBJECT_BOX, 30)
    ring = color.context_ring_mask(target)
    ex1, ey1, ex2, ey2 = target.expanded_box
    assert ring.shape == (ey2 - ey1, ex2 - ex1) and ring.dtype == bool
    x1, y1, x2, y2 = target.bbox_px
    assert not ring[y1 - ey1:y2 - ey1, x1 - ex1:x2 - ex1].any()
    assert ring.sum() == ring.size - (x2 - x1) * (y2 - y1)


def test_stable_context_requires_valid_warp_and_low_target_gradient(
    obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    frame = flat_frame(80)
    target = obs_of(frame, SUBJECT_BOX, 30)
    ring = color.context_ring_mask(target)
    none_valid = _warp(frame, target, valid=np.zeros(ring.shape, dtype=bool))
    assert not color.stable_context_mask(target, none_valid, policy).any()
    all_valid = _warp(frame, target)
    stable = color.stable_context_mask(target, all_valid, policy)
    assert np.array_equal(stable, ring)

    edged = frame.copy()
    ex1, ey1, _, _ = target.expanded_box
    # A vertical step in the ring (left of the subject) produces a large Sobel magnitude.
    edged[:, :SUBJECT_BOX[0] - 2] = 10
    edged[:, SUBJECT_BOX[0] - 2:] = 240
    sharp = obs_of(edged, SUBJECT_BOX, 31)
    sharp_stable = color.stable_context_mask(sharp, _warp(edged, sharp), policy)
    mag = color._sobel_magnitude_8bit(sharp.expanded_crop)
    assert (mag[sharp_stable] <= policy.color.max_context_gradient_8bit).all()
    assert sharp_stable.sum() < ring.sum()


# ------------------------------------------------------------------------------------------------
# apply_color
# ------------------------------------------------------------------------------------------------


def test_apply_color_rint_and_clip() -> None:
    crop = np.array([[[0, 10, 200]]], dtype=np.uint8)
    out = color.apply_color(crop, _fit((1.1, 0.5, 2.0), (5.4, -3.2, 0.0)))
    assert out.dtype == np.uint8 and out.shape == crop.shape
    assert out[0, 0, 0] == 5   # rint(0*1.1 + 5.4) = 5
    assert out[0, 0, 1] == 2   # rint(10*0.5 - 3.2) = rint(1.8) = 2
    assert out[0, 0, 2] == 255  # clip 400
    with pytest.raises(ValueError, match="BGR uint8"):
        color.apply_color(crop[..., 0].copy(), _fit((1, 1, 1), (0, 0, 0)))


def test_apply_color_is_deterministic() -> None:
    crop = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    fit = _fit((0.9, 1.1, 1.0), (-2.0, 3.5, 0.25))
    assert np.array_equal(color.apply_color(crop, fit), color.apply_color(crop.copy(), fit))


# ------------------------------------------------------------------------------------------------
# Gate boundaries
# ------------------------------------------------------------------------------------------------


def test_samples_gate_boundary(policy: PolicyConfig) -> None:
    limit = policy.color.min_context_samples
    inside = color.samples_gate(limit, policy)
    outside = color.samples_gate(limit - 1, policy)
    assert inside.ok and inside.rule_code is PHOTO and inside.stage is PolicyStage.COLOR
    assert not outside.ok and outside.failure_code is PHOTO
    assert inside.policy_key == "color.min_context_samples"


def test_gain_gates_boundary(policy: PolicyConfig) -> None:
    lo, hi = policy.color.gain_min, policy.color.gain_max
    assert color.gain_min_gate(lo, policy).ok
    low = color.gain_min_gate(math.nextafter(lo, 0.0), policy)
    assert not low.ok and low.failure_code is LIGHT and low.policy_key == "color.gain_min"
    assert color.gain_max_gate(hi, policy).ok
    high = color.gain_max_gate(math.nextafter(hi, 10.0), policy)
    assert not high.ok and high.failure_code is LIGHT and high.policy_key == "color.gain_max"


def test_bias_gates_boundary(policy: PolicyConfig) -> None:
    lo, hi = policy.color.bias_min_8bit, policy.color.bias_max_8bit
    assert color.bias_min_gate(lo, policy).ok
    low = color.bias_min_gate(math.nextafter(float(lo), -1e9), policy)
    assert not low.ok and low.failure_code is LIGHT and low.policy_key == "color.bias_min_8bit"
    assert color.bias_max_gate(hi, policy).ok
    high = color.bias_max_gate(math.nextafter(float(hi), 1e9), policy)
    assert not high.ok and high.failure_code is LIGHT and high.policy_key == "color.bias_max_8bit"


def test_residual_gate_boundary(policy: PolicyConfig) -> None:
    limit = policy.color.max_mean_abs_residual_8bit
    inside = color.residual_gate(limit, policy)
    outside = color.residual_gate(math.nextafter(limit, 1e9), policy)
    assert inside.ok and inside.rule_code is PHOTO
    assert not outside.ok and outside.failure_code is PHOTO
    assert inside.policy_key == "color.max_mean_abs_residual_8bit"


# ------------------------------------------------------------------------------------------------
# Fit on seeded / hand-built warps
# ------------------------------------------------------------------------------------------------


def test_identity_fit_is_near_identity(
    obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    frame = textured_frame(7)
    target = obs_of(frame, SUBJECT_BOX, 30)
    result = color.fit_color(target, _warp(frame, target), policy)
    assert result.accepted and result.reason is PHOTO
    assert result.samples >= policy.color.min_context_samples
    np.testing.assert_allclose(result.gain, (1.0, 1.0, 1.0), atol=0.05)
    np.testing.assert_allclose(result.bias, (0.0, 0.0, 0.0), atol=2.0)
    assert result.mean_abs_residual <= policy.color.max_mean_abs_residual_8bit
    assert all(g.ok for g in result.gates)


def _photometric_pair(
    obs_of: Callable[..., Obs], donor: np.ndarray, target_frame: np.ndarray,
) -> tuple[Obs, Warp]:
    target = obs_of(target_frame, SUBJECT_BOX, 30)
    sx, sy, valid = align.source_maps(np.eye(3), target.expanded_box, (FRAME_W, FRAME_H))
    ey1, ey2 = target.expanded_box[1], target.expanded_box[3]
    ex1, ex2 = target.expanded_box[0], target.expanded_box[2]
    warp = Warp(
        crop=np.ascontiguousarray(donor[ey1:ey2, ex1:ex2]),
        valid_mask=valid, source_x=sx, source_y=sy,
    )
    return target, warp


def test_recovers_in_range_gain_and_bias(
    obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    donor = (textured_frame(3) // 2 + 40).astype(np.uint8)
    target_frame = np.clip(np.rint(1.10 * donor.astype(np.float64) + 8.0), 0, 255).astype(np.uint8)
    target, warp = _photometric_pair(obs_of, donor, target_frame)
    result = color.fit_color(target, warp, policy)
    assert result.accepted and result.reason is PHOTO
    np.testing.assert_allclose(result.gain, (1.10, 1.10, 1.10), atol=0.03)
    np.testing.assert_allclose(result.bias, (8.0, 8.0, 8.0), atol=1.0)


def test_too_few_stable_samples_is_photometric_incompatible(
    obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    frame = flat_frame(90)
    target = obs_of(frame, SUBJECT_BOX, 30)
    ring = color.context_ring_mask(target)
    coords = np.argwhere(ring)

    def restricted(n: int) -> Warp:
        mask = np.zeros(ring.shape, dtype=bool)
        for row, col in coords[:n]:
            mask[row, col] = True
        return _warp(frame, target, valid=mask)

    inside = color.fit_color(target, restricted(policy.color.min_context_samples), policy)
    outside = color.fit_color(target, restricted(policy.color.min_context_samples - 1), policy)
    assert inside.accepted and inside.reason is PHOTO
    assert inside.samples == policy.color.min_context_samples
    assert not outside.accepted and outside.reason is PHOTO
    assert outside.gates[0].policy_key == "color.min_context_samples"
    assert outside.gain == (1.0, 1.0, 1.0) and outside.bias == (0.0, 0.0, 0.0)


def test_gain_above_max_is_lighting_out_of_range(
    obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    donor = (textured_frame(4) // 2 + 40).astype(np.uint8)
    target_frame = np.clip(np.rint(1.50 * donor.astype(np.float64)), 0, 255).astype(np.uint8)
    result = color.fit_color(*_photometric_pair(obs_of, donor, target_frame), policy)
    assert not result.accepted and result.reason is LIGHT
    failed = next(g for g in result.gates if not g.ok)
    assert failed.policy_key == "color.gain_max" and failed.failure_code is LIGHT


def test_gain_below_min_is_lighting_out_of_range(
    obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    donor = (textured_frame(5) // 2 + 80).astype(np.uint8)
    target_frame = np.clip(np.rint(0.50 * donor.astype(np.float64)), 0, 255).astype(np.uint8)
    result = color.fit_color(*_photometric_pair(obs_of, donor, target_frame), policy)
    assert not result.accepted and result.reason is LIGHT
    failed = next(g for g in result.gates if not g.ok)
    assert failed.policy_key == "color.gain_min" and failed.failure_code is LIGHT


def test_bias_below_min_is_lighting_out_of_range(
    obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    donor = (textured_frame(7) // 2 + 80).astype(np.uint8)
    target_frame = np.clip(donor.astype(np.int16) - 25, 0, 255).astype(np.uint8)
    result = color.fit_color(*_photometric_pair(obs_of, donor, target_frame), policy)
    assert not result.accepted and result.reason is LIGHT
    failed = next(g for g in result.gates if not g.ok)
    assert failed.policy_key == "color.bias_min_8bit" and failed.failure_code is LIGHT


def test_bias_above_max_is_lighting_out_of_range(
    obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    donor = (textured_frame(6) // 2 + 40).astype(np.uint8)
    target_frame = np.clip(donor.astype(np.int16) + 25, 0, 255).astype(np.uint8)
    result = color.fit_color(*_photometric_pair(obs_of, donor, target_frame), policy)
    assert not result.accepted and result.reason is LIGHT
    failed = next(g for g in result.gates if not g.ok)
    assert failed.policy_key == "color.bias_max_8bit" and failed.failure_code is LIGHT


def test_checkerboard_offset_fails_residual(
    obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    donor = (textured_frame(8) // 2 + 64).astype(np.uint8)
    parity = (np.arange(FRAME_H)[:, None] + np.arange(FRAME_W)) % 2
    delta = np.where(parity == 0, 30, -30).astype(np.int16)
    target_frame = np.clip(donor.astype(np.int16) + delta[..., None], 0, 255).astype(np.uint8)
    result = color.fit_color(*_photometric_pair(obs_of, donor, target_frame), policy)
    assert not result.accepted and result.reason is PHOTO
    failed = next(g for g in result.gates if not g.ok)
    assert failed.policy_key == "color.max_mean_abs_residual_8bit"
    assert failed.failure_code is PHOTO


def test_fit_color_is_deterministic(
    obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    frame = textured_frame(11)
    target = obs_of(frame, SUBJECT_BOX, 30)
    warp = _warp(frame, target)
    a = color.fit_color(target, warp, policy)
    b = color.fit_color(target, warp, policy)
    assert a.gain == b.gain and a.bias == b.bias
    assert a.mean_abs_residual == b.mean_abs_residual and a.samples == b.samples
    applied = color.apply_color(warp.crop, a)
    assert np.array_equal(applied, color.apply_color(warp.crop, b))
