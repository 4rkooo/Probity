"""Lane B: alignment confidence, inverse source maps, warp, coverage/corner/scale measures."""

from __future__ import annotations

import math
from collections.abc import Callable

import cv2
import numpy as np
import pytest

from probity.domain.enums import AlignmentMethod, PolicyStage, ReasonCode
from probity.domain.policy import PolicyConfig
from probity.reconstruction import align
from probity.reconstruction.types import Alignment, Obs

from .scenes import FRAME_H, FRAME_W, SUBJECT_BOX, scaling, translation, warp_perspective


def accepted(matrix: np.ndarray) -> Alignment:
    return Alignment(
        method=AlignmentMethod.AKAZE_HOMOGRAPHY, matrix=matrix, inlier_ratio=1.0,
        median_reproj_px=0.0, valid_coverage=1.0, corner_outside_fraction=0.0, A=1.0,
        accepted=True, reason=ReasonCode.ALIGNMENT_AKAZE_ACCEPTED,
    )


# ------------------------------------------------------------------------------------------------
# A (section 9)
# ------------------------------------------------------------------------------------------------


def test_alignment_confidence_formula() -> None:
    assert align.alignment_confidence(1.0, 0.0, 1.0) == pytest.approx(1.0)
    assert align.alignment_confidence(0.65, 1.0, 0.85) == pytest.approx(
        0.40 * 0.65 + 0.30 * 0.5 + 0.30 * 0.85)
    # Reprojection term clips at 2 px: 2 px and 9 px both score zero.
    assert align.alignment_confidence(0.5, 2.0, 0.5) == pytest.approx(0.35)
    assert align.alignment_confidence(0.5, 9.0, 0.5) == pytest.approx(0.35)


@pytest.mark.parametrize(
    "args", [(1.1, 0.0, 1.0), (-0.1, 0.0, 1.0), (1.0, -0.5, 1.0), (1.0, math.nan, 1.0),
             (1.0, 0.0, 1.0001), (math.inf, 0.0, 1.0)])
def test_alignment_confidence_rejects_out_of_range(args: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError, match="must be"):
        align.alignment_confidence(*args)


# ------------------------------------------------------------------------------------------------
# Source maps and warp
# ------------------------------------------------------------------------------------------------


def test_source_maps_are_the_inverse_transform() -> None:
    h = np.array([[1.05, 0.02, -7.0], [-0.01, 0.98, 4.0], [1e-5, -2e-5, 1.0]])
    box = (100, 90, 160, 130)
    sx, sy, valid = align.source_maps(h, box, (FRAME_W, FRAME_H))
    assert sx.dtype == sy.dtype == np.float32 and valid.dtype == bool
    assert sx.shape == sy.shape == valid.shape == (40, 60)
    ys, xs = np.mgrid[90:130, 100:160]
    pts = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2).astype(np.float64)
    back = cv2.perspectiveTransform(pts, np.linalg.inv(h)).reshape(40, 60, 2)
    np.testing.assert_allclose(sx, back[..., 0], atol=1e-3)
    np.testing.assert_allclose(sy, back[..., 1], atol=1e-3)
    assert valid.all()


def test_valid_mask_is_closed_interval_on_float32_coordinates() -> None:
    # Target column x maps to donor column x + 10: columns up to W-11 are valid, W-10 is not.
    box = (FRAME_W - 14, 0, FRAME_W, 2)
    sx, _, valid = align.source_maps(translation(-10.0, 0.0), box, (FRAME_W, FRAME_H))
    assert sx[0, 3] == FRAME_W - 1 and valid[0, 3]
    assert sx[0, 4] == FRAME_W and not valid[0, 4]
    assert valid[:, :4].all() and not valid[:, 4:].any()


def test_points_beyond_the_horizon_are_invalid() -> None:
    # w' = 1 - x/100 for the inverse: columns >= 100 project behind the camera.
    inv = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.01, 0.0, 1.0]])
    sx, sy, valid = align.source_maps(np.linalg.inv(inv), (40, 0, 160, 4), (10_000, 10_000))
    cols = np.arange(40, 160)
    assert not valid[:, cols >= 100].any()
    assert np.isfinite(sx).all() and np.isfinite(sy).all()
    assert (sx[:, cols >= 100] == -1.0).all()


def test_integer_translation_warp_reproduces_target_exactly(
    scene: np.ndarray, obs_of: Callable[..., Obs]
) -> None:
    dx, dy = 7, -5
    donor_frame = scene
    target_frame = np.ascontiguousarray(np.roll(scene, (dy, dx), axis=(0, 1)))
    target = obs_of(target_frame, SUBJECT_BOX, 30)
    warp = align.warp_donor(donor_frame, accepted(translation(dx, dy)), target)
    ex1, ey1, ex2, ey2 = target.expanded_box
    assert warp.crop.shape == target.expanded_crop.shape and warp.crop.dtype == np.uint8
    assert np.array_equal(warp.crop, target.expanded_crop)
    assert warp.valid_mask.all()
    np.testing.assert_array_equal(warp.source_x[0], np.arange(ex1, ex2, dtype=np.float32) - dx)
    np.testing.assert_array_equal(warp.source_y[:, 0], np.arange(ey1, ey2, dtype=np.float32) - dy)


def test_warp_equals_documented_remap(scene: np.ndarray, obs_of: Callable[..., Obs]) -> None:
    h = scaling(1.1, 240.0, 144.0) @ translation(2.3, -1.7)
    target = obs_of(warp_perspective(scene, h), SUBJECT_BOX, 30)
    warp = align.warp_donor(scene, accepted(h), target)
    expected = cv2.remap(scene, warp.source_x, warp.source_y, cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_REFLECT_101)
    assert np.array_equal(warp.crop, expected)
    # The known homography is recovered pixel-for-pixel away from interpolation ringing.
    diff = np.abs(warp.crop.astype(int) - target.expanded_crop.astype(int))
    assert float(np.median(diff)) <= 2.0


def test_warp_is_deterministic(scene: np.ndarray, obs_of: Callable[..., Obs]) -> None:
    h = translation(3.25, 1.5)
    target = obs_of(scene, SUBJECT_BOX, 30)
    a = align.warp_donor(scene, accepted(h), target)
    b = align.warp_donor(scene.copy(), accepted(h.copy()), target)
    for x, y in [(a.crop, b.crop), (a.valid_mask, b.valid_mask), (a.source_x, b.source_x),
                 (a.source_y, b.source_y)]:
        assert np.array_equal(x, y)


def test_warp_refuses_unaccepted_or_missing_alignment(
    scene: np.ndarray, obs_of: Callable[..., Obs]
) -> None:
    target = obs_of(scene, SUBJECT_BOX, 30)
    rejected = Alignment(
        method=AlignmentMethod.AKAZE_HOMOGRAPHY, matrix=translation(1, 1), inlier_ratio=0.1,
        median_reproj_px=5.0, valid_coverage=1.0, corner_outside_fraction=0.0, A=0.1,
        accepted=False, reason=ReasonCode.ALIGNMENT_FAILED,
    )
    with pytest.raises(ValueError, match="accepted"):
        align.warp_donor(scene, rejected, target)
    with pytest.raises(ValueError, match="BGR uint8"):
        align.warp_donor(scene[..., 0].copy(), accepted(translation(1, 1)), target)
    with pytest.raises(ValueError, match="singular"):
        align.warp_donor(scene, accepted(np.zeros((3, 3))), target)


# ------------------------------------------------------------------------------------------------
# Coverage, corners, scale
# ------------------------------------------------------------------------------------------------


def test_valid_coverage_and_corner_fraction(scene: np.ndarray, obs_of: Callable[..., Obs]
                                            ) -> None:
    target = obs_of(scene, SUBJECT_BOX, 30)
    ex1, ey1, ex2, ey2 = target.expanded_box
    width = ex2 - ex1
    size = (FRAME_W, FRAME_H)
    assert align.valid_coverage(translation(0, 0), target, size) == 1.0
    assert align.corner_outside_fraction(translation(0, 0), target, size) == 0.0
    # Shift so the donor source starts 10 columns left of the frame: those columns are invalid.
    m = translation(ex1 + 10, 0)
    assert align.valid_coverage(m, target, size) == pytest.approx((width - 10) / width)
    assert align.corner_outside_fraction(m, target, size) == 0.5
    far = translation(10_000, 10_000)
    assert align.valid_coverage(far, target, size) == 0.0
    assert align.corner_outside_fraction(far, target, size) == 1.0


def test_transform_scale() -> None:
    box = SUBJECT_BOX
    assert align.transform_scale(np.eye(3), box) == pytest.approx(1.0)
    assert align.transform_scale(scaling(1.3, 10, 20), box) == pytest.approx(1.3)
    assert align.transform_scale(np.diag([2.0, 0.5, 1.0]), box) == pytest.approx(1.0)
    assert align.transform_scale(np.diag([-1.0, 1.0, 1.0]), box) == 0.0  # mirrored
    horizon = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1 / 200, 0.0, 1.0]])
    assert align.transform_scale(horizon, box) == 0.0  # box straddles the horizon


# ------------------------------------------------------------------------------------------------
# Gate boundaries: just inside passes with the accept code, just outside rejects ALIGNMENT_FAILED
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("accept", [ReasonCode.ALIGNMENT_AKAZE_ACCEPTED,
                                    ReasonCode.ALIGNMENT_ECC_FALLBACK_ACCEPTED])
def test_corner_gate_boundary(policy: PolicyConfig, accept: ReasonCode) -> None:
    limit = policy.alignment.max_corner_outside_fraction
    inside = align.corner_gate(limit, policy, accept)
    outside = align.corner_gate(math.nextafter(limit, 1.0), policy, accept)
    assert inside.ok and inside.rule_code is accept and inside.stage is PolicyStage.ALIGN
    assert not outside.ok and outside.failure_code is ReasonCode.ALIGNMENT_FAILED
    assert inside.policy_key == "alignment.max_corner_outside_fraction"
    # Real corner fractions are multiples of 1/4: one corner outside already fails.
    assert not align.corner_gate(0.25, policy, accept).ok and align.corner_gate(0.0, policy).ok


@pytest.mark.parametrize("accept", [ReasonCode.ALIGNMENT_AKAZE_ACCEPTED,
                                    ReasonCode.ALIGNMENT_ECC_FALLBACK_ACCEPTED])
def test_coverage_gate_boundary(policy: PolicyConfig, accept: ReasonCode) -> None:
    limit = policy.alignment.min_valid_coverage
    inside = align.coverage_gate(limit, policy, accept)
    outside = align.coverage_gate(math.nextafter(limit, 0.0), policy, accept)
    assert inside.ok and inside.rule_code is accept
    assert not outside.ok and outside.failure_code is ReasonCode.ALIGNMENT_FAILED
    assert (inside.policy_key, inside.operator, inside.threshold) == (
        "alignment.min_valid_coverage", ">=", limit)
