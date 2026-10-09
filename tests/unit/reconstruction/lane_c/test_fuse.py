"""Lane C step 9: winner-take-all 8x8 fusion, keep / no-gain / residual reason codes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from probity.domain.enums import ReasonCode
from probity.domain.errors import ValidationFailed
from probity.domain.policy import PolicyConfig
from probity.reconstruction import fuse

from ._builders import (
    FRAME_H,
    FRAME_W,
    TRACK,
    VIDEO,
    checker,
    crop_shape,
    flat_frame,
    make_donor,
    make_obs,
    make_target,
    paint_crop_tile,
    subject_tiles,
)


TILE = (16, 16, 24, 24)


def _cand(
    lut: int,
    s_donor: float,
    *,
    coverage: float = 1.0,
    residual: float | None = 0.0,
    R: float = 0.8,
) -> fuse.TileCandidate:
    return fuse.TileCandidate(lut, s_donor, coverage, residual, R)


def test_relative_improvement_formula(policy: PolicyConfig) -> None:
    eps = policy.fusion.sharpness_ratio_epsilon
    assert fuse.relative_improvement(0.40, 0.20, policy) == pytest.approx(1.0, abs=1e-15)
    assert fuse.relative_improvement(0.10, 0.0, policy) == pytest.approx(0.10 / eps, abs=1e-15)
    assert fuse.relative_improvement(0.10, eps, policy) == pytest.approx((0.10 - eps) / eps, abs=1e-15)
    assert fuse.relative_improvement(0.20, 0.20, policy) == 0.0


def test_selection_score_formula(policy: PolicyConfig) -> None:
    limit = policy.color.max_mean_abs_residual_8bit
    got = fuse.selection_score(0.20, 0.50, 9.0, policy)
    assert got == pytest.approx(0.50 * 0.20 + 0.30 * 0.50 + 0.20 * (1.0 - 9.0 / limit), abs=1e-15)


@pytest.mark.parametrize("R", [-0.01, 1.01, math.nan])
def test_selection_score_rejects_bad_R(policy: PolicyConfig, R: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fuse.selection_score(0.2, R, 0.0, policy)


# ------------------------------------------------------------------------------------------------
# decide_tile: every fusion threshold, just inside and just outside
# ------------------------------------------------------------------------------------------------


def test_keep_original_sharpness_inside_and_outside(policy: PolicyConfig) -> None:
    limit = policy.fusion.keep_original_sharpness
    donor = [_cand(1, 1.0)]
    inside = fuse.decide_tile(0, TILE, limit, donor, policy)
    assert inside.reason_code is ReasonCode.KEEP_ORIGINAL_CLEAR
    assert inside.source_index == 0
    assert inside.score == 0.0
    outside = fuse.decide_tile(0, TILE, limit - 1e-12, donor, policy)
    assert outside.reason_code is ReasonCode.BORROW_TILE_ACCEPTED
    assert outside.source_index == 1


def test_min_sharpness_improvement_inside_and_outside(policy: PolicyConfig) -> None:
    s_target = 0.20
    need = policy.fusion.min_sharpness_improvement
    s_inside = s_target + need * s_target
    while fuse.relative_improvement(s_inside, s_target, policy) < need:
        s_inside = math.nextafter(s_inside, 1.0)
    s_outside = math.nextafter(s_inside, 0.0)
    assert fuse.relative_improvement(s_inside, s_target, policy) >= need
    assert fuse.relative_improvement(s_outside, s_target, policy) < need
    inside = fuse.decide_tile(1, TILE, s_target, [_cand(1, s_inside)], policy)
    assert inside.reason_code is ReasonCode.BORROW_TILE_ACCEPTED
    assert inside.source_index == 1
    outside = fuse.decide_tile(1, TILE, s_target, [_cand(1, s_outside)], policy)
    assert outside.reason_code is ReasonCode.NO_TILE_IMPROVED
    assert outside.source_index == 0


def test_required_coverage_inside_and_outside(policy: PolicyConfig) -> None:
    required = policy.fusion.required_valid_tile_coverage
    s_target, s_donor = 0.10, 1.0
    inside = fuse.decide_tile(2, TILE, s_target, [_cand(1, s_donor, coverage=required)], policy)
    assert inside.reason_code is ReasonCode.BORROW_TILE_ACCEPTED
    outside = fuse.decide_tile(
        2, TILE, s_target, [_cand(1, s_donor, coverage=required - 1.0 / 64.0)], policy
    )
    assert outside.reason_code is ReasonCode.PHOTOMETRIC_INCOMPATIBLE
    assert outside.source_index == 0


def test_residual_gate_inside_and_outside(policy: PolicyConfig) -> None:
    limit = policy.color.max_mean_abs_residual_8bit
    s_target, s_donor = 0.10, 1.0
    inside = fuse.decide_tile(3, TILE, s_target, [_cand(1, s_donor, residual=limit)], policy)
    assert inside.reason_code is ReasonCode.BORROW_TILE_ACCEPTED
    assert inside.residual_8bit == limit
    outside = fuse.decide_tile(
        3, TILE, s_target, [_cand(1, s_donor, residual=limit + 1e-9)], policy
    )
    assert outside.reason_code is ReasonCode.PHOTOMETRIC_INCOMPATIBLE
    assert outside.source_index == 0


def test_tie_breaks_by_earlier_lut_index(policy: PolicyConfig) -> None:
    s_target, s_donor = 0.10, 1.0
    twins = [_cand(1, s_donor, R=0.7), _cand(2, s_donor, R=0.7)]
    decided = fuse.decide_tile(4, TILE, s_target, twins, policy)
    assert decided.reason_code is ReasonCode.BORROW_TILE_ACCEPTED
    assert decided.source_index == 1
    reversed_order = fuse.decide_tile(4, TILE, s_target, [_cand(2, s_donor, R=0.7)], policy)
    assert reversed_order.source_index == 2


def test_higher_selection_score_wins_over_lut_order(policy: PolicyConfig) -> None:
    s_target = 0.10
    low_r = _cand(1, 0.40, residual=0.0, R=0.4)
    high_r = _cand(2, 0.40, residual=0.0, R=0.9)
    decided = fuse.decide_tile(5, TILE, s_target, [low_r, high_r], policy)
    assert decided.source_index == 2
    assert decided.reason_code is ReasonCode.BORROW_TILE_ACCEPTED


def test_empty_candidates_is_no_tile_improved(policy: PolicyConfig) -> None:
    decided = fuse.decide_tile(6, TILE, 0.10, [], policy)
    assert decided.reason_code is ReasonCode.NO_TILE_IMPROVED
    assert decided.source_index == 0


# ------------------------------------------------------------------------------------------------
# fuse_tiles: paste exactly one donor, never blend
# ------------------------------------------------------------------------------------------------


def _ranked_pair(
    target: object,
    policy: PolicyConfig,
    warped_a: np.ndarray,
    warped_b: np.ndarray,
    *,
    valid_a: np.ndarray | None = None,
    valid_b: np.ndarray | None = None,
) -> tuple[object, ...]:
    a = make_donor(target, 20, policy, Q=0.90, warped=warped_a, valid=valid_a)  # type: ignore[arg-type]
    b = make_donor(target, 22, policy, Q=0.70, warped=warped_b, valid=valid_b)  # type: ignore[arg-type]
    ranked = fuse.rank_donors([b, a], policy)
    assert [d.obs.frame_number for d in ranked] == [20, 22]
    return ranked


def test_fuse_tiles_winner_take_all_never_blends(policy: PolicyConfig) -> None:
    target_img, target = make_target(policy)
    shape = crop_shape(target)
    a = checker(shape, 10)
    b = checker(shape, 10, phase=1)
    ranked = _ranked_pair(target, policy, a, b)
    result = fuse.fuse_tiles(target_img, target, ranked, policy)
    assert result.donor_lut_order == (20, 22)
    assert result.result_frame is not target_img
    assert np.array_equal(target_img, flat_frame())
    boxes = subject_tiles(policy)
    assert len(result.tile_decisions) == len(boxes)
    mean = ((a.astype(np.float64) + b.astype(np.float64)) / 2.0).astype(np.uint8)
    ex1, ey1, _, _ = target.expanded_box
    for td, box in zip(result.tile_decisions, boxes, strict=True):
        assert td.reason_code is ReasonCode.BORROW_TILE_ACCEPTED
        assert td.source_index == 1
        x1, y1, x2, y2 = box
        pasted = result.result_frame[y1:y2, x1:x2]
        winner = a[y1 - ey1 : y2 - ey1, x1 - ex1 : x2 - ex1]
        blended = mean[y1 - ey1 : y2 - ey1, x1 - ex1 : x2 - ex1]
        assert np.array_equal(pasted, winner)
        assert not np.array_equal(pasted, blended)
    # Pixels outside the subject box stay the unchanged target.
    x1, y1, x2, y2 = target.bbox_px
    outside = np.ones((FRAME_H, FRAME_W), dtype=bool)
    outside[y1:y2, x1:x2] = False
    assert np.array_equal(result.result_frame[outside], target_img[outside])


def test_fuse_tiles_residual_inside_and_outside(policy: PolicyConfig) -> None:
    target_img, target = make_target(policy)
    shape = crop_shape(target)
    limit = int(policy.color.max_mean_abs_residual_8bit)
    inside_d = make_donor(target, 20, policy, warped=checker(shape, limit))
    outside_d = make_donor(target, 22, policy, Q=0.6, warped=checker(shape, limit + 1))
    inside = fuse.fuse_tiles(target_img, target, (inside_d,), policy)
    assert {td.reason_code for td in inside.tile_decisions} == {ReasonCode.BORROW_TILE_ACCEPTED}
    outside = fuse.fuse_tiles(target_img, target, (outside_d,), policy)
    assert {td.reason_code for td in outside.tile_decisions} == {ReasonCode.PHOTOMETRIC_INCOMPATIBLE}
    assert all(td.source_index == 0 for td in outside.tile_decisions)
    assert np.array_equal(outside.result_frame, target_img)


def test_fuse_tiles_improvement_inside_and_outside(policy: PolicyConfig) -> None:
    target_img, target = make_target(policy)
    shape = crop_shape(target)
    # amp 4: relative improvement vs a flat target is ~0.25 (>= 0.15). amp 1 is ~0.08.
    inside_d = make_donor(target, 20, policy, warped=checker(shape, 4))
    outside_d = make_donor(target, 22, policy, Q=0.6, warped=checker(shape, 1))
    inside = fuse.fuse_tiles(target_img, target, (inside_d,), policy)
    assert {td.reason_code for td in inside.tile_decisions} == {ReasonCode.BORROW_TILE_ACCEPTED}
    outside = fuse.fuse_tiles(target_img, target, (outside_d,), policy)
    assert {td.reason_code for td in outside.tile_decisions} == {ReasonCode.NO_TILE_IMPROVED}
    assert np.array_equal(outside.result_frame, target_img)


def test_fuse_tiles_keep_original_when_target_already_sharp(policy: PolicyConfig) -> None:
    frame = checker((FRAME_H, FRAME_W), 40)
    target = make_obs(frame, 30, policy)
    shape = crop_shape(target)
    donor = make_donor(target, 20, policy, warped=checker(shape, 10))
    result = fuse.fuse_tiles(frame, target, (donor,), policy)
    assert {td.reason_code for td in result.tile_decisions} == {ReasonCode.KEEP_ORIGINAL_CLEAR}
    assert all(td.source_index == 0 for td in result.tile_decisions)
    assert np.array_equal(result.result_frame, frame)


def test_fuse_tiles_coverage_hole_is_photometric(policy: PolicyConfig) -> None:
    target_img, target = make_target(policy)
    shape = crop_shape(target)
    warped = checker(shape, 10)
    valid = np.ones(shape, dtype=bool)
    hole = subject_tiles(policy)[0]
    hx1, hy1, hx2, hy2 = hole
    paint_crop_tile(valid, target, hole, np.zeros((hy2 - hy1, hx2 - hx1), dtype=bool))
    donor = make_donor(target, 20, policy, warped=warped, valid=valid)
    result = fuse.fuse_tiles(target_img, target, (donor,), policy)
    by_box = {td.box: td for td in result.tile_decisions}
    assert by_box[hole].reason_code is ReasonCode.PHOTOMETRIC_INCOMPATIBLE
    assert by_box[hole].source_index == 0
    others = [td for td in result.tile_decisions if td.box != hole]
    assert others and all(td.reason_code is ReasonCode.BORROW_TILE_ACCEPTED for td in others)


def test_fuse_tiles_splits_tiles_across_donors_without_averaging(policy: PolicyConfig) -> None:
    target_img, target = make_target(policy)
    shape = crop_shape(target)
    boxes = subject_tiles(policy)
    a = checker(shape, 10)
    b = checker(shape, 10, phase=1)
    # Right-half tiles of A fail the residual gate (amp 40); B stays eligible there.
    for box in boxes:
        if box[0] >= 32:
            paint_crop_tile(a, target, box, checker((box[3] - box[1], box[2] - box[0]), 40))
    ranked = _ranked_pair(target, policy, a, b)
    result = fuse.fuse_tiles(target_img, target, ranked, policy)
    ex1, ey1, _, _ = target.expanded_box
    saw_a = saw_b = False
    for td, box in zip(result.tile_decisions, boxes, strict=True):
        x1, y1, x2, y2 = box
        pasted = result.result_frame[y1:y2, x1:x2]
        crop_a = a[y1 - ey1 : y2 - ey1, x1 - ex1 : x2 - ex1]
        crop_b = b[y1 - ey1 : y2 - ey1, x1 - ex1 : x2 - ex1]
        if x1 >= 32:
            assert td.source_index == 2
            assert np.array_equal(pasted, crop_b)
            saw_b = True
        else:
            assert td.source_index == 1
            assert np.array_equal(pasted, crop_a)
            saw_a = True
        assert not np.array_equal(pasted, ((crop_a.astype(np.float64) + crop_b.astype(np.float64)) / 2.0).astype(np.uint8))
    assert saw_a and saw_b


def test_fuse_tiles_rejects_donors_not_in_lut_order(policy: PolicyConfig) -> None:
    _, target = make_target(policy)
    a = make_donor(target, 20, policy, Q=0.9)
    b = make_donor(target, 22, policy, Q=0.6)
    with pytest.raises(ValidationFailed, match="LUT order"):
        fuse.fuse_tiles(flat_frame(), target, (b, a), policy)


def test_fuse_tiles_rejects_target_frame_as_donor(policy: PolicyConfig) -> None:
    img, target = make_target(policy)
    donor = make_donor(target, target.frame_number, policy)
    with pytest.raises(ValidationFailed, match="target frame"):
        fuse.fuse_tiles(img, target, (donor,), policy)


@pytest.mark.parametrize("field", ["video_id", "track_id"])
def test_fuse_tiles_never_crosses_tracks_or_videos(policy: PolicyConfig, field: str) -> None:
    img, target = make_target(policy)
    other = {"video_id": VIDEO, "track_id": TRACK, field: "01a1219b-7a80-7bd1-b62c-000000000000"}
    donors = [make_donor(target, 20, policy), make_donor(target, 22, policy, Q=0.6, **other)]
    with pytest.raises(ValidationFailed, match="video/track"):
        fuse.fuse_tiles(img, target, donors, policy)


def test_fuse_tiles_rejects_over_max_count(policy: PolicyConfig) -> None:
    img, target = make_target(policy)
    donors = [make_donor(target, 20 + i, policy, Q=0.9 - 0.01 * i) for i in range(policy.donor.max_count + 1)]
    with pytest.raises(ValidationFailed, match="max_count"):
        fuse.fuse_tiles(img, target, donors, policy)
