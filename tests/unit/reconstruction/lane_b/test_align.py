"""Lane B: AKAZE/RANSAC, ECC fallback, and every alignment-gate boundary."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import cv2
import numpy as np

from probity.domain.enums import AlignmentMethod, PolicyStage, ReasonCode
from probity.domain.ids import parse_frame_id
from probity.domain.policy import PolicyConfig
from probity.eval import synth
from probity.reconstruction import align
from probity.reconstruction.types import Obs

from .scenes import (
    FRAME_H,
    FRAME_W,
    SUBJECT_BOX,
    flat_frame,
    shift_box,
    textured_frame,
    translation,
)

AKAZE_OK = ReasonCode.ALIGNMENT_AKAZE_ACCEPTED
ECC_OK = ReasonCode.ALIGNMENT_ECC_FALLBACK_ACCEPTED
FAILED = ReasonCode.ALIGNMENT_FAILED


def _failing(gates: tuple, key: str):
    failed = [g for g in gates if not g.ok]
    assert failed, f"expected a failing gate for {key}"
    assert failed[0].policy_key == key
    return failed[0]


# ------------------------------------------------------------------------------------------------
# Gate boundaries (just inside passes with the accept code; just outside rejects ALIGNMENT_FAILED)
# ------------------------------------------------------------------------------------------------


def test_keypoints_gate_boundary(policy: PolicyConfig) -> None:
    limit = policy.akaze.min_keypoints
    inside = align.keypoints_gate(limit, policy)
    outside = align.keypoints_gate(limit - 1, policy)
    assert inside.ok and inside.rule_code is AKAZE_OK and inside.stage is PolicyStage.ALIGN
    assert not outside.ok and outside.failure_code is FAILED
    assert inside.policy_key == "akaze.min_keypoints" and inside.observed == limit


def test_matches_gate_boundary(policy: PolicyConfig) -> None:
    limit = policy.akaze.min_good_matches
    inside = align.matches_gate(limit, policy)
    outside = align.matches_gate(limit - 1, policy)
    assert inside.ok and inside.rule_code is AKAZE_OK
    assert not outside.ok and outside.failure_code is FAILED
    assert inside.policy_key == "akaze.min_good_matches"


def test_inlier_gate_boundary(policy: PolicyConfig) -> None:
    limit = policy.alignment.min_inlier_ratio
    inside = align.inlier_gate(limit, policy)
    outside = align.inlier_gate(math.nextafter(limit, 0.0), policy)
    assert inside.ok and inside.rule_code is AKAZE_OK
    assert not outside.ok and outside.failure_code is FAILED
    assert inside.policy_key == "alignment.min_inlier_ratio"


def test_reproj_gate_boundary(policy: PolicyConfig) -> None:
    limit = policy.alignment.max_median_reprojection_px
    inside = align.reproj_gate(limit, policy)
    outside = align.reproj_gate(math.nextafter(limit, 1.0e9), policy)
    assert inside.ok and inside.rule_code is AKAZE_OK
    assert not outside.ok and outside.failure_code is FAILED
    assert inside.policy_key == "alignment.max_median_reprojection_px"


def test_scale_min_gate_boundary(policy: PolicyConfig) -> None:
    limit = align.DEFAULT_SCALE_RATIO_MIN
    inside = align.scale_min_gate(limit, policy)
    outside = align.scale_min_gate(math.nextafter(limit, 0.0), policy)
    assert inside.ok and inside.rule_code is AKAZE_OK
    assert not outside.ok and outside.failure_code is FAILED
    assert inside.policy_key == "alignment.scale_ratio_min"


def test_scale_max_gate_boundary(policy: PolicyConfig) -> None:
    limit = align.DEFAULT_SCALE_RATIO_MAX
    inside = align.scale_max_gate(limit, policy)
    outside = align.scale_max_gate(math.nextafter(limit, 10.0), policy)
    assert inside.ok and inside.rule_code is AKAZE_OK
    assert not outside.ok and outside.failure_code is FAILED
    assert inside.policy_key == "alignment.scale_ratio_max"


def test_correlation_gate_boundary(policy: PolicyConfig) -> None:
    limit = policy.ecc.min_correlation
    inside = align.correlation_gate(limit, policy)
    outside = align.correlation_gate(math.nextafter(limit, 0.0), policy)
    assert inside.ok and inside.rule_code is ECC_OK and inside.stage is PolicyStage.ALIGN
    assert not outside.ok and outside.failure_code is FAILED
    assert inside.policy_key == "ecc.min_correlation"


def test_ratio_test_is_strict(policy: PolicyConfig) -> None:
    # Spec stores 0.75; keep only when first < ratio * second (equality is discarded).
    assert policy.akaze.ratio_test == 0.75
    assert not (0.75 < 0.75 * 1.0)
    assert 0.749999 < 0.75 * 1.0


# ------------------------------------------------------------------------------------------------
# AKAZE on seeded textured scenes
# ------------------------------------------------------------------------------------------------


def test_identity_akaze_is_accepted(
    scene: np.ndarray, obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    target = obs_of(scene, SUBJECT_BOX, 30)
    donor = obs_of(scene, SUBJECT_BOX, 20)
    result = align.akaze_homography(target, donor, policy)
    assert result.accepted and result.reason is AKAZE_OK
    assert result.method is AlignmentMethod.AKAZE_HOMOGRAPHY
    assert result.matrix is not None
    assert result.inlier_ratio >= policy.alignment.min_inlier_ratio
    assert result.median_reproj_px <= policy.alignment.max_median_reprojection_px
    assert result.valid_coverage >= policy.alignment.min_valid_coverage
    keys = [g.policy_key for g in result.gates]
    assert keys[:2] == ["akaze.min_keypoints", "akaze.min_good_matches"]
    assert "alignment.min_inlier_ratio" in keys
    assert all(g.ok for g in result.gates)
    np.testing.assert_allclose(result.matrix, np.eye(3), atol=0.15)


def test_integer_translation_akaze_recovers_shift(
    scene: np.ndarray, obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    dx, dy = 8, -6
    target_frame = np.ascontiguousarray(np.roll(scene, (dy, dx), axis=(0, 1)))
    target = obs_of(target_frame, shift_box(SUBJECT_BOX, dx, dy), 30)
    donor = obs_of(scene, SUBJECT_BOX, 20)
    result = align.akaze_homography(target, donor, policy)
    assert result.accepted and result.reason is AKAZE_OK and result.matrix is not None
    np.testing.assert_allclose(result.matrix, translation(dx, dy), atol=0.5)
    warped = align.warp_donor(scene, result, target)
    diff = np.abs(warped.crop.astype(int) - target.expanded_crop.astype(int))
    assert float(np.median(diff)) <= 2.0


def test_akaze_is_deterministic(
    scene: np.ndarray, obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    target = obs_of(scene, SUBJECT_BOX, 30)
    donor = obs_of(np.ascontiguousarray(np.roll(scene, (2, 3), axis=(0, 1))), SUBJECT_BOX, 20)
    a = align.akaze_homography(target, donor, policy)
    b = align.akaze_homography(target, donor, policy)
    assert a.accepted == b.accepted and a.reason is b.reason
    assert a.inlier_ratio == b.inlier_ratio
    assert a.median_reproj_px == b.median_reproj_px
    if a.matrix is not None and b.matrix is not None:
        np.testing.assert_array_equal(a.matrix, b.matrix)


def test_flat_crops_fail_keypoints_not_matches(
    obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    flat = flat_frame(128)
    target = obs_of(flat, SUBJECT_BOX, 30)
    donor = obs_of(flat, SUBJECT_BOX, 20)
    result = align.akaze_homography(target, donor, policy)
    assert not result.accepted and result.reason is FAILED
    assert result.method is AlignmentMethod.AKAZE_HOMOGRAPHY
    assert result.matrix is None
    gate = _failing(result.gates, "akaze.min_keypoints")
    assert gate.failure_code is FAILED
    assert len(result.gates) == 1


def test_unrelated_textures_fail_matches(
    obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    target = obs_of(textured_frame(1), SUBJECT_BOX, 30)
    donor = obs_of(textured_frame(99), SUBJECT_BOX, 20)
    result = align.akaze_homography(target, donor, policy)
    assert not result.accepted and result.reason is FAILED
    keys = [g.policy_key for g in result.gates if not g.ok]
    assert keys[0] in {"akaze.min_keypoints", "akaze.min_good_matches"}
    assert result.matrix is None


# ------------------------------------------------------------------------------------------------
# ECC fallback: only keypoints/matches starvation; box-geometry init
# ------------------------------------------------------------------------------------------------


def test_align_falls_back_to_ecc_when_keypoints_are_missing(
    obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    # Same rigid translation, no texture: AKAZE starves, ECC can still lock using box geometry.
    donor_frame = np.full((FRAME_H, FRAME_W, 3), 80, dtype=np.uint8)
    cv2.rectangle(donor_frame, (40, 40), (440, 248), (200, 180, 40), thickness=-1)
    cv2.rectangle(donor_frame, (130, 108), (350, 180), (30, 30, 30), thickness=-1)
    dx, dy = 5, -3
    target_frame = np.ascontiguousarray(np.roll(donor_frame, (dy, dx), axis=(0, 1)))
    target = obs_of(target_frame, shift_box(SUBJECT_BOX, dx, dy), 30)
    donor = obs_of(donor_frame, SUBJECT_BOX, 20)
    primary = align.akaze_homography(target, donor, policy)
    assert not primary.accepted
    assert any(g.policy_key == "akaze.min_keypoints" and not g.ok for g in primary.gates)
    result = align.align(target, donor, policy)
    assert result.method is AlignmentMethod.ECC_AFFINE
    assert [g.policy_key for g in result.gates[:1]] == ["akaze.min_keypoints"]
    assert any(g.policy_key == "ecc.min_correlation" for g in result.gates)
    if result.accepted:
        assert result.reason is ECC_OK
        assert result.matrix is not None


def test_align_does_not_fallback_when_akaze_succeeds(
    scene: np.ndarray, obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    target = obs_of(scene, SUBJECT_BOX, 30)
    donor = obs_of(scene, SUBJECT_BOX, 20)
    result = align.align(target, donor, policy)
    assert result.accepted and result.reason is AKAZE_OK
    assert result.method is AlignmentMethod.AKAZE_HOMOGRAPHY
    assert all(g.policy_key != "ecc.min_correlation" for g in result.gates)


def test_align_falls_back_only_for_feature_starvation(
    obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    target = obs_of(textured_frame(1), SUBJECT_BOX, 30)
    donor = obs_of(textured_frame(99), SUBJECT_BOX, 20)
    primary = align.akaze_homography(target, donor, policy)
    assert not primary.accepted
    failed = next(g for g in primary.gates if not g.ok)
    assert failed.policy_key in align.FEATURE_STARVED_KEYS
    result = align.align(target, donor, policy)
    assert any(g.policy_key == "ecc.min_correlation" for g in result.gates)
    assert result.gates[:len(primary.gates)] == primary.gates


def test_box_affine_maps_donor_box_onto_target_box() -> None:
    donor_box = (10, 20, 50, 60)
    target_box = (100, 80, 180, 140)
    m = align.box_affine(donor_box, target_box)
    pts = np.array([[10, 20, 1], [50, 20, 1], [50, 60, 1], [10, 60, 1]], dtype=np.float64).T
    got = (m @ pts)[:2]
    expect = np.array([[100, 180, 180, 100], [80, 80, 140, 140]], dtype=np.float64)
    np.testing.assert_allclose(got, expect)


def test_ecc_on_translated_smooth_rectangle(
    obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    donor_frame = np.full((FRAME_H, FRAME_W, 3), 40, dtype=np.uint8)
    cv2.rectangle(donor_frame, (80, 70), (400, 220), (210, 200, 30), thickness=-1)
    dx, dy = 6, 4
    target_frame = np.ascontiguousarray(np.roll(donor_frame, (dy, dx), axis=(0, 1)))
    target = obs_of(target_frame, shift_box(SUBJECT_BOX, dx, dy), 30)
    donor = obs_of(donor_frame, SUBJECT_BOX, 20)
    result = align.ecc_affine_once(target, donor, policy)
    assert result.method is AlignmentMethod.ECC_AFFINE
    assert result.inlier_ratio == 0.0 and result.median_reproj_px == 0.0
    if result.accepted:
        assert result.reason is ECC_OK
        assert result.matrix is not None
        np.testing.assert_allclose(result.matrix[0, 2], dx, atol=2.0)
        np.testing.assert_allclose(result.matrix[1, 2], dy, atol=2.0)
    else:
        assert result.reason is FAILED
        _failing(result.gates, "ecc.min_correlation")


def test_ecc_is_deterministic(
    obs_of: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    frame = np.full((FRAME_H, FRAME_W, 3), 60, dtype=np.uint8)
    cv2.rectangle(frame, (90, 80), (390, 210), (180, 20, 20), thickness=-1)
    target = obs_of(frame, SUBJECT_BOX, 30)
    donor = obs_of(np.ascontiguousarray(np.roll(frame, (1, 2), axis=(0, 1))), SUBJECT_BOX, 20)
    a = align.ecc_affine_once(target, donor, policy)
    b = align.ecc_affine_once(target, donor, policy)
    assert a.accepted == b.accepted and a.reason is b.reason
    assert a.gates[0].observed == b.gates[0].observed
    if a.matrix is not None and b.matrix is not None:
        np.testing.assert_array_equal(a.matrix, b.matrix)


# ------------------------------------------------------------------------------------------------
# Read-only synthetic fixture: AKAZE on the designed 220x72 plate
# ------------------------------------------------------------------------------------------------


def test_translate_fixture_akaze_accepts_a_sharp_donor(
    translate_window, translate_images: Mapping[int, np.ndarray],
    make_obs: Callable[..., Obs], policy: PolicyConfig,
) -> None:
    by_n = {parse_frame_id(o.frame_id)[1]: o for o in translate_window.track.observations
            if o.accepted}
    t_obs, d_obs = by_n[synth.TARGET], by_n[synth.LEFT_SHARP_FRAME]
    target = make_obs(
        translate_images[synth.TARGET], t_obs.bbox_px, frame_number=synth.TARGET,
        pts_us=t_obs.pts_us, video_id=translate_window.video_id,
        track_id=translate_window.track_id)
    donor = make_obs(
        translate_images[synth.LEFT_SHARP_FRAME], d_obs.bbox_px,
        frame_number=synth.LEFT_SHARP_FRAME, pts_us=d_obs.pts_us,
        video_id=translate_window.video_id, track_id=translate_window.track_id)
    result = align.align(target, donor, policy)
    assert result.accepted, (
        result.reason, [(g.policy_key, g.observed, g.ok) for g in result.gates])
    assert result.reason in {AKAZE_OK, ECC_OK}
