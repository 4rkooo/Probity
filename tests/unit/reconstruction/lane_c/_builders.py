"""Hand-built lane C inputs: ``Obs`` and ``AlignedDonor`` objects constructed directly.

No other lane's implementation is called; alignment and color fits are literal dataclasses with the
values a test needs. Images are tiny synthetic BGR uint8 arrays built from seeded patterns.
"""

from __future__ import annotations

import numpy as np

from probity.domain.enums import AlignmentMethod, ReasonCode
from probity.domain.policy import PolicyConfig
from probity.reconstruction import fuse
from probity.reconstruction import quality as q
from probity.reconstruction.types import (
    AlignedDonor,
    Alignment,
    BBox,
    ColorFit,
    Obs,
    QualityScores,
)

VIDEO = "01a1219b-7a80-7bd1-b62c-670da892173e"
TRACK = "01a1219b-7a80-7bd1-b62c-670da8921730"
FRAME_W, FRAME_H = 96, 64
SUBJECT: BBox = (16, 16, 48, 40)  # 32 x 24 px -> 4 x 3 tiles of 8 x 8
TARGET_FRAME = 30
FPS_US = 66_667
FLAT = 128


def quality(Q: float) -> QualityScores:
    return QualityScores(D=0.9, S=0.5, Z=0.5, E=1.0, O=1.0, P=1.0, Q=Q)


def alignment(A: float, *, accepted: bool = True) -> Alignment:
    return Alignment(
        method=AlignmentMethod.AKAZE_HOMOGRAPHY,
        matrix=np.eye(3),
        inlier_ratio=1.0,
        median_reproj_px=0.0,
        valid_coverage=1.0,
        corner_outside_fraction=0.0,
        A=A,
        accepted=accepted,
        reason=ReasonCode.ALIGNMENT_AKAZE_ACCEPTED if accepted else ReasonCode.ALIGNMENT_FAILED,
    )


def color(*, accepted: bool = True) -> ColorFit:
    return ColorFit(
        gain=(1.0, 1.0, 1.0),
        bias=(0.0, 0.0, 0.0),
        mean_abs_residual=0.0,
        samples=500,
        accepted=accepted,
        reason=ReasonCode.PHOTOMETRIC_INCOMPATIBLE,
    )


def flat_frame(value: int = FLAT) -> np.ndarray:
    return np.full((FRAME_H, FRAME_W, 3), value, dtype=np.uint8)


def checker(shape: tuple[int, int], amplitude: int, base: int = FLAT, phase: int = 0) -> np.ndarray:
    """1-px checkerboard ``base +/- amplitude`` (every pixel differs from ``base`` by exactly
    ``amplitude``), BGR uint8."""
    h, w = shape
    yy, xx = np.indices((h, w))
    sign = np.where((yy + xx + phase) % 2 == 0, 1, -1)
    gray = (base + sign * amplitude).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def make_obs(
    frame: np.ndarray,
    frame_number: int,
    cfg: PolicyConfig,
    *,
    box: BBox = SUBJECT,
    video_id: str = VIDEO,
    track_id: str = TRACK,
    detector_backed: bool = True,
    bridged: bool = False,
) -> Obs:
    h, w = frame.shape[:2]
    expanded = q.expand_box(box, cfg.crop.alignment_context_expand, w, h)
    return Obs(
        frame_number=frame_number,
        pts_us=frame_number * FPS_US,
        video_id=video_id,
        track_id=track_id,
        bbox_px=box,
        detector_backed=detector_backed,
        bridged=bridged,
        confidence=0.9,
        occluded_fraction=0.0,
        crop=q.crop(frame, box).copy(),
        expanded_crop=q.crop(frame, expanded).copy(),
        expanded_box=expanded,
        frame_size=(w, h),
    )


def make_target(cfg: PolicyConfig, frame: np.ndarray | None = None) -> tuple[np.ndarray, Obs]:
    img = flat_frame() if frame is None else frame
    return img, make_obs(img, TARGET_FRAME, cfg)


def crop_shape(target: Obs) -> tuple[int, int]:
    ex1, ey1, ex2, ey2 = target.expanded_box
    return ey2 - ey1, ex2 - ex1


def subject_slice(target: Obs) -> tuple[slice, slice]:
    """Subject box in target-crop coordinates."""
    ex1, ey1, _, _ = target.expanded_box
    x1, y1, x2, y2 = target.bbox_px
    return slice(y1 - ey1, y2 - ey1), slice(x1 - ex1, x2 - ex1)


def make_donor(
    target: Obs,
    frame_number: int,
    cfg: PolicyConfig,
    *,
    Q: float = 0.7,
    A: float = 0.9,
    T: float | None = None,
    warped: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    video_id: str = VIDEO,
    track_id: str = TRACK,
    detector_backed: bool = True,
    bridged: bool = False,
    aligned: bool = True,
    colored: bool = True,
    R: float | None = None,
) -> AlignedDonor:
    """A donor whose warped crop is ``warped`` (default: a 1-px checker of amplitude 10) in the
    target-crop frame; ``T`` defaults to the policy temporal preference for its frame offset."""
    shape = crop_shape(target)
    crop = checker(shape, 10) if warped is None else warped
    mask = np.ones(shape, dtype=bool) if valid is None else valid
    frame = flat_frame()
    obs = make_obs(
        frame,
        frame_number,
        cfg,
        box=target.bbox_px,
        video_id=video_id,
        track_id=track_id,
        detector_backed=detector_backed,
        bridged=bridged,
    )
    if T is None:
        T = fuse.temporal_preference((frame_number - target.frame_number) * FPS_US / 1e6, cfg)
    qs = quality(Q)
    if R is None:
        R = fuse.rank_score(qs, A, T)
    ys, xs = np.indices(shape, dtype=np.float32)
    ex1, ey1, _, _ = target.expanded_box
    return AlignedDonor(
        obs=obs,
        quality=qs,
        alignment=alignment(A, accepted=aligned),
        color=color(accepted=colored),
        warped_crop=np.ascontiguousarray(crop),
        valid_mask=mask,
        source_x=xs + np.float32(ex1),
        source_y=ys + np.float32(ey1),
        T=T,
        R=R,
    )
