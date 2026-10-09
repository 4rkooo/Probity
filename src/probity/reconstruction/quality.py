"""Section 9 observation-quality measures. Every threshold and normalizer comes from PolicyConfig.

Formula weights (Q) are part of ``probity-tile-v1`` and are not policy gates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from probity.domain.policy import PolicyConfig
from probity.reconstruction.types import QualityScores

BBox = tuple[int, int, int, int]

Q_WEIGHTS = {"D": 0.25, "S": 0.25, "Z": 0.15, "E": 0.15, "O": 0.10, "P": 0.10}
LUMA_FLOOR_8BIT = 1.0


def clip01(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("score component is not finite")
    return min(1.0, max(0.0, value))


# ------------------------------------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------------------------------------


def box_size(box: BBox) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return x2 - x1, y2 - y1


def box_aspect(box: BBox) -> float:
    w, h = box_size(box)
    return w / h


def expand_box(box: BBox, fraction: float, width: int, height: int) -> BBox:
    """Grow the box by ``fraction`` of its size (half per side), clipped to the frame."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = w * fraction / 2.0, h * fraction / 2.0
    return (
        max(0, math.floor(x1 - dx)),
        max(0, math.floor(y1 - dy)),
        min(width, math.ceil(x2 + dx)),
        min(height, math.ceil(y2 + dy)),
    )


def crop(img: np.ndarray, box: BBox) -> np.ndarray:
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]


def to_luma(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# ------------------------------------------------------------------------------------------------
# Components
# ------------------------------------------------------------------------------------------------


def resize_long_edge(gray: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = long_edge / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(gray, size, interpolation=interp)


def laplacian(gray: np.ndarray) -> np.ndarray:
    return cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F, ksize=1)


def laplacian_variance(gray: np.ndarray) -> float:
    return float(laplacian(gray).var())


def sharpness_score(subject_luma: np.ndarray, cfg: PolicyConfig) -> float:
    """S: Laplacian variance of the subject crop resized to a fixed long edge, normalized."""
    resized = resize_long_edge(subject_luma, cfg.quality.resize_long_edge_px)
    return clip01(laplacian_variance(resized) / cfg.quality.laplacian_variance_normalizer)


def size_score(box: BBox, cfg: PolicyConfig) -> float:
    w, h = box_size(box)
    return clip01(math.sqrt(w * h) / cfg.quality.subject_size_normalizer_px)


def clipped_fraction(subject_luma: np.ndarray, cfg: PolicyConfig) -> float:
    q = cfg.quality
    clipped = (subject_luma < q.black_luma_8bit) | (subject_luma > q.white_luma_8bit)
    return float(clipped.mean())


def exposure_score(subject_luma: np.ndarray, cfg: PolicyConfig) -> float:
    return 1.0 - clip01(clipped_fraction(subject_luma, cfg) / cfg.quality.max_clipped_fraction)


def occlusion_score(occluded_fraction: float) -> float:
    return 1.0 - clip01(occluded_fraction)


def pose_score(aspect: float, median_track_aspect: float, cfg: PolicyConfig) -> float:
    ratio = abs(math.log(aspect / median_track_aspect))
    return 1.0 - clip01(ratio / math.log(cfg.quality.pose_aspect_ratio_limit))


def observation_quality(d: float, s: float, z: float, e: float, o: float, p: float) -> float:
    w = Q_WEIGHTS
    return clip01(w["D"] * d + w["S"] * s + w["Z"] * z + w["E"] * e + w["O"] * o + w["P"] * p)


def exposure_delta_stops(subject_luma_a: np.ndarray, subject_luma_b: np.ndarray) -> float:
    """|log2(mean luma ratio)| over the subject boxes, with a 1/255 floor against log(0)."""
    a = max(float(subject_luma_a.mean()), LUMA_FLOOR_8BIT)
    b = max(float(subject_luma_b.mean()), LUMA_FLOOR_8BIT)
    return abs(math.log2(a / b))


def measure_observation(
    frame_bgr: np.ndarray,
    box: BBox,
    confidence: float,
    occluded_fraction: float,
    median_track_aspect: float,
    cfg: PolicyConfig,
) -> QualityScores:
    luma = crop(to_luma(frame_bgr), box)
    d = clip01(confidence)
    s = sharpness_score(luma, cfg)
    z = size_score(box, cfg)
    e = exposure_score(luma, cfg)
    o = occlusion_score(occluded_fraction)
    p = pose_score(box_aspect(box), median_track_aspect, cfg)
    return QualityScores(D=d, S=s, Z=z, E=e, O=o, P=p, Q=observation_quality(d, s, z, e, o, p))


# ------------------------------------------------------------------------------------------------
# Tile sharpness (fusion)
# ------------------------------------------------------------------------------------------------


def tile_sharpness_scores(
    subject_luma: np.ndarray, rel_tiles: Sequence[BBox], cfg: PolicyConfig
) -> list[float]:
    """Per-tile normalized sharpness on the same resized grid and normalizer as ``S``.

    ``rel_tiles`` are tile boxes relative to the subject crop's top-left corner.
    """
    h, w = subject_luma.shape[:2]
    resized = resize_long_edge(subject_luma, cfg.quality.resize_long_edge_px)
    lap = laplacian(resized)
    fx, fy = resized.shape[1] / w, resized.shape[0] / h
    out: list[float] = []
    for x1, y1, x2, y2 in rel_tiles:
        rx1, ry1 = math.floor(x1 * fx), math.floor(y1 * fy)
        rx2, ry2 = max(rx1 + 1, math.ceil(x2 * fx)), max(ry1 + 1, math.ceil(y2 * fy))
        var = float(lap[ry1:ry2, rx1:rx2].var())
        out.append(clip01(var / cfg.quality.laplacian_variance_normalizer))
    return out


def tiles(box: BBox, tile_px: int) -> list[BBox]:
    """Row-major tiles of ``tile_px`` covering ``box``; edge tiles are clipped, never padded."""
    x1, y1, x2, y2 = box
    out: list[BBox] = []
    for ty in range(y1, y2, tile_px):
        for tx in range(x1, x2, tile_px):
            out.append((tx, ty, min(tx + tile_px, x2), min(ty + tile_px, y2)))
    return out
