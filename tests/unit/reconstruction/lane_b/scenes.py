"""Lane B hand-built, seeded scenes and transforms (test inputs only)."""

from __future__ import annotations

import cv2
import numpy as np

from probity.reconstruction.types import BBox

FRAME_W, FRAME_H = 480, 288
SUBJECT_BOX: BBox = (130, 108, 350, 180)  # 220 x 72, the synthetic plate size
VIDEO_ID = "vid_lane_b"
TRACK_ID = "trk_lane_b"
SCENE_SEED = 20261009


def textured_frame(seed: int, width: int = FRAME_W, height: int = FRAME_H) -> np.ndarray:
    """Deterministic BGR uint8 scene with corners and edges (rectangles, strokes, soft noise)."""
    gen = np.random.default_rng(seed)
    noise = gen.integers(0, 256, size=(height // 8 + 1, width // 8 + 1, 3), dtype=np.uint8)
    img = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    for _ in range(90):
        x, y = int(gen.integers(0, width)), int(gen.integers(0, height))
        w, h = int(gen.integers(6, 40)), int(gen.integers(6, 30))
        color = tuple(int(c) for c in gen.integers(0, 256, size=3))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=-1)
    for _ in range(40):
        p1 = (int(gen.integers(0, width)), int(gen.integers(0, height)))
        p2 = (int(gen.integers(0, width)), int(gen.integers(0, height)))
        color = tuple(int(c) for c in gen.integers(0, 256, size=3))
        cv2.line(img, p1, p2, color, thickness=int(gen.integers(1, 4)))
    return np.ascontiguousarray(img)


def flat_frame(value: int, width: int = FRAME_W, height: int = FRAME_H) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def warp_perspective(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """``out(H @ p) = img(p)``: content moved by ``matrix`` (reflected borders, Lanczos-4)."""
    return cv2.warpPerspective(
        img, matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def shift_box(box: BBox, dx: int, dy: int) -> BBox:
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def scaling(s: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[s, 0.0, cx * (1 - s)], [0.0, s, cy * (1 - s)], [0.0, 0.0, 1.0]])
