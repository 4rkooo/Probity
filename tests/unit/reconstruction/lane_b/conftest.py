"""Lane B fixtures: seeded, hand-built scenes only (no other lane's implementation)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from probity.reconstruction.types import BBox, Obs

from .scenes import SCENE_SEED, TRACK_ID, VIDEO_ID, textured_frame


@pytest.fixture(scope="session")
def scene() -> np.ndarray:
    img = textured_frame(SCENE_SEED)
    img.setflags(write=False)
    return img


@pytest.fixture
def obs_of(make_obs: Callable[..., Obs]) -> Callable[..., Obs]:
    """``obs_of(frame, box, frame_number)`` -> Obs on the lane-B video/track."""

    def _obs(frame: np.ndarray, box: BBox, frame_number: int, **kw: object) -> Obs:
        return make_obs(
            frame, box, frame_number=frame_number, pts_us=frame_number * 66_667,
            video_id=VIDEO_ID, track_id=TRACK_ID, **kw,
        )

    return _obs
