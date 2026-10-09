"""Lane E: NON-EVIDENTIARY conventional upscaler baseline (section 12). SIGNATURES FROZEN.

The baseline (Real-ESRGAN x4plus in live mode) is for side-by-side evaluation only. Its output is
written only under ``derived/{video_id}/baseline/{run_id}/`` as an ``AssetRef`` of kind
``BASELINE`` with ``non_evidentiary=True``; ``io.assert_evidentiary_input`` rejects it, so it can
never become a donor, tracker input, or ``FrameReference``. Nothing in the reconstruction path
imports this module. Tests use a fake ``Upscaler``; no model download, no network.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from probity.domain.models import AssetRef
from probity.ports import DerivedStore
from probity.reconstruction.types import Obs

LANE = "lane E"
BASELINE_KIND = "baseline"
BASELINE_FILENAME = "baseline.png"
BASELINE_MODEL_ID = "real-esrgan-x4plus"


class Upscaler(Protocol):
    model_id: str
    model_sha256: str | None
    license_note: str

    def upscale(self, bgr: np.ndarray) -> np.ndarray: ...


def baseline_uri(video_id: str, run_id: str, filename: str = BASELINE_FILENAME) -> str:
    """``derived/{video_id}/baseline/{run_id}/{filename}``."""
    raise NotImplementedError(LANE)


def run_baseline(upscaler: Upscaler, target: Obs) -> np.ndarray:
    """Upscale ``target.crop`` only; returns BGR uint8. Never reads donors or results."""
    raise NotImplementedError(LANE)


def write_baseline(image: np.ndarray, store: DerivedStore, *, case_id: str, video_id: str,
                   run_id: str, upscaler: Upscaler, created_at: str) -> AssetRef:
    """Atomically write a lossless PNG via ``store.derived_path(video_id, "baseline", run_id, ...)``
    and return its ``AssetRef`` (kind BASELINE, non_evidentiary=True)."""
    raise NotImplementedError(LANE)
