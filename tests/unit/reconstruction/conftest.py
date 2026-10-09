"""Shared, READ-ONLY test support for Person 2 lanes. FROZEN: lanes may not edit this file.

Every array handed out is write-locked (``flags.writeable = False``); copy before mutating.
Nothing here writes to the repository. Inputs are the committed synthetic fixtures, the committed
policy, and seeded generators only (no network, no model downloads).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from probity.domain.models import (
    AssetRef,
    PixelProvenance,
    PolicyDecision,
    ReconstructionRun,
)
from probity.domain.policy import PolicyConfig, load_policy
from probity.eval.window import WindowBundle, load_window
from probity.reconstruction import quality as q
from probity.reconstruction.io import load_frame, read_json, read_png
from probity.reconstruction.provenance import decode_provenance
from probity.reconstruction.types import BBox, Obs, ProvenanceArrays

REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = REPO_ROOT / "config" / "policy.demo.yaml"
SYNTH_ROOT = REPO_ROOT / "fixtures" / "synthetic"
TRANSLATE_ROOT = SYNTH_ROOT / "plate_translate_v1"
SINGLE_DONOR_ROOT = SYNTH_ROOT / "plate_single_donor_v1"
COMPLETED_DIR = TRANSLATE_ROOT / "reconstructions" / "completed"
REFUSED_DIR = SINGLE_DONOR_ROOT / "reconstructions" / "refused"
SEED = 20261009


def frozen_array(arr: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(arr)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class GoldenRun:
    root: Path
    run: ReconstructionRun
    decisions: tuple[PolicyDecision, ...]
    provenance: PixelProvenance | None
    arrays: ProvenanceArrays | None
    npz_bytes: bytes | None
    result: np.ndarray | None
    assets: tuple[AssetRef, ...]


def load_golden(root: Path) -> GoldenRun:
    run = ReconstructionRun.model_validate(read_json(root / "run.json"))
    decisions = tuple(PolicyDecision.model_validate(d) for d in read_json(root / "decisions.json"))
    provenance = arrays = npz = result = None
    assets: tuple[AssetRef, ...] = ()
    if (root / "provenance.json").is_file():
        provenance = PixelProvenance.model_validate(read_json(root / "provenance.json"))
        npz = (root / "provenance.npz").read_bytes()
        raw = decode_provenance(npz)
        arrays = ProvenanceArrays(
            cls=frozen_array(raw.cls), source_index=frozen_array(raw.source_index),
            source_x=frozen_array(raw.source_x), source_y=frozen_array(raw.source_y),
            lut=provenance.source_lut)
        result = frozen_array(read_png(root / "result.png"))
        assets = tuple(AssetRef.model_validate(a) for a in read_json(root / "assets.json"))
    return GoldenRun(root, run, decisions, provenance, arrays, npz, result, assets)


def window_images(window: WindowBundle) -> dict[int, np.ndarray]:
    """frame_number -> write-locked BGR uint8 frame, decoded and pixel-hash verified."""
    return {ref.frame_number: frozen_array(load_frame(ref, window.resolver))
            for ref in window.frames}


def build_obs(frame: np.ndarray, box: BBox, *, frame_number: int, pts_us: int, video_id: str,
              track_id: str, cfg: PolicyConfig, confidence: float = 0.9,
              occluded_fraction: float = 0.0, detector_backed: bool = True,
              bridged: bool = False) -> Obs:
    """Hand-build an ``Obs`` from a full BGR frame and a full-frame half-open box."""
    h, w = frame.shape[:2]
    expanded = q.expand_box(box, cfg.crop.alignment_context_expand, w, h)
    return Obs(
        frame_number=frame_number, pts_us=pts_us, video_id=video_id, track_id=track_id,
        bbox_px=box, detector_backed=detector_backed, bridged=bridged, confidence=confidence,
        occluded_fraction=occluded_fraction, crop=frozen_array(q.crop(frame, box).copy()),
        expanded_crop=frozen_array(q.crop(frame, expanded).copy()), expanded_box=expanded,
        frame_size=(w, h))


# ------------------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def policy() -> PolicyConfig:
    """The committed demo policy (config/policy.demo.yaml). PolicyConfig is frozen."""
    return load_policy(POLICY_PATH)


@pytest.fixture(scope="session")
def translate_window() -> WindowBundle:
    return load_window(TRANSLATE_ROOT)


@pytest.fixture(scope="session")
def single_donor_window() -> WindowBundle:
    return load_window(SINGLE_DONOR_ROOT)


@pytest.fixture(scope="session")
def translate_images(translate_window: WindowBundle) -> Mapping[int, np.ndarray]:
    return window_images(translate_window)


@pytest.fixture(scope="session")
def single_donor_images(single_donor_window: WindowBundle) -> Mapping[int, np.ndarray]:
    return window_images(single_donor_window)


@pytest.fixture(scope="session")
def golden_completed() -> GoldenRun:
    return load_golden(COMPLETED_DIR)


@pytest.fixture(scope="session")
def golden_refused() -> GoldenRun:
    return load_golden(REFUSED_DIR)


@pytest.fixture
def rng() -> np.random.Generator:
    """Fresh deterministic generator per test."""
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def make_obs(policy: PolicyConfig) -> Callable[..., Obs]:
    """``make_obs(frame, box, frame_number=..., pts_us=..., video_id=..., track_id=..., **kw)``."""

    def _make(frame: np.ndarray, box: BBox, **kwargs: object) -> Obs:
        return build_obs(frame, box, cfg=policy, **kwargs)  # type: ignore[arg-type]

    return _make
