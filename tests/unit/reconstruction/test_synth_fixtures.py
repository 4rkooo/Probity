"""Committed synthetic windows: integrity, determinism, and the design properties goldens need."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np
import pytest

from probity.domain.enums import ObservationSource
from probity.domain.ids import parse_frame_id
from probity.domain.policy import default_policy
from probity.eval import synth
from probity.eval.window import WindowBundle, load_truth, load_window
from probity.reconstruction import quality as q
from probity.reconstruction.io import load_frame, pixel_sha256, read_png

REPO = Path(__file__).resolve().parents[3]
SYNTH_ROOT = REPO / "fixtures" / "synthetic"
CFG = default_policy()
SPECS = {s.fixture_id: s for s in synth.default_windows()}


@pytest.fixture(scope="module", params=sorted(SPECS))
def window(request: pytest.FixtureRequest) -> WindowBundle:
    return load_window(SYNTH_ROOT / request.param)


@pytest.fixture(scope="module")
def translate() -> WindowBundle:
    return load_window(SYNTH_ROOT / "plate_translate_v1")


def _same_opencv(root: Path) -> bool:
    return bool(load_truth(root)["opencv_version"] == cv2.__version__)


def test_window_records_validate_and_cohere(window: WindowBundle) -> None:
    sc = SPECS[window.fixture_id].scene
    assert len(window.frames) == synth.N_FRAMES == len(window.manifest.frames)
    assert window.manifest.source_sha256 == window.source_sha256
    assert window.manifest.time_base == f"1/{sc.fps}"
    for n, ref in enumerate(window.frames):
        assert ref.frame_number == n and ref.pts_us == n * 1_000_000 // sc.fps
        assert ref.video_id == window.video_id
    assert window.frames[-1].pts_us >= 4_900_000
    assert all(d.video_id == window.video_id for d in window.detections)
    assert window.config_sha256 == CFG.config_sha256


def test_committed_pngs_match_recorded_pixel_hashes(window: WindowBundle) -> None:
    hashes = [pixel_sha256(load_frame(ref, window.resolver)) for ref in window.frames]
    assert synth.source_digest(hashes) == window.source_sha256


def test_ground_truth_is_isolated_from_frame_inputs(window: WindowBundle) -> None:
    for ref in window.frames:
        assert ref.lossless_png_uri is not None and "ground_truth" not in ref.lossless_png_uri
    assert load_truth(window.root)["evaluation_only"] is True


def test_rendering_is_deterministic_in_process() -> None:
    spec = SPECS["plate_translate_v1"]
    first = synth.render_window(spec)
    second = synth.render_window(spec)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a.image, b.image)


def test_regeneration_reproduces_committed_window(window: WindowBundle, tmp_path: Path) -> None:
    if not _same_opencv(window.root):
        pytest.skip(f"fixtures rendered with a different OpenCV build than {cv2.__version__}")
    out = synth.write_window(SPECS[window.fixture_id], tmp_path, CFG)
    for name in ("frames.json", "frame_manifest.json", "detections.json", "track.json",
                 "window.json", "ground_truth/truth.json"):
        assert (out / name).read_bytes() == (window.root / name).read_bytes(), name


# ------------------------------------------------------------------------------------------------
# Fixture design properties (the goldens are only meaningful if these hold)
# ------------------------------------------------------------------------------------------------


def _measure(window: WindowBundle) -> dict[int, dict[str, float]]:
    track = window.track
    dets = {d.detection_id: d for d in window.detections}
    target_n = parse_frame_id(window.target_frame_id)[1]
    accepted = [o for o in track.observations if o.accepted]
    median_aspect = statistics.median(q.box_aspect(o.bbox_px) for o in accepted)
    target_img = load_frame(window.frame(target_n), window.resolver)
    target_luma = q.crop(q.to_luma(target_img), window.target_bbox_px)
    out: dict[int, dict[str, float]] = {}
    for obs in accepted:
        n = parse_frame_id(obs.frame_id)[1]
        img = load_frame(window.frame(n), window.resolver)
        occ = (dets[obs.detection_id].occlusion_score or 0.0) if obs.detection_id else 0.0
        qc = q.measure_observation(img, obs.bbox_px, obs.confidence, occ, median_aspect, CFG)
        out[n] = {
            "S": qc.S, "Q": qc.Q, "occluded": 1.0 - qc.O,
            "stops": q.exposure_delta_stops(q.crop(q.to_luma(img), obs.bbox_px), target_luma),
            "detector": float(obs.source is ObservationSource.DETECTOR),
        }
    return out


def _survivors(window: WindowBundle) -> list[int]:
    m = _measure(window)
    target_n = parse_frame_id(window.target_frame_id)[1]
    tq = m[target_n]["Q"]
    d = CFG.donor
    return sorted(
        n for n, v in m.items()
        if n != target_n and v["detector"] == 1.0 and v["occluded"] <= d.max_occluded_fraction
        and v["Q"] >= d.min_quality and v["Q"] - tq >= d.min_quality_gain
        and v["stops"] <= d.max_exposure_delta_stops
    )


def test_target_is_the_blurriest_unaltered_frame(window: WindowBundle) -> None:
    m = _measure(window)
    target_n = parse_frame_id(window.target_frame_id)[1]
    others = [v["S"] for n, v in m.items() if n not in (target_n, synth.INCOMPATIBLE_FRAME)]
    assert m[target_n]["S"] < min(others)
    sigma = [max(f.sigma_left, f.sigma_right) for f in SPECS[window.fixture_id].frames]
    assert sigma[target_n] == max(sigma)


def test_target_meets_size_and_quality_floors(window: WindowBundle) -> None:
    w, h = q.box_size(window.target_bbox_px)
    assert w >= CFG.target.min_width_px and h >= CFG.target.min_height_px
    target_n = parse_frame_id(window.target_frame_id)[1]
    assert _measure(window)[target_n]["Q"] >= CFG.target.min_quality
    assert h >= 72, "AKAZE needs roughly 70+ px plate height to find keypoints"


def test_track_is_confirmed_by_detector_observations_only(window: WindowBundle) -> None:
    t = window.track
    detector = [o for o in t.observations if o.source is ObservationSource.DETECTOR and o.accepted]
    assert t.confirmed and t.detector_observation_count == len(detector)
    assert len(detector) >= CFG.track.min_detector_observations
    assert statistics.fmean(o.confidence for o in detector) >= CFG.track.min_mean_confidence
    bridged = [parse_frame_id(o.frame_id)[1] for o in t.observations
               if o.source is ObservationSource.CSRT_BRIDGE]
    assert bridged == list(synth.GAP_FRAMES)


def test_obstructed_frame_is_occluded_beyond_limit(window: WindowBundle) -> None:
    n = synth.OBSTRUCTED_FRAME
    truth = load_truth(window.root)["frames"][n]
    assert truth["occluded_fraction"] > CFG.donor.max_occluded_fraction
    mask = read_png(window.root / "ground_truth" / f"occlusion_mask_f{n:04d}.png")
    x1, y1, x2, y2 = synth.detection_box(SPECS[window.fixture_id].scene, n)
    measured = mask[y1:y2, x1:x2, 0].mean() / 255.0
    assert measured == pytest.approx(truth["occluded_fraction"], abs=1e-3)
    assert _measure(window)[n]["occluded"] > CFG.donor.max_occluded_fraction


def test_incompatible_frame_exceeds_exposure_limit(window: WindowBundle) -> None:
    stops = _measure(window)[synth.INCOMPATIBLE_FRAME]["stops"]
    assert stops > CFG.donor.max_exposure_delta_stops
    assert stops == pytest.approx(abs(math.log2(0.45)), abs=0.1)


def test_translate_window_has_at_least_three_sharper_donors(translate: WindowBundle) -> None:
    survivors = _survivors(translate)
    assert len(survivors) >= 3
    assert {synth.LEFT_SHARP_FRAME, synth.RIGHT_SHARP_FRAME} <= set(survivors)
    assert synth.OBSTRUCTED_FRAME not in survivors and synth.INCOMPATIBLE_FRAME not in survivors


def test_single_donor_window_has_exactly_one_survivor() -> None:
    w = load_window(SYNTH_ROOT / "plate_single_donor_v1")
    assert _survivors(w) == [synth.SINGLE_DONOR_FRAME]


def test_fixture_footprint_is_small() -> None:
    total = sum(p.stat().st_size for p in SYNTH_ROOT.rglob("*") if p.is_file())
    assert total < 8_000_000


def test_truth_records_generator_versions(window: WindowBundle) -> None:
    truth = json.loads((window.root / "ground_truth" / "truth.json").read_text(encoding="utf-8"))
    assert truth["synth_version"] == synth.SYNTH_VERSION
    assert truth["plate_text"] == SPECS[window.fixture_id].scene.plate_text
