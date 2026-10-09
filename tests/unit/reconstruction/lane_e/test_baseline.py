"""Lane E: non-evidentiary Real-ESRGAN baseline isolation (section 10, 12, 14 case 5)."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from probity.domain.enums import AssetKind, InferenceMode, PolicyOutcome, ReasonCode, SubjectType
from probity.domain.ids import frame_id
from probity.domain.models import AssetRef, FrameReference
from probity.domain.policy import PolicyConfig
from probity.eval.window import WindowBundle
from probity.ports import TrackRequest
from probity.reconstruction.baseline import (
    BASELINE_FILENAME,
    BASELINE_KIND,
    BASELINE_MODEL_ID,
    baseline_uri,
    run_baseline,
    write_baseline,
)
from probity.reconstruction.decisions import DecisionLog
from probity.reconstruction.determinism import FixedClock, seeded_uuid7
from probity.reconstruction.io import (
    NonEvidentiaryInput,
    assert_evidentiary_input,
    assert_evidentiary_uri,
    frame_png_uri,
    pixel_sha256,
    read_json,
    read_png,
)
from probity.reconstruction.preflight import run_preflight
from probity.reconstruction.tracking import TrackMeta, confirm_track
from probity.reconstruction.types import Obs

REPO = Path(__file__).resolve().parents[4]
VID = seeded_uuid7("lane-e:video", 1_760_000_000_000)
CASE = seeded_uuid7("lane-e:case", 1_760_000_000_000)
RUN = "run1"
CREATED = "2026-10-09T17:00:00.000000Z"
CLOCK = FixedClock(CREATED)


@dataclass
class FakeUpscaler:
    """Deterministic stand-in: ``cv2.resize`` only. No weights, no network."""

    model_id: str = BASELINE_MODEL_ID
    model_sha256: str | None = "ab" * 32
    license_note: str = "test fake; BSD-3-Clause; no Real-ESRGAN weights downloaded"
    scale: int = 4
    seen: list[tuple[int, ...]] = field(default_factory=list)

    def upscale(self, bgr: np.ndarray) -> np.ndarray:
        self.seen.append(tuple(bgr.shape))
        h, w = bgr.shape[:2]
        return cv2.resize(bgr, (w * self.scale, h * self.scale), interpolation=cv2.INTER_NEAREST)


class TmpDerivedStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def derived_path(self, video_id: str, kind: str, entity_id: str, filename: str) -> str:
        return str(self.root / "derived" / video_id / kind / entity_id / filename)


def _asset(*, kind: AssetKind = AssetKind.BASELINE, non_evidentiary: bool = True,
           video_id: str = VID, run_id: str = RUN) -> AssetRef:
    return AssetRef(
        asset_id=seeded_uuid7(f"lane-e:asset:{kind}", 1_760_000_000_000), case_id=CASE, kind=kind,
        storage_uri=f"derived/{video_id}/baseline/{run_id}/{BASELINE_FILENAME}",
        media_type="image/png", byte_length=1, sha256="0" * 64, non_evidentiary=non_evidentiary,
        created_at=CREATED,
    )


def _crop() -> np.ndarray:
    rng = np.random.default_rng(20261009)
    return rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)


def _obs(make_obs: Callable[..., Obs], crop: np.ndarray | None = None) -> Obs:
    frame = np.zeros((40, 48, 3), dtype=np.uint8)
    if crop is None:
        crop = _crop()
    frame[4:16, 8:24] = crop
    return make_obs(frame, (8, 4, 24, 16), frame_number=10, pts_us=0, video_id=VID, track_id=CASE)


def _recon_modules() -> list[Path]:
    return sorted(
        p for p in (REPO / "src" / "probity" / "reconstruction").glob("*.py")
        if p.name != "baseline.py"
    )


# ------------------------------------------------------------------------------------------------
# URI layout
# ------------------------------------------------------------------------------------------------


def test_baseline_uri_is_derived_video_baseline_run() -> None:
    uri = baseline_uri(VID, RUN)
    assert uri == f"derived/{VID}/{BASELINE_KIND}/{RUN}/{BASELINE_FILENAME}"
    parts = uri.split("/")
    assert parts[0] == "derived" and parts[2] == "baseline" and "source" not in parts


@pytest.mark.parametrize("run_id", [RUN, "r", "run_1.0"])
def test_baseline_uri_accepts_safe_run_ids(run_id: str) -> None:
    assert baseline_uri(VID, run_id).endswith(f"/{run_id}/{BASELINE_FILENAME}")


@pytest.mark.parametrize("run_id", ["..", "../frames", "foo/bar", ".hidden", "", "r id"])
def test_baseline_uri_rejects_unsafe_run_ids(run_id: str) -> None:
    with pytest.raises(ValueError):
        baseline_uri(VID, run_id)


def test_baseline_uri_requires_uuid7_video_id() -> None:
    with pytest.raises(ValueError):
        baseline_uri("not-a-uuid", RUN)


# ------------------------------------------------------------------------------------------------
# run_baseline uses the target crop only
# ------------------------------------------------------------------------------------------------


def test_run_baseline_upsamples_the_target_crop_only(make_obs: Callable[..., Obs]) -> None:
    crop = _crop()
    target = _obs(make_obs, crop)
    upscaler = FakeUpscaler()
    out = run_baseline(upscaler, target)
    assert upscaler.seen == [crop.shape]
    assert out.shape == (crop.shape[0] * 4, crop.shape[1] * 4, 3)
    assert out.dtype == np.uint8
    expected = cv2.resize(crop, (crop.shape[1] * 4, crop.shape[0] * 4),
                          interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(out, expected)
    assert out.flags["C_CONTIGUOUS"]


def test_run_baseline_rejects_non_integer_scale(make_obs: Callable[..., Obs]) -> None:
    class Jagged:
        model_id = BASELINE_MODEL_ID
        model_sha256 = "ab" * 32
        license_note = "test"

        def upscale(self, bgr: np.ndarray) -> np.ndarray:
            return cv2.resize(bgr, (bgr.shape[1] * 2, bgr.shape[0] * 3),
                              interpolation=cv2.INTER_NEAREST)

    with pytest.raises(ValueError, match="integer scale"):
        run_baseline(Jagged(), _obs(make_obs))


# ------------------------------------------------------------------------------------------------
# write_baseline: BASELINE + non_evidentiary under derived/.../baseline/
# ------------------------------------------------------------------------------------------------


def test_write_baseline_records_model_hash_and_license(tmp_path: Path) -> None:
    image = cv2.resize(_crop(), (64, 48), interpolation=cv2.INTER_NEAREST)
    upscaler = FakeUpscaler()
    store = TmpDerivedStore(tmp_path)
    asset = write_baseline(image, store, case_id=CASE, video_id=VID, run_id=RUN,
                           upscaler=upscaler, created_at=CREATED)
    assert asset.kind is AssetKind.BASELINE
    assert asset.non_evidentiary is True
    assert asset.storage_uri == baseline_uri(VID, RUN)
    png = tmp_path / "derived" / VID / "baseline" / RUN / BASELINE_FILENAME
    meta_path = tmp_path / "derived" / VID / "baseline" / RUN / "baseline.json"
    assert png.is_file() and meta_path.is_file()
    assert np.array_equal(read_png(png), image)
    meta = read_json(meta_path)
    assert meta["non_evidentiary"] is True
    assert meta["label"] == "non-evidentiary baseline"
    assert meta["kind"] == "BASELINE"
    assert meta["model_id"] == BASELINE_MODEL_ID
    assert meta["model_sha256"] == upscaler.model_sha256
    assert meta["license_note"] == upscaler.license_note
    assert meta["pixel_sha256"] == pixel_sha256(image)
    assert meta["storage_uri"] == asset.storage_uri
    text = meta_path.read_bytes()
    assert b"\r\n" not in text and text.endswith(b"\n")


def test_write_baseline_is_deterministic(tmp_path: Path) -> None:
    image = np.full((8, 8, 3), 40, dtype=np.uint8)
    upscaler = FakeUpscaler()
    a = write_baseline(image, TmpDerivedStore(tmp_path / "a"), case_id=CASE, video_id=VID,
                       run_id=RUN, upscaler=upscaler, created_at=CREATED)
    b = write_baseline(image, TmpDerivedStore(tmp_path / "b"), case_id=CASE, video_id=VID,
                       run_id=RUN, upscaler=upscaler, created_at=CREATED)
    assert a.asset_id == b.asset_id and a.sha256 == b.sha256


def test_write_baseline_requires_recorded_license(tmp_path: Path) -> None:
    upscaler = FakeUpscaler(license_note="  ")
    with pytest.raises(ValueError, match="license"):
        write_baseline(np.zeros((4, 4, 3), np.uint8), TmpDerivedStore(tmp_path), case_id=CASE,
                       video_id=VID, run_id=RUN, upscaler=upscaler, created_at=CREATED)


# ------------------------------------------------------------------------------------------------
# Generated isolation (section 14 case 5): inside vs outside the frames/ path
# ------------------------------------------------------------------------------------------------


def test_loader_accepts_source_frame_uri_just_inside_the_frames_path() -> None:
    uri = frame_png_uri(VID, 1)
    assert uri.split("/")[2] == "frames"
    assert_evidentiary_uri(uri, VID)
    ref = FrameReference.create(
        frame_id=frame_id(VID, 1), video_id=VID, frame_number=1, pts_us=0, is_keyframe=True,
        width_px=16, height_px=12, lossless_png_uri=uri, pixel_sha256="0" * 64,
    )
    assert assert_evidentiary_input(ref) is ref


def test_loader_rejects_baseline_uri_just_outside_the_frames_path() -> None:
    uri = baseline_uri(VID, RUN)
    assert uri.split("/")[2] == "baseline"
    with pytest.raises(NonEvidentiaryInput) as exc:
        assert_evidentiary_uri(uri, VID)
    assert exc.value.reason_code is ReasonCode.GENERATED_INPUT_REJECTED


def test_loader_rejects_baseline_asset_as_frame_input() -> None:
    with pytest.raises(NonEvidentiaryInput) as exc:
        assert_evidentiary_input(_asset())
    assert exc.value.reason_code is ReasonCode.GENERATED_INPUT_REJECTED


def test_frame_reference_cannot_point_at_baseline() -> None:
    with pytest.raises(ValidationError):
        FrameReference.create(
            frame_id=frame_id(VID, 1), video_id=VID, frame_number=1, pts_us=0, is_keyframe=True,
            width_px=16, height_px=12, lossless_png_uri=baseline_uri(VID, RUN),
            pixel_sha256="0" * 64,
        )


def test_baseline_asset_must_be_non_evidentiary() -> None:
    with pytest.raises(ValidationError):
        _asset(non_evidentiary=False)


def test_tracker_rejects_baseline_as_frame_input(policy: PolicyConfig) -> None:
    req = TrackRequest(
        track_id=seeded_uuid7("lane-e:track", 1_760_000_000_000), case_id=CASE, video_id=VID,
        source_sha256="0" * 64, subject_type=SubjectType.LICENSE_PLATE,
        seed_frame_id=frame_id(VID, 0), seed_bbox_px=(0, 0, 48, 20),
    )
    meta = TrackMeta(tracker_version="test", detector_model_id=None, mode=InferenceMode.FIXTURE,
                     created_at=CREATED)
    with pytest.raises(NonEvidentiaryInput) as exc:
        confirm_track(req, [_asset()], [], load=lambda ref: np.zeros((1, 1, 3), np.uint8),
                      cfg=policy, meta=meta, bridger_factory=None)
    assert exc.value.reason_code is ReasonCode.GENERATED_INPUT_REJECTED


def test_reconstructor_refuses_baseline_as_target(translate_window: WindowBundle,
                                                  translate_images: dict[int, np.ndarray],
                                                  policy: PolicyConfig) -> None:
    win = translate_window
    frames: dict[str, object] = {r.frame_id: r for r in win.frames}
    frames[win.target_frame_id] = _asset(video_id=win.video_id)
    dets: dict[str, list] = {}
    for det in win.detections:
        dets.setdefault(det.frame_id, []).append(det)
    log = DecisionLog(seeded_uuid7("lane-e:preflight", 1_760_000_000_000), CLOCK)
    result = run_preflight(
        log, track=win.track, video_id=win.video_id, case_id=win.case_id,
        target_frame_id=win.target_frame_id, target_box=win.target_bbox_px,
        frames=frames,  # type: ignore[arg-type]
        load=lambda ref: translate_images[ref.frame_number], detections=dets, cfg=policy,
    )
    assert result.refusal is ReasonCode.GENERATED_INPUT_REJECTED
    assert log.rows[-1].rule_code is ReasonCode.GENERATED_INPUT_REJECTED
    assert log.rows[-1].outcome is PolicyOutcome.REJECT


def test_reconstruction_path_never_imports_baseline() -> None:
    for path in _recon_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
            else:
                continue
            assert all("baseline" not in name.split(".") for name in names), path.name
