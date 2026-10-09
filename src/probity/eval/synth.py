"""Deterministic synthetic plate windows: lossless frames, detections, track, and ground truth.

Each window is a fixed camera watching one fronto-parallel synthetic plate on a car body that
translates slowly. Per-frame blur, exposure, and a transient occluder are set by a schedule so the
target frame is the blurriest and the expected donor outcomes are known in advance.

Plates are rendered with ``cv2.putText`` Hershey fonts (no system font files) at 4x and
downsampled with INTER_AREA. The plate text is evaluation ground truth only: it is written to
``ground_truth/`` and never reaches the reconstruction path.

Regenerate:  uv run python -m probity.eval.synth --out fixtures/synthetic
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from probity.domain.enums import InferenceMode, SubjectType
from probity.domain.ids import canonical_sha256, frame_id
from probity.domain.models import (
    Detection,
    FrameManifest,
    FrameManifestEntry,
    FrameReference,
    Track,
)
from probity.domain.policy import PolicyConfig, default_policy
from probity.ports import TrackRequest
from probity.reconstruction.determinism import FixedClock, seeded_uuid7, unit_float
from probity.reconstruction.io import (
    FixtureFrameResolver,
    frame_png_uri,
    load_frame,
    write_json,
    write_png,
)
from probity.reconstruction.tracking import (
    TRACK_LOGIC_VERSION,
    ReplayBridger,
    TrackedDetection,
    TrackMeta,
    confirm_track,
)

SYNTH_VERSION = "probity-synth-v1"
DETECTOR_MODEL_ID = "fixture/synthetic-gt-detector-v1"
TRACKER_VERSION = f"{TRACK_LOGIC_VERSION}+synthetic-replay"
SYNTH_TRACKER_ID = 1
RENDER_SCALE = 4
BASE_TIME = "2026-10-09T17:00:00.000000Z"

BGR = tuple[int, int, int]


@dataclass(frozen=True)
class FrameSpec:
    """Degradations for one frame. Blur sigma ramps linearly from the plate's left to right edge."""

    sigma_left: float
    sigma_right: float
    exposure_gain: float = 1.0
    occluder_fraction: float = 0.0
    detected: bool = True


@dataclass(frozen=True)
class SceneSpec:
    width: int = 480
    height: int = 288
    fps: int = 15
    plate_w: int = 220
    plate_h: int = 72
    plate_text: str = "PRB 4K7"
    start_x: float = 64.0
    start_y: float = 112.0
    velocity_x: float = 0.6
    bob_amp_px: float = 1.5
    bob_period_frames: int = 40
    plate_bg: BGR = (196, 200, 204)
    plate_ink: BGR = (70, 64, 60)
    car_body: BGR = (150, 86, 40)
    occluder: BGR = (96, 104, 110)


@dataclass(frozen=True)
class WindowSpec:
    fixture_id: str
    seed: str
    target_frame: int
    frames: tuple[FrameSpec, ...]
    scene: SceneSpec = field(default_factory=SceneSpec)
    description: str = ""

    @property
    def n_frames(self) -> int:
        return len(self.frames)


# ------------------------------------------------------------------------------------------------
# Schedules
# ------------------------------------------------------------------------------------------------

N_FRAMES = 75
TARGET = 37
TARGET_SIGMA = 2.5
OBSTRUCTED_FRAME = 34
INCOMPATIBLE_FRAME = 40
LEFT_SHARP_FRAME = 31
RIGHT_SHARP_FRAME = 43
MILD_FRAMES = (26, 48)
GAP_FRAMES = (52, 53)


def _background_sigma(seed: str, n: int, lo: float, hi: float) -> float:
    return round(lo + (hi - lo) * unit_float(f"{seed}:sigma:{n}"), 3)


def translate_schedule(seed: str, target_sigma: float = TARGET_SIGMA) -> tuple[FrameSpec, ...]:
    """Target blurriest; two half-sharp donors (left/right), two mildly sharper donors,
    one obstructed sharp frame, one under-exposed sharp frame, and a two-frame detector gap."""
    t = target_sigma
    out: list[FrameSpec] = []
    for n in range(N_FRAMES):
        if n == TARGET:
            spec = FrameSpec(t, t)
        elif n == LEFT_SHARP_FRAME:
            spec = FrameSpec(0.0, t)
        elif n == RIGHT_SHARP_FRAME:
            spec = FrameSpec(t, 0.0)
        elif n in MILD_FRAMES:
            spec = FrameSpec(round(0.45 * t, 3), round(0.45 * t, 3))
        elif n == OBSTRUCTED_FRAME:
            spec = FrameSpec(0.0, 0.0, occluder_fraction=0.30)
        elif n == INCOMPATIBLE_FRAME:
            spec = FrameSpec(0.0, 0.0, exposure_gain=0.45)
        else:
            s = _background_sigma(seed, n, 0.80 * t, 0.98 * t)
            spec = FrameSpec(s, s)
        if n in GAP_FRAMES:
            spec = FrameSpec(spec.sigma_left, spec.sigma_right, detected=False)
        out.append(spec)
    return tuple(out)


SINGLE_DONOR_FRAME = 43


def single_donor_schedule(seed: str, target_sigma: float = TARGET_SIGMA
                          ) -> tuple[FrameSpec, ...]:
    """Same scene; every other frame is about as blurry as the target, so one donor survives."""
    t = target_sigma
    out: list[FrameSpec] = []
    for n in range(N_FRAMES):
        if n == TARGET:
            spec = FrameSpec(t, t)
        elif n == SINGLE_DONOR_FRAME:
            spec = FrameSpec(0.0, 0.0)
        elif n == OBSTRUCTED_FRAME:
            spec = FrameSpec(0.0, 0.0, occluder_fraction=0.30)
        elif n == INCOMPATIBLE_FRAME:
            spec = FrameSpec(0.0, 0.0, exposure_gain=0.45)
        else:
            s = _background_sigma(seed, n, 0.80 * t, 0.95 * t)
            spec = FrameSpec(s, s)
        if n in GAP_FRAMES:
            spec = FrameSpec(spec.sigma_left, spec.sigma_right, detected=False)
        out.append(spec)
    return tuple(out)


def default_windows() -> tuple[WindowSpec, ...]:
    return (
        WindowSpec(
            fixture_id="plate_translate_v1",
            seed="probity-synth-plate-translate-v1",
            target_frame=TARGET,
            frames=translate_schedule("probity-synth-plate-translate-v1"),
            description="Fronto-parallel plate, slow translation; >=2 compatible donors.",
        ),
        WindowSpec(
            fixture_id="plate_single_donor_v1",
            seed="probity-synth-plate-single-donor-v1",
            target_frame=TARGET,
            frames=single_donor_schedule("probity-synth-plate-single-donor-v1"),
            description="Same scene; only one donor is clearer than the target.",
        ),
    )


# ------------------------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------------------------


def pts_us(n: int, fps: int) -> int:
    return (n * 1_000_000) // fps


def plate_origin(scene: SceneSpec, n: int) -> tuple[float, float]:
    """Top-left of the plate in output pixels, quantized to the 1/RENDER_SCALE render grid."""
    x = scene.start_x + scene.velocity_x * n
    y = scene.start_y + scene.bob_amp_px * math.sin(2.0 * math.pi * n / scene.bob_period_frames)
    return round(x * RENDER_SCALE) / RENDER_SCALE, round(y * RENDER_SCALE) / RENDER_SCALE


def gt_box(scene: SceneSpec, n: int) -> tuple[float, float, float, float]:
    x, y = plate_origin(scene, n)
    return (x, y, x + scene.plate_w, y + scene.plate_h)


def detection_box(scene: SceneSpec, n: int) -> tuple[int, int, int, int]:
    x, y = plate_origin(scene, n)
    x1, y1 = math.floor(x + 0.5), math.floor(y + 0.5)
    return (x1, y1, x1 + scene.plate_w, y1 + scene.plate_h)


def _render_background(scene: SceneSpec) -> np.ndarray:
    s = RENDER_SCALE
    w, h = scene.width * s, scene.height * s
    rows = np.linspace(0.0, 1.0, h, dtype=np.float64)[:, None]
    sky = np.array([200, 170, 140], dtype=np.float64)
    ground = np.array([110, 112, 108], dtype=np.float64)
    img = (sky * (1 - rows) + ground * rows)[:, None, :].repeat(w, axis=1)
    img = img.astype(np.uint8)
    road_y = int(scene.height * 0.62)
    cv2.rectangle(img, (0, road_y * s), (w, h), (84, 86, 88), thickness=-1)
    for x in range(10, scene.width, 60):
        y = int(scene.height * 0.93)
        cv2.rectangle(img, (x * s, y * s), ((x + 30) * s, (y + 4) * s), (200, 200, 196), -1)
    bx1, bx2 = int(scene.width * 0.80), int(scene.width * 0.97)
    cv2.rectangle(img, (bx1 * s, 12 * s), (bx2 * s, 100 * s), (120, 130, 150), -1)
    for gx in range(bx1 + 6, bx2 - 10, 16):
        for gy in range(20, 90, 18):
            cv2.rectangle(img, (gx * s, gy * s), ((gx + 9) * s, (gy + 11) * s), (70, 60, 50), -1)
    return img


def _render_plate(scene: SceneSpec) -> np.ndarray:
    s = RENDER_SCALE
    pw, ph = scene.plate_w * s, scene.plate_h * s
    plate = np.empty((ph, pw, 3), dtype=np.uint8)
    plate[:] = scene.plate_bg
    border = max(1, round(scene.plate_h / 24)) * s
    cv2.rectangle(plate, (border // 2, border // 2), (pw - border // 2, ph - border // 2),
                  scene.plate_ink, thickness=border)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, round(scene.plate_h / 14)) * s
    target_w = 0.84 * pw
    (tw, _), _ = cv2.getTextSize(scene.plate_text, font, 1.0, thickness)
    scale = target_w / tw
    (tw, th), _ = cv2.getTextSize(scene.plate_text, font, scale, thickness)
    org = ((pw - tw) // 2, (ph + th) // 2)
    cv2.putText(plate, scene.plate_text, org, font, scale, scene.plate_ink, thickness, cv2.LINE_AA)
    return plate


def _blur_ramp(img: np.ndarray, x_left: float, x_right: float, s_left: float, s_right: float
               ) -> np.ndarray:
    def blur(sigma: float) -> np.ndarray:
        if sigma <= 0.0:
            return img.astype(np.float64)
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REFLECT_101).astype(np.float64)

    left, right = blur(s_left), blur(s_right)
    if s_left == s_right:
        return left
    cols = np.arange(img.shape[1], dtype=np.float64)
    w = np.clip((cols - x_left) / (x_right - x_left), 0.0, 1.0)[None, :, None]
    return left * (1.0 - w) + right * w


@dataclass(frozen=True)
class RenderedFrame:
    image: np.ndarray
    clean: np.ndarray
    occlusion_mask: np.ndarray | None


def render_frame(scene: SceneSpec, spec: FrameSpec, n: int, background: np.ndarray,
                 plate: np.ndarray) -> RenderedFrame:
    s = RENDER_SCALE
    canvas = background.copy()
    x, y = plate_origin(scene, n)
    px, py = round(x * s), round(y * s)
    ph, pw = plate.shape[:2]
    mx, top, bottom = round(0.25 * pw), round(1.2 * ph), round(1.0 * ph)
    cv2.rectangle(canvas, (px - mx, py - top), (px + pw + mx, py + ph + bottom),
                  scene.car_body, thickness=-1)
    cv2.rectangle(canvas, (px - mx // 3, py - round(1.05 * ph)),
                  (px + pw + mx // 3, py - round(0.45 * ph)), (60, 40, 30), thickness=-1)
    canvas[py:py + ph, px:px + pw] = plate
    clean = cv2.resize(canvas, (scene.width, scene.height), interpolation=cv2.INTER_AREA)

    mask = None
    if spec.occluder_fraction > 0.0:
        ow = round(pw * spec.occluder_fraction)
        ox = px + (pw - ow) // 2
        cv2.rectangle(canvas, (ox, py - ph // 2), (ox + ow - 1, py + ph + ph // 2 - 1),
                      scene.occluder, thickness=-1)
        full = np.zeros(canvas.shape[:2], dtype=np.uint8)
        full[py:py + ph, ox:ox + ow] = 255
        mask = cv2.resize(full, (scene.width, scene.height), interpolation=cv2.INTER_AREA)

    out = cv2.resize(canvas, (scene.width, scene.height), interpolation=cv2.INTER_AREA)
    blurred = _blur_ramp(out, x, x + scene.plate_w, spec.sigma_left, spec.sigma_right)
    exposed = blurred * spec.exposure_gain
    image = np.clip(np.floor(exposed + 0.5), 0, 255).astype(np.uint8)
    return RenderedFrame(image=image, clean=clean, occlusion_mask=mask)


def render_window(spec: WindowSpec) -> list[RenderedFrame]:
    background = _render_background(spec.scene)
    plate = _render_plate(spec.scene)
    return [render_frame(spec.scene, fs, n, background, plate) for n, fs in enumerate(spec.frames)]


# ------------------------------------------------------------------------------------------------
# Records
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowIds:
    case_id: str
    video_id: str
    track_id: str

    @classmethod
    def for_window(cls, spec: WindowSpec, clock: FixedClock) -> WindowIds:
        ms = clock.unix_ms
        return cls(
            case_id=seeded_uuid7(f"{spec.seed}:case", ms),
            video_id=seeded_uuid7(f"{spec.seed}:video", ms),
            track_id=seeded_uuid7(f"{spec.seed}:track", ms + 1),
        )


def source_digest(pixel_hashes: Sequence[str]) -> str:
    """Synthetic windows have no MP4; their source identity is the ordered pixel-hash sequence."""
    return canonical_sha256({"synth_version": SYNTH_VERSION, "pixel_sha256": list(pixel_hashes)})


def detection_confidence(spec: WindowSpec, n: int) -> float:
    fs = spec.frames[n]
    sigma = max(fs.sigma_left, fs.sigma_right)
    jitter = 0.04 * unit_float(f"{spec.seed}:conf:{n}") - 0.02
    value = 0.86 - 0.10 * sigma - 0.40 * fs.occluder_fraction + jitter
    return round(min(0.95, max(0.05, value)), 4)


def build_frames(spec: WindowSpec, ids: WindowIds, hashes: Sequence[str], clock: FixedClock
                 ) -> list[FrameReference]:
    sc = spec.scene
    return [
        FrameReference.create(
            created_at=clock.at(0),
            frame_id=frame_id(ids.video_id, n),
            video_id=ids.video_id,
            frame_number=n,
            pts_us=pts_us(n, sc.fps),
            is_keyframe=True,
            width_px=sc.width,
            height_px=sc.height,
            lossless_png_uri=frame_png_uri(ids.video_id, n),
            pixel_sha256=hashes[n],
            decoder=SYNTH_VERSION,
            pixel_format="rgb24",
        )
        for n in range(spec.n_frames)
    ]


def build_manifest(spec: WindowSpec, ids: WindowIds, digest: str, clock: FixedClock
                   ) -> FrameManifest:
    sc = spec.scene
    return FrameManifest.create(
        created_at=clock.at(0),
        video_id=ids.video_id,
        source_sha256=digest,
        time_base=f"1/{sc.fps}",
        extractor=f"{SYNTH_VERSION} (lossless PNG sequence, no container)",
        frames=tuple(
            FrameManifestEntry(frame_number=n, pts_us=pts_us(n, sc.fps), pts=n, is_keyframe=True,
                               width_px=sc.width, height_px=sc.height)
            for n in range(spec.n_frames)
        ),
    )


def build_detections(spec: WindowSpec, ids: WindowIds, clock: FixedClock) -> list[Detection]:
    sc = spec.scene
    out: list[Detection] = []
    for n, fs in enumerate(spec.frames):
        if not fs.detected:
            continue
        x1, y1, x2, y2 = detection_box(sc, n)
        out.append(
            Detection.create(
                created_at=clock.at(1 + n),
                detection_id=seeded_uuid7(f"{spec.seed}:det:{n}", clock.unix_ms + 2),
                video_id=ids.video_id,
                frame_id=frame_id(ids.video_id, n),
                pts_us=pts_us(n, sc.fps),
                model_id=DETECTOR_MODEL_ID,
                model_sha256=None,
                class_id=0,
                class_name="license_plate",
                confidence=detection_confidence(spec, n),
                bbox_px=(x1, y1, x2, y2),
                bbox_norm=(x1 / sc.width, y1 / sc.height, x2 / sc.width, y2 / sc.height),
                occlusion_score=round(fs.occluder_fraction, 4) if fs.occluder_fraction else None,
                inference_mode=InferenceMode.FIXTURE,
            )
        )
    return out


def tracker_inputs(spec: WindowSpec, detections: Sequence[Detection]) -> dict[str, object]:
    """Recorded tracker inputs for fixture replay. The synthetic window has one subject, so every
    detection carries ByteTrack ID 1; bridge boxes for detector-gap frames are the generator's
    ground-truth boxes standing in for CSRT output (disclosed in ``bridge_source``)."""
    return {
        "tracker_ids": {d.detection_id: SYNTH_TRACKER_ID for d in detections},
        "bridge_boxes": {str(n): list(detection_box(spec.scene, n))
                         for n, fs in enumerate(spec.frames) if not fs.detected},
        "bridge_source": "synthetic ground-truth boxes standing in for OpenCV CSRT",
    }


def build_track(spec: WindowSpec, ids: WindowIds, digest: str, frames: Sequence[FrameReference],
                detections: Sequence[Detection], raw: dict[str, Any], root: Path,
                cfg: PolicyConfig, clock: FixedClock) -> Track:
    """Track from ``tracking.confirm_track`` over the recorded tracker inputs."""
    seed = next(d for d in detections if d.frame_id == frame_id(ids.video_id, spec.target_frame))
    request = TrackRequest(
        track_id=ids.track_id, case_id=ids.case_id, video_id=ids.video_id,
        source_sha256=digest, subject_type=SubjectType.LICENSE_PLATE,
        seed_frame_id=seed.frame_id, seed_bbox_px=seed.bbox_px,
        seed_detection_id=seed.detection_id)
    tracked = [TrackedDetection(d, raw["tracker_ids"].get(d.detection_id)) for d in detections]
    bridge = {int(n): (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
              for n, b in raw["bridge_boxes"].items()}
    resolver = FixtureFrameResolver(root, ids.video_id)
    meta = TrackMeta(tracker_version=TRACKER_VERSION, detector_model_id=DETECTOR_MODEL_ID,
                     mode=InferenceMode.FIXTURE, created_at=clock.at(200))
    outcome = confirm_track(request, frames, tracked, lambda ref: load_frame(ref, resolver),
                            cfg, meta, lambda: ReplayBridger(bridge))
    return outcome.track


def ground_truth(spec: WindowSpec, rendered: Sequence[RenderedFrame]) -> dict[str, object]:
    """EVALUATION ONLY. Never read by tracking or reconstruction."""
    sc = spec.scene
    frames = []
    for n, (fs, rf) in enumerate(zip(spec.frames, rendered, strict=True)):
        occluded = 0.0
        if rf.occlusion_mask is not None:
            x1, y1, x2, y2 = detection_box(sc, n)
            occluded = round(float(rf.occlusion_mask[y1:y2, x1:x2].mean() / 255.0), 4)
        frames.append({
            "frame_number": n,
            "plate_box_float": list(gt_box(sc, n)),
            "plate_origin": list(plate_origin(sc, n)),
            "occluded_fraction": occluded,
            **asdict(fs),
        })
    return {
        "evaluation_only": True,
        "synth_version": SYNTH_VERSION,
        "fixture_id": spec.fixture_id,
        "seed": spec.seed,
        "description": spec.description,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "target_frame": spec.target_frame,
        "plate_text": sc.plate_text,
        "scene": asdict(sc),
        "frames": frames,
    }


# ------------------------------------------------------------------------------------------------
# Writer
# ------------------------------------------------------------------------------------------------


def write_window(spec: WindowSpec, out_root: Path, cfg: PolicyConfig) -> Path:
    root = out_root / spec.fixture_id
    clock = FixedClock(BASE_TIME)
    ids = WindowIds.for_window(spec, clock)
    rendered = render_window(spec)

    hashes = [write_png(root / "frames" / f"f{n:04d}.png", rf.image)
              for n, rf in enumerate(rendered)]
    digest = source_digest(hashes)
    frames = build_frames(spec, ids, hashes, clock)
    detections = build_detections(spec, ids, clock)
    raw = tracker_inputs(spec, detections)
    track = build_track(spec, ids, digest, frames, detections, raw, root, cfg, clock)

    write_json(root / "frame_manifest.json", build_manifest(spec, ids, digest, clock))
    write_json(root / "frames.json", frames)
    write_json(root / "detections.json", detections)
    write_json(root / "tracker_inputs.json", raw)
    write_json(root / "track.json", track)
    write_json(root / "window.json", {
        "fixture_id": spec.fixture_id,
        "synth_version": SYNTH_VERSION,
        "case_id": ids.case_id,
        "video_id": ids.video_id,
        "track_id": ids.track_id,
        "source_sha256": digest,
        "target_frame_id": frame_id(ids.video_id, spec.target_frame),
        "target_bbox_px": list(detection_box(spec.scene, spec.target_frame)),
        "config_sha256": cfg.config_sha256,
    })

    gt = root / "ground_truth"
    write_png(gt / "clean_target.png", rendered[spec.target_frame].clean)
    for n, rf in enumerate(rendered):
        if rf.occlusion_mask is not None:
            mask_bgr = cv2.cvtColor(rf.occlusion_mask, cv2.COLOR_GRAY2BGR)
            write_png(gt / f"occlusion_mask_f{n:04d}.png", mask_bgr)
    write_json(gt / "truth.json", ground_truth(spec, rendered))
    return root


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("fixtures") / "synthetic")
    args = parser.parse_args(argv)
    cfg = default_policy()
    for spec in default_windows():
        print(write_window(spec, args.out, cfg))


if __name__ == "__main__":
    main()
