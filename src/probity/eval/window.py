"""Load a committed synthetic window (frames, detections, track) with frozen-model validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from probity.domain.models import Detection, FrameManifest, FrameReference, Track
from probity.reconstruction.io import FixtureFrameResolver, read_json

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class WindowBundle:
    root: Path
    fixture_id: str
    case_id: str
    video_id: str
    track_id: str
    source_sha256: str
    target_frame_id: str
    target_bbox_px: BBox
    config_sha256: str
    manifest: FrameManifest
    frames: tuple[FrameReference, ...]
    detections: tuple[Detection, ...]
    track: Track

    @property
    def resolver(self) -> FixtureFrameResolver:
        return FixtureFrameResolver(self.root, self.video_id)

    def frame(self, frame_number: int) -> FrameReference:
        ref = self.frames[frame_number]
        if ref.frame_number != frame_number:
            raise ValueError("frames.json must be ordered by frame_number")
        return ref

    def frame_by_id(self, frame_id: str) -> FrameReference:
        for ref in self.frames:
            if ref.frame_id == frame_id:
                return ref
        raise KeyError(frame_id)


def load_window(root: Path) -> WindowBundle:
    meta = read_json(root / "window.json")
    bundle = WindowBundle(
        root=root,
        fixture_id=meta["fixture_id"],
        case_id=meta["case_id"],
        video_id=meta["video_id"],
        track_id=meta["track_id"],
        source_sha256=meta["source_sha256"],
        target_frame_id=meta["target_frame_id"],
        target_bbox_px=tuple(meta["target_bbox_px"]),
        config_sha256=meta["config_sha256"],
        manifest=FrameManifest.model_validate(read_json(root / "frame_manifest.json")),
        frames=tuple(FrameReference.model_validate(f) for f in read_json(root / "frames.json")),
        detections=tuple(Detection.model_validate(d) for d in read_json(root / "detections.json")),
        track=Track.model_validate(read_json(root / "track.json")),
    )
    if bundle.track.video_id != bundle.video_id or bundle.track.track_id != bundle.track_id:
        raise ValueError("track does not belong to this window")
    return bundle


def load_tracker_inputs(root: Path) -> dict[str, Any]:
    """Recorded ByteTrack IDs and bridge boxes for fixture replay."""
    raw: dict[str, Any] = read_json(root / "tracker_inputs.json")
    if set(raw) != {"tracker_ids", "bridge_boxes", "bridge_source"}:
        raise ValueError("tracker_inputs.json has unexpected keys")
    return raw


def load_truth(root: Path) -> dict[str, Any]:
    """EVALUATION ONLY: generator ground truth. The reconstruction path never calls this."""
    truth: dict[str, Any] = read_json(root / "ground_truth" / "truth.json")
    if truth.get("evaluation_only") is not True:
        raise ValueError("ground truth must be marked evaluation_only")
    return truth
