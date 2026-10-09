"""Fixture YOLO/ByteTrack adapter: replay on-disk detections and tracks.

Never calls a network, a weights file, or an OCR/plate-text path. Outputs are the stored
records in a stable order so the same golden inputs always produce the same domain objects.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

from probity.domain.enums import AdapterMode, HealthStatus
from probity.domain.errors import FixtureNotFound
from probity.domain.ids import parse_frame_id, utc_now
from probity.domain.models import AdapterHealth, Detection, FrameReference, Track
from probity.ports import TrackRequest

ADAPTER_NAME = "yolo-bytetrack"
SCHEMA_VERSION = "1.0"
DEFAULT_MODEL_ID = "fixture/yolov8n-plate-sign-v1"

_DETECTION_FILES = ("detections.json", "detections.jsonl")
_TRACK_FILES = ("track.json", "tracks.json", "track_confirmed.json", "track_not_confirmed.json")


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict[str, object]]:
    if path.suffix == ".jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    data = _read_json(path)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"{path} is not a JSON object or array")


def _detection_paths(root: Path) -> list[Path]:
    found: list[Path] = []
    for name in _DETECTION_FILES:
        path = root / name
        if path.is_file():
            found.append(path)
    extra = root / "detections"
    if extra.is_dir():
        found.extend(
            sorted(
                p for p in extra.iterdir()
                if p.suffix in {".json", ".jsonl"} and p.is_file()
            )
        )
    return found


def _track_paths(root: Path) -> list[Path]:
    found: list[Path] = []
    for name in _TRACK_FILES:
        path = root / name
        if path.is_file():
            found.append(path)
    extra = root / "tracks"
    if extra.is_dir():
        found.extend(sorted(p for p in extra.iterdir() if p.suffix == ".json" and p.is_file()))
    return found


class FixtureYoloAdapter:
    """``DetectorTracker`` that serves verified on-disk detections and tracks."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._detections = self._load_detections()
        self._tracks = self._load_tracks()

    @property
    def adapter_name(self) -> str:
        return ADAPTER_NAME

    @property
    def model_id(self) -> str | None:
        if self._detections:
            return self._detections[0].model_id
        if self._tracks:
            return self._tracks[0].detector_model_id
        return DEFAULT_MODEL_ID

    @property
    def mode(self) -> AdapterMode:
        return AdapterMode.FIXTURE

    @property
    def schema_version(self) -> str:
        return SCHEMA_VERSION

    def _load_detections(self) -> tuple[Detection, ...]:
        if not self._root.is_dir():
            return ()
        out: list[Detection] = []
        seen: set[str] = set()
        for path in _detection_paths(self._root):
            for raw in _records(path):
                det = Detection.model_validate(raw)
                if det.detection_id in seen:
                    continue
                seen.add(det.detection_id)
                out.append(det)
        out.sort(key=lambda d: (d.video_id, parse_frame_id(d.frame_id)[1], d.detection_id))
        return tuple(out)

    def _load_tracks(self) -> tuple[Track, ...]:
        if not self._root.is_dir():
            return ()
        out: list[Track] = []
        seen: set[str] = set()
        for path in _track_paths(self._root):
            for raw in _records(path):
                track = Track.model_validate(raw)
                if track.track_id in seen:
                    continue
                seen.add(track.track_id)
                out.append(track)
        out.sort(key=lambda t: (t.video_id, t.track_id))
        return tuple(out)

    async def health(self) -> AdapterHealth:
        started = time.monotonic()
        ok = bool(self._detections or self._tracks)
        detail = None if ok else f"no detections or tracks under {self._root.as_posix()}"
        return AdapterHealth(
            adapter_name=self.adapter_name,
            model_id=self.model_id,
            mode=self.mode,
            status=HealthStatus.OK if ok else HealthStatus.UNAVAILABLE,
            schema_version=self.schema_version,
            checked_at=utc_now(),
            latency_ms=max(0, int((time.monotonic() - started) * 1000)),
            detail=detail,
        )

    async def detect(
        self, frames: Sequence[FrameReference], classes: set[str]
    ) -> Sequence[Detection]:
        if not self._detections:
            raise FixtureNotFound(f"no fixture detections under {self._root.as_posix()}")
        wanted_frames = {ref.frame_id for ref in frames}
        return tuple(
            det
            for det in self._detections
            if det.frame_id in wanted_frames and det.class_name in classes
        )

    async def track(self, request: TrackRequest) -> Track:
        if not self._tracks:
            raise FixtureNotFound(f"no fixture tracks under {self._root.as_posix()}")
        for candidate in self._tracks:
            if candidate.track_id == request.track_id:
                if candidate.video_id != request.video_id:
                    raise FixtureNotFound(
                        f"track {request.track_id} belongs to video {candidate.video_id}"
                    )
                return candidate
        for candidate in self._tracks:
            if (
                candidate.video_id == request.video_id
                and candidate.seed_frame_id == request.seed_frame_id
                and candidate.seed_bbox_px == request.seed_bbox_px
            ):
                return candidate
        raise FixtureNotFound(
            f"no fixture track for {request.track_id} under {self._root.as_posix()}"
        )
