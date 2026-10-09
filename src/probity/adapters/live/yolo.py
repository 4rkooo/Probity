"""Live YOLO + ByteTrack adapter.

Ultralytics is imported only when a local checkpoint file exists. CoreWeave remote execution is
not wired: this branch has no verified endpoint, so that path raises ``NotConfigured`` instead of
guessing a URL. Tests that need ultralytics skip when it is absent; weights are never downloaded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from probity.domain.enums import AdapterMode, ErrorCode, HealthStatus, InferenceMode
from probity.domain.errors import ProbityError, ValidationFailed
from probity.domain.ids import utc_now
from probity.domain.models import AdapterHealth, Detection, FrameReference, Track
from probity.domain.policy import PolicyConfig, default_policy
from probity.ports import TrackRequest
from probity.reconstruction.determinism import seeded_uuid7
from probity.reconstruction.types import TrackedDetection

ADAPTER_NAME = "yolo-bytetrack"
SCHEMA_VERSION = "1.0"
DEFAULT_MODEL_ID = "yolov8n-plate-sign-v1"
DEFAULT_CLASS_NAMES: dict[int, str] = {0: "license_plate", 1: "traffic_sign", 2: "vehicle"}

FrameLoader = Callable[[FrameReference], Any]


class NotConfigured(ProbityError):
    """Live YOLO/ByteTrack or CoreWeave remote execution is not available on this machine."""

    code = ErrorCode.SPONSOR_UNAVAILABLE


def ultralytics_available() -> bool:
    """True when the optional ``live-yolo`` extra is installed. Does not import the package."""
    return importlib.util.find_spec("ultralytics") is not None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise NotConfigured(
            "ultralytics is not installed. Optional extra "
            'live-yolo = ["ultralytics>=8.3,<9", "lap>=0.5.12"] is requested of Person 1 '
            "(already on person2/requests-for-p1). Live tests skip without it."
        ) from exc
    return YOLO


class LiveYoloAdapter:
    """``DetectorTracker`` for a local Ultralytics checkpoint. CoreWeave stays NotConfigured."""

    def __init__(
        self,
        *,
        weights_path: Path | str | None = None,
        model_id: str | None = None,
        coreweave: bool = False,
        class_names: Mapping[int, str] | None = None,
        frames: Sequence[FrameReference] | None = None,
        frame_loader: FrameLoader | None = None,
        policy: PolicyConfig | None = None,
    ) -> None:
        self._weights = Path(weights_path) if weights_path is not None else None
        self._model_id = model_id or DEFAULT_MODEL_ID
        self._coreweave = coreweave
        if class_names is not None:
            self._class_names = dict(class_names)
        else:
            self._class_names = dict(DEFAULT_CLASS_NAMES)
        self._frames = tuple(frames) if frames is not None else ()
        self._load = frame_loader
        self._policy = policy if policy is not None else default_policy()
        self._model: Any = None
        if self._weights is not None and self._weights.is_file():
            self._model_sha256 = _file_sha256(self._weights)
        else:
            self._model_sha256 = None

    @property
    def adapter_name(self) -> str:
        return ADAPTER_NAME

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def mode(self) -> AdapterMode:
        if self._unready_reason() is None:
            return AdapterMode.LIVE
        return AdapterMode.DISABLED

    @property
    def schema_version(self) -> str:
        return SCHEMA_VERSION

    def _unready_reason(self) -> str | None:
        if self._coreweave:
            return (
                "CoreWeave remote YOLO execution is not configured. "
                "No endpoint or API URL is defined on this branch."
            )
        if self._weights is None or not self._weights.is_file():
            return "No local YOLO checkpoint file; downloads are disabled."
        if not ultralytics_available():
            return (
                "ultralytics is not installed. Optional extra "
                'live-yolo = ["ultralytics>=8.3,<9", "lap>=0.5.12"].'
            )
        return None

    def run_on_coreweave(self) -> None:
        """Refuse remote GPU dispatch. Event-day SDKs are unknown; do not invent a URL."""
        raise NotConfigured(
            "CoreWeave remote YOLO execution is not configured. "
            "No endpoint, queue URL, or API path is defined on this branch."
        )

    def _require_local(self) -> None:
        reason = self._unready_reason()
        if reason is not None:
            raise NotConfigured(reason)

    def _yolo(self) -> Any:
        self._require_local()
        assert self._weights is not None
        if self._model is None:
            self._model = _import_yolo()(str(self._weights))
        return self._model

    async def health(self) -> AdapterHealth:
        started = time.monotonic()
        reason = self._unready_reason()
        status = HealthStatus.OK if reason is None else (
            HealthStatus.UNAVAILABLE if self._coreweave else HealthStatus.DISABLED
        )
        return AdapterHealth(
            adapter_name=self.adapter_name,
            model_id=self.model_id,
            mode=self.mode,
            status=status,
            schema_version=self.schema_version,
            checked_at=utc_now(),
            latency_ms=max(0, int((time.monotonic() - started) * 1000)),
            detail=reason,
        )

    def _class_name(self, class_id: int, names: Mapping[int, str] | None) -> str:
        if names is not None and class_id in names:
            return str(names[class_id])
        return self._class_names.get(class_id, f"class_{class_id}")

    def _to_detection(
        self,
        ref: FrameReference,
        class_id: int,
        class_name: str,
        confidence: float,
        xyxy: Sequence[float],
        tracker_id: int | None,
    ) -> Detection:
        x1 = max(0, min(ref.width_px - 1, int(round(xyxy[0]))))
        y1 = max(0, min(ref.height_px - 1, int(round(xyxy[1]))))
        x2 = max(x1 + 1, min(ref.width_px, int(round(xyxy[2]))))
        y2 = max(y1 + 1, min(ref.height_px, int(round(xyxy[3]))))
        label = f"yolo:{ref.frame_id}:{x1},{y1},{x2},{y2}:{class_id}:{tracker_id}"
        return Detection.create(
            detection_id=seeded_uuid7(label, 1_760_000_000_000),
            video_id=ref.video_id,
            frame_id=ref.frame_id,
            pts_us=ref.pts_us,
            model_id=self._model_id,
            model_sha256=self._model_sha256,
            class_id=class_id,
            class_name=class_name,
            confidence=min(1.0, max(0.0, float(confidence))),
            bbox_px=(x1, y1, x2, y2),
            bbox_norm=(
                x1 / ref.width_px, y1 / ref.height_px, x2 / ref.width_px, y2 / ref.height_px
            ),
            occlusion_score=None,
            inference_mode=InferenceMode.LIVE,
        )

    def _predict_boxes(
        self, ref: FrameReference, *, track: bool
    ) -> list[tuple[Detection, int | None]]:
        if self._load is None:
            raise ValidationFailed(
                "LiveYoloAdapter needs a bound frame loader; no remote decode URL exists"
            )
        image = self._load(ref)
        model = self._yolo()
        if track:
            results = model.track(source=image, persist=True, verbose=False)
        else:
            results = model.predict(source=image, verbose=False)
        out: list[tuple[Detection, int | None]] = []
        for result in results:
            names = getattr(result, "names", None)
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls.item())
                class_name = self._class_name(class_id, names)
                conf = float(box.conf.item())
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                tid = int(box.id.item()) if getattr(box, "id", None) is not None else None
                out.append((self._to_detection(ref, class_id, class_name, conf, xyxy, tid), tid))
        return out

    async def detect(
        self, frames: Sequence[FrameReference], classes: set[str]
    ) -> Sequence[Detection]:
        self._require_local()
        found: list[Detection] = []
        for ref in frames:
            for det, _tid in self._predict_boxes(ref, track=False):
                if det.class_name in classes:
                    found.append(det)
        found.sort(key=lambda d: (d.video_id, d.frame_id, d.detection_id))
        return tuple(found)

    async def track(self, request: TrackRequest) -> Track:
        self._require_local()
        if not self._frames:
            raise ValidationFailed("LiveYoloAdapter.track requires frames bound at construction")
        if self._load is None:
            raise ValidationFailed(
                "LiveYoloAdapter needs a bound frame loader; no remote decode URL exists"
            )
        from probity.reconstruction.tracking import TrackMeta, confirm_track

        tracked: list[TrackedDetection] = []
        for ref in sorted(self._frames, key=lambda item: item.frame_number):
            if ref.video_id != request.video_id:
                raise ValidationFailed(f"frame {ref.frame_id} belongs to another video")
            for det, tid in self._predict_boxes(ref, track=True):
                tracked.append(TrackedDetection(det, tid))
        meta = TrackMeta(
            tracker_version="probity-track-v1+ultralytics-bytetrack",
            detector_model_id=self._model_id,
            mode=InferenceMode.LIVE,
            created_at=utc_now(),
        )
        outcome = confirm_track(
            request,
            self._frames,
            tracked,
            self._load,
            self._policy,
            meta,
            None,
        )
        return outcome.track
