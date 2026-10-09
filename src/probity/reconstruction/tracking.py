"""Confirm one rigid subject track over a bounded window (section 9, algorithm step 3).

ByteTrack IDs (from the detector adapter) drive association; this module adds the policy gates:
association IoU, per-step motion, box scale/aspect against the seed, H-S histogram appearance,
and competing-box collision. A bridger (OpenCV CSRT live, recorded boxes in fixtures) may cover at
most ``track.max_bridge_gap_frames`` consecutive sampled frames without an accepted detection;
bridged observations never count toward confirmation and never donate pixels. Every evaluated
observation is recorded, accepted or rejected, with its rule code; the gate detail is kept in
``ObservationAudit`` because the frozen ``TrackObservation`` carries only the reason code.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from probity.domain.enums import InferenceMode, ObservationSource, ReasonCode, TrackState
from probity.domain.errors import ProbityError, ValidationFailed
from probity.domain.models import Detection, FrameReference, Track, TrackObservation
from probity.domain.policy import PolicyConfig
from probity.ports import TrackRequest
from probity.reconstruction.decisions import Operator, passes
from probity.reconstruction.io import assert_evidentiary_input

BBox = tuple[int, int, int, int]
FrameLoader = Callable[[FrameReference], np.ndarray]

TRACK_LOGIC_VERSION = "probity-track-v1"
HS_BINS = (30, 32)
BRIDGE_CONFIDENCE = 0.0


# ------------------------------------------------------------------------------------------------
# Inputs and outputs
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackedDetection:
    """A detector box plus the ByteTrack ID the tracker assigned to it (None if unassociated)."""

    detection: Detection
    tracker_id: int | None


class Bridger(Protocol):
    def start(self, frame_number: int, image: np.ndarray, box: BBox) -> None: ...
    def step(self, frame_number: int, image: np.ndarray) -> BBox | None: ...


BridgerFactory = Callable[[], Bridger]


@dataclass(frozen=True)
class ObservationAudit:
    frame_number: int
    source: ObservationSource
    accepted: bool
    reason_code: ReasonCode | None
    gate: str
    observed: float | str | None
    operator: Operator
    threshold: float | str | None
    detail: str
    detection_id: str | None = None


@dataclass(frozen=True)
class TrackingOutcome:
    track: Track
    audit: tuple[ObservationAudit, ...]
    sampled_frame_numbers: tuple[int, ...]
    tracker_id: int | None


@dataclass(frozen=True)
class TrackMeta:
    tracker_version: str
    detector_model_id: str | None
    mode: InferenceMode
    created_at: str


# ------------------------------------------------------------------------------------------------
# Geometry and appearance
# ------------------------------------------------------------------------------------------------


def area(box: BBox) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def center_step_diag(box: BBox, prev: BBox) -> float:
    """Center displacement divided by the previous box diagonal."""
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    px, py = (prev[0] + prev[2]) / 2.0, (prev[1] + prev[3]) / 2.0
    diag = math.hypot(prev[2] - prev[0], prev[3] - prev[1])
    return math.hypot(cx - px, cy - py) / diag


def scale_ratio(box: BBox, ref: BBox) -> float:
    return math.sqrt(area(box) / area(ref))


def aspect_change(box: BBox, ref: BBox) -> float:
    a = (box[2] - box[0]) / (box[3] - box[1])
    r = (ref[2] - ref[0]) / (ref[3] - ref[1])
    return abs(a / r - 1.0)


def hs_histogram(bgr: np.ndarray, box: BBox) -> np.ndarray:
    """Hue-saturation histogram of the subject box; brightness-robust by construction."""
    x1, y1, x2, y2 = box
    hsv = cv2.cvtColor(np.ascontiguousarray(bgr[y1:y2, x1:x2]), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HS_BINS), [0, 180, 0, 256])
    return np.asarray(hist, dtype=np.float32)


def hist_correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


def clip_box(box: BBox, width: int, height: int) -> BBox | None:
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


# ------------------------------------------------------------------------------------------------
# Identity gates
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentityMetrics:
    association_iou: float
    center_step: float
    scale: float
    aspect: float
    appearance: float
    competing_iou: float


@dataclass(frozen=True)
class GateFailure:
    gate: str
    observed: float
    operator: Operator
    threshold: float
    detail: str


def identity_gates(m: IdentityMetrics, cfg: PolicyConfig) -> GateFailure | None:
    """Return the first failing identity gate, in a fixed order, or None if all pass."""
    t = cfg.track
    gates: tuple[tuple[str, float, Operator, float, str], ...] = (
        ("track.min_iou_for_association", m.association_iou, ">=", t.min_iou_for_association,
         "Too little overlap with the previous track box"),
        ("track.max_center_step_diag", m.center_step, "<=", t.max_center_step_diag,
         "Implausible per-step center jump"),
        ("track.box_scale_ratio_min", m.scale, ">=", t.box_scale_ratio_min,
         "Box much smaller than the seed"),
        ("track.box_scale_ratio_max", m.scale, "<=", t.box_scale_ratio_max,
         "Box much larger than the seed"),
        ("track.max_aspect_change", m.aspect, "<=", t.max_aspect_change,
         "Aspect ratio inconsistent with the seed"),
        ("track.min_hsv_hist_correlation", m.appearance, ">=", t.min_hsv_hist_correlation,
         "Appearance (H-S histogram) differs from the seed"),
        ("track.max_competing_box_iou", m.competing_iou, "<=", t.max_competing_box_iou,
         "Another detected box collides with this one"),
    )
    for key, observed, op, threshold, detail in gates:
        if not passes(observed, op, threshold):
            return GateFailure(key, observed, op, threshold, detail)
    return None


# ------------------------------------------------------------------------------------------------
# Window sampling
# ------------------------------------------------------------------------------------------------


def window_bounds(seed_pts_us: int, request: TrackRequest, cfg: PolicyConfig) -> tuple[int, int]:
    """The narrower of the request radius and policy radius (refuse-leaning)."""
    radius = min(request.window_radius_us, round(cfg.track.window_radius_s * 1_000_000))
    return max(0, seed_pts_us - radius), seed_pts_us + radius


def sample_window(frames: Sequence[FrameReference], seed: FrameReference, start_us: int,
                  end_us: int, max_fps: float, detection_frames: set[str]
                  ) -> list[FrameReference]:
    """Frames on a stride grid anchored at the seed (<= max_fps), plus the seed and every frame
    that already has a detection. Ordered by frame number."""
    inside = sorted((f for f in frames if start_us <= f.pts_us <= end_us),
                    key=lambda f: f.frame_number)
    stride = 1
    if len(inside) >= 2:
        steps = [b.pts_us - a.pts_us for a, b in zip(inside, inside[1:], strict=False)]
        dt = statistics.median(steps)
        # PTS are integer microseconds, so a measured step may be 1 us short of the true one.
        stride = max(1, math.ceil((1_000_000 / max_fps) / (dt + 1)))
    return [f for f in inside
            if (f.frame_number - seed.frame_number) % stride == 0
            or f.frame_id == seed.frame_id or f.frame_id in detection_frames]


# ------------------------------------------------------------------------------------------------
# Bridgers
# ------------------------------------------------------------------------------------------------


def csrt_available() -> bool:
    legacy = getattr(cv2, "legacy", None)
    return hasattr(cv2, "TrackerCSRT_create") or (
        legacy is not None and hasattr(legacy, "TrackerCSRT_create"))


class CsrtBridger:
    """OpenCV CSRT (needs the opencv-contrib build)."""

    def __init__(self) -> None:
        if not csrt_available():
            raise ProbityError(
                "OpenCV CSRT is unavailable; install opencv-contrib-python-headless",
                details={"cv2_version": cv2.__version__})
        self._tracker: object | None = None
        self._size: tuple[int, int] = (0, 0)

    @staticmethod
    def _create() -> object:
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        return cv2.legacy.TrackerCSRT_create()  # type: ignore[attr-defined]

    def start(self, frame_number: int, image: np.ndarray, box: BBox) -> None:
        x1, y1, x2, y2 = box
        self._tracker = self._create()
        self._size = (image.shape[1], image.shape[0])
        self._tracker.init(image, (x1, y1, x2 - x1, y2 - y1))  # type: ignore[attr-defined]

    def step(self, frame_number: int, image: np.ndarray) -> BBox | None:
        if self._tracker is None:
            return None
        ok, rect = self._tracker.update(image)  # type: ignore[attr-defined]
        if not ok:
            return None
        x, y, w, h = (float(v) for v in rect)
        return clip_box((round(x), round(y), round(x + w), round(y + h)), *self._size)


class ReplayBridger:
    """Recorded bridge boxes keyed by frame number (fixture mode). Unrecorded frames are lost."""

    def __init__(self, boxes: Mapping[int, BBox]) -> None:
        self._boxes = dict(boxes)
        self._started = False

    def start(self, frame_number: int, image: np.ndarray, box: BBox) -> None:
        self._started = True

    def step(self, frame_number: int, image: np.ndarray) -> BBox | None:
        return self._boxes.get(frame_number) if self._started else None


# ------------------------------------------------------------------------------------------------
# Track confirmation
# ------------------------------------------------------------------------------------------------


@dataclass
class _Seed:
    ref: FrameReference
    box: BBox
    source: ObservationSource
    detection: Detection | None
    confidence: float
    tracker_id: int | None


def _check_inputs(request: TrackRequest, frames: Sequence[object],
                  detections: Sequence[TrackedDetection]) -> list[FrameReference]:
    refs = [assert_evidentiary_input(f) for f in frames]
    for ref in refs:
        if ref.video_id != request.video_id:
            raise ValidationFailed(f"frame {ref.frame_id} belongs to another video")
    known = {r.frame_id for r in refs}
    for td in detections:
        det = td.detection
        if det.video_id != request.video_id:
            raise ValidationFailed(f"detection {det.detection_id} belongs to another video")
        if det.frame_id not in known:
            raise ValidationFailed(f"detection {det.detection_id} references an unknown frame")
    if request.seed_frame_id not in known:
        raise ValidationFailed("seed frame is not among the supplied frames")
    return refs


def _resolve_seed(request: TrackRequest, seed_ref: FrameReference,
                  sampled: Sequence[FrameReference],
                  by_frame: Mapping[str, list[TrackedDetection]], cfg: PolicyConfig) -> _Seed:
    min_iou = cfg.track.min_iou_for_association
    roi = request.seed_bbox_px
    in_seed = by_frame.get(seed_ref.frame_id, [])
    if request.seed_detection_id is not None:
        match = [td for td in in_seed if td.detection.detection_id == request.seed_detection_id]
        if not match:
            raise ValidationFailed("seed_detection_id is not a detection in the seed frame")
        td = match[0]
        if iou(td.detection.bbox_px, roi) < min_iou:
            raise ValidationFailed("seed_bbox_px does not match the seed detection")
        return _Seed(seed_ref, td.detection.bbox_px, ObservationSource.DETECTOR, td.detection,
                     td.detection.confidence, td.tracker_id)
    overlapping = sorted(
        (td for td in in_seed if td.tracker_id is not None
         and iou(td.detection.bbox_px, roi) >= min_iou),
        key=lambda td: (-iou(td.detection.bbox_px, roi), td.detection.detection_id))
    if overlapping:
        td = overlapping[0]
        return _Seed(seed_ref, td.detection.bbox_px, ObservationSource.DETECTOR, td.detection,
                     td.detection.confidence, td.tracker_id)
    # Analyst ROI with no detector box in the seed frame: adopt the ByteTrack ID of the nearest
    # overlapping detection within the bridge limit; the seed itself stays ANALYST_ROI (D=0.50).
    seed_pos = next(i for i, f in enumerate(sampled) if f.frame_id == seed_ref.frame_id)
    best: tuple[int, int, float, str, int] | None = None
    for i, f in enumerate(sampled):
        offset = abs(i - seed_pos)
        if offset == 0 or offset > cfg.track.max_bridge_gap_frames + 1:
            continue
        for td in by_frame.get(f.frame_id, []):
            ov = iou(td.detection.bbox_px, roi)
            if td.tracker_id is None or ov < min_iou:
                continue
            key = (offset, i, -ov, td.detection.detection_id, td.tracker_id)
            if best is None or key < best:
                best = key
    return _Seed(seed_ref, roi, ObservationSource.ANALYST_ROI, None,
                 cfg.quality.analyst_seed_confidence, best[4] if best else None)


def confirm_track(
    request: TrackRequest,
    frames: Sequence[object],
    detections: Sequence[TrackedDetection],
    load: FrameLoader,
    cfg: PolicyConfig,
    meta: TrackMeta,
    bridger_factory: BridgerFactory | None,
) -> TrackingOutcome:
    """Build the confirmed (or NOT_CONFIRMED) track for ``request``. Never borrows across videos;
    only source-decoded ``FrameReference`` inputs are accepted."""
    refs = _check_inputs(request, frames, detections)
    by_id = {r.frame_id: r for r in refs}
    seed_ref = by_id[request.seed_frame_id]
    start_us, end_us = window_bounds(seed_ref.pts_us, request, cfg)
    by_frame: dict[str, list[TrackedDetection]] = {}
    for td in detections:
        if start_us <= td.detection.pts_us <= end_us:
            by_frame.setdefault(td.detection.frame_id, []).append(td)
    sampled = sample_window(refs, seed_ref, start_us, end_us, cfg.detect.max_sample_fps,
                            set(by_frame))
    seed = _resolve_seed(request, seed_ref, sampled, by_frame, cfg)

    images: dict[str, np.ndarray] = {}

    def image(ref: FrameReference) -> np.ndarray:
        if ref.frame_id not in images:
            images[ref.frame_id] = load(ref)
        return images[ref.frame_id]

    seed_img = image(seed_ref)
    seed_hist = hs_histogram(seed_img, seed.box)
    tid = seed.tracker_id
    observations: list[TrackObservation] = []
    audit: list[ObservationAudit] = []

    def others_iou(box: BBox, frame: str, exclude: str | None) -> float:
        return max((iou(box, td.detection.bbox_px) for td in by_frame.get(frame, [])
                    if td.detection.detection_id != exclude
                    and (tid is None or td.tracker_id != tid)), default=0.0)

    def record(ref: FrameReference, source: ObservationSource, box: BBox, conf: float,
               det: Detection | None, failure: GateFailure | None, *,
               reason: ReasonCode = ReasonCode.IDENTITY_GEOMETRY_MISMATCH) -> None:
        accepted = failure is None
        observations.append(TrackObservation(
            frame_id=ref.frame_id, pts_us=ref.pts_us, bbox_px=box, source=source,
            detection_id=det.detection_id if det else None, confidence=conf, accepted=accepted,
            reason_code=None if accepted else reason))
        audit.append(ObservationAudit(
            frame_number=ref.frame_number, source=source, accepted=accepted,
            reason_code=None if accepted else reason,
            gate=failure.gate if failure else "identity",
            observed=failure.observed if failure else None,
            operator=failure.operator if failure else "none",
            threshold=failure.threshold if failure else None,
            detail=failure.detail if failure else "All identity gates passed",
            detection_id=det.detection_id if det else None))

    # Seed: collision is the only gate that applies to the reference observation itself.
    seed_comp = others_iou(seed.box, seed_ref.frame_id,
                           seed.detection.detection_id if seed.detection else None)
    seed_fail = None
    if not passes(seed_comp, "<=", cfg.track.max_competing_box_iou):
        seed_fail = GateFailure("track.max_competing_box_iou", seed_comp, "<=",
                                cfg.track.max_competing_box_iou,
                                "Seed box collides with another detected box")
    record(seed_ref, seed.source, seed.box, seed.confidence, seed.detection, seed_fail)

    def metrics(ref: FrameReference, box: BBox, prev: BBox, det_id: str | None
                ) -> IdentityMetrics:
        return IdentityMetrics(
            association_iou=iou(box, prev), center_step=center_step_diag(box, prev),
            scale=scale_ratio(box, seed.box), aspect=aspect_change(box, seed.box),
            appearance=hist_correlation(hs_histogram(image(ref), box), seed_hist),
            competing_iou=others_iou(box, ref.frame_id, det_id))

    def end(ref: FrameReference, gate: str, observed: float | None, detail: str) -> None:
        audit.append(ObservationAudit(
            ref.frame_number, ObservationSource.CSRT_BRIDGE, False, ReasonCode.TRACK_NOT_CONFIRMED,
            gate, observed, "<=" if observed is not None else "none",
            float(cfg.track.max_bridge_gap_frames) if observed is not None else None,
            f"Track ends in this direction: {detail}"))

    def walk(order: Sequence[FrameReference]) -> None:
        """A gap frame (no accepted detection) is covered only by a bridge box that passes every
        identity gate; a lost or rejected bridge, a gap over the limit, or no bridger ends the
        direction. Later same-ID detections are recorded as rejected."""
        max_gap = cfg.track.max_bridge_gap_frames
        prev, gap, broken = seed.box, 0, False
        bridger = bridger_factory() if bridger_factory is not None else None
        if bridger is not None:
            bridger.start(seed_ref.frame_number, seed_img, seed.box)
        for ref in order:
            ours = sorted(
                (td for td in by_frame.get(ref.frame_id, [])
                 if tid is not None and td.tracker_id == tid),
                key=lambda td: (-iou(td.detection.bbox_px, prev), td.detection.detection_id))
            if broken:
                for td in ours:
                    record(ref, ObservationSource.DETECTOR, td.detection.bbox_px,
                           td.detection.confidence, td.detection,
                           GateFailure("track.max_bridge_gap_frames", float(gap), "<=",
                                       float(max_gap), "Continuity broken earlier in this "
                                       "direction; identity cannot be carried across the gap"),
                           reason=ReasonCode.TRACK_NOT_CONFIRMED)
                continue
            if ours:
                for td in ours[1:]:
                    record(ref, ObservationSource.DETECTOR, td.detection.bbox_px,
                           td.detection.confidence, td.detection,
                           GateFailure("bytetrack_id", float(len(ours)), "<=", 1.0,
                                       "Duplicate box with the same ByteTrack ID"))
                det = ours[0].detection
                fail = identity_gates(metrics(ref, det.bbox_px, prev, det.detection_id), cfg)
                record(ref, ObservationSource.DETECTOR, det.bbox_px, det.confidence, det, fail)
                if fail is None:
                    prev, gap = det.bbox_px, 0
                    if bridger is not None:
                        bridger.start(ref.frame_number, image(ref), det.bbox_px)
                    continue
            gap += 1
            if bridger is None:
                broken = True
                end(ref, "bridge_unavailable", None, "no bridger configured")
                continue
            box = bridger.step(ref.frame_number, image(ref))
            if gap > max_gap:
                broken = True
                end(ref, "track.max_bridge_gap_frames", float(gap), "gap exceeds bridge limit")
                continue
            if box is None:
                broken = True
                end(ref, "bridge_lost", None, "bridger lost the subject")
                continue
            fail = identity_gates(metrics(ref, box, prev, None), cfg)
            record(ref, ObservationSource.CSRT_BRIDGE, box, BRIDGE_CONFIDENCE, None, fail)
            if fail is None:
                prev = box
            else:
                broken = True
                end(ref, fail.gate, None, "bridge box failed an identity gate")

    if seed_fail is None:
        seed_pos = next(i for i, f in enumerate(sampled) if f.frame_id == seed_ref.frame_id)
        walk(sampled[seed_pos + 1:])
        walk(list(reversed(sampled[:seed_pos])))

    track = _assemble(request, seed, observations, sampled, start_us, end_us, cfg, meta,
                      seed_rejected=seed_fail is not None)
    audit.sort(key=lambda a: a.frame_number)
    return TrackingOutcome(track=track, audit=tuple(audit),
                           sampled_frame_numbers=tuple(f.frame_number for f in sampled),
                           tracker_id=tid)


def _assemble(request: TrackRequest, seed: _Seed, observations: list[TrackObservation],
              sampled: Sequence[FrameReference], start_us: int, end_us: int, cfg: PolicyConfig,
              meta: TrackMeta, *, seed_rejected: bool) -> Track:
    ordered = tuple(sorted(observations, key=lambda o: (o.pts_us, o.source != "DETECTOR",
                                                        o.detection_id or "")))
    detector = [o for o in ordered if o.accepted and o.source is ObservationSource.DETECTOR]
    mean_conf = round(statistics.fmean(o.confidence for o in detector), 4) if detector else 0.0
    continuity = round(len(detector) / len(sampled), 4) if sampled else 0.0
    t = cfg.track
    confirmed = (not seed_rejected and len(detector) >= t.min_detector_observations
                 and mean_conf >= t.min_mean_confidence)
    reasons: tuple[ReasonCode, ...]
    if confirmed:
        reasons = (ReasonCode.TRACK_CONFIRMED,)
    elif seed_rejected:
        reasons = (ReasonCode.TRACK_NOT_CONFIRMED, ReasonCode.IDENTITY_GEOMETRY_MISMATCH)
    else:
        reasons = (ReasonCode.TRACK_NOT_CONFIRMED,)
    return Track.create(
        created_at=meta.created_at,
        track_id=request.track_id,
        case_id=request.case_id,
        video_id=request.video_id,
        subject_type=request.subject_type,
        seed_frame_id=seed.ref.frame_id,
        seed_bbox_px=seed.box,
        seed_detection_id=seed.detection.detection_id if seed.detection else None,
        tracker_version=meta.tracker_version,
        detector_model_id=meta.detector_model_id,
        window_start_us=start_us,
        window_end_us=end_us,
        state=TrackState.CONFIRMED if confirmed else TrackState.NOT_CONFIRMED,
        mean_confidence=mean_conf,
        continuity_score=continuity,
        confirmed=confirmed,
        detector_observation_count=len(detector),
        observations=ordered,
        reason_codes=reasons,
        mode=meta.mode,
    )
