"""Pre-alignment hard gates (section 9, algorithm steps 3-5): track, target, and donor preflight.

Every gate is recorded. Track and target gates are all logged and the first failure becomes the
refusal reason. A donor is evaluated in a fixed order (lineage, identity, then measured gates);
a rejected donor records its first failing gate, a passing donor records every gate. A donor never
moves between tracks or videos: lineage is checked before any pixel is loaded.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections.abc import Mapping, Sequence

import numpy as np

from probity.domain.enums import ObservationSource, PolicyOutcome, PolicyStage, ReasonCode
from probity.domain.errors import ProbityError
from probity.domain.ids import parse_frame_id
from probity.domain.models import Detection, FrameReference, Track, TrackObservation
from probity.domain.policy import PolicyConfig
from probity.ports import CancelToken
from probity.reconstruction import quality as q
from probity.reconstruction.decisions import DecisionLog, Gate, first_failure
from probity.reconstruction.io import (
    FrameIntegrityError,
    NonEvidentiaryInput,
    assert_evidentiary_input,
)
from probity.reconstruction.tracking import iou
from probity.reconstruction.types import (
    BBox,
    DonorMeasure,
    FrameLoader,
    PreflightResult,
    QualityScores,
    SubjectMeasure,
)

__all__ = ["DonorMeasure", "PreflightResult", "SubjectMeasure", "run_preflight"]


# ------------------------------------------------------------------------------------------------
# Measures
# ------------------------------------------------------------------------------------------------


def _contains(outer: BBox, inner: BBox) -> bool:
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])


def occluded_fraction(box: BBox, frame_w: int, frame_h: int, own: Detection | None,
                      frame_detections: Sequence[Detection]) -> float:
    """Fraction of ``box`` covered by other detections or lying outside the frame, or the
    detector's own ``occlusion_score`` if larger. A detection that fully contains the subject
    (its carrier, e.g. the vehicle) is not a foreground occluder."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    covered = np.ones((h, w), dtype=bool)
    vx1, vy1 = max(0, x1) - x1, max(0, y1) - y1
    vx2, vy2 = min(frame_w, x2) - x1, min(frame_h, y2) - y1
    if vx2 > vx1 and vy2 > vy1:
        covered[vy1:vy2, vx1:vx2] = False
    for det in frame_detections:
        if own is not None and det.detection_id == own.detection_id:
            continue
        if _contains(det.bbox_px, box):
            continue
        ox1, oy1 = max(det.bbox_px[0], x1) - x1, max(det.bbox_px[1], y1) - y1
        ox2, oy2 = min(det.bbox_px[2], x2) - x1, min(det.bbox_px[3], y2) - y1
        if ox2 > ox1 and oy2 > oy1:
            covered[oy1:oy2, ox1:ox2] = True
    reported = own.occlusion_score if own is not None and own.occlusion_score is not None else 0.0
    return max(float(covered.mean()), float(reported))


def median_track_aspect(track: Track) -> float:
    return statistics.median(q.box_aspect(o.bbox_px) for o in track.observations if o.accepted)


def scale_ratio(box: BBox, ref: BBox) -> float:
    (bw, bh), (rw, rh) = q.box_size(box), q.box_size(ref)
    return math.sqrt((bw * bh) / (rw * rh))


def aspect_change(box: BBox, ref: BBox) -> float:
    return abs(q.box_aspect(box) / q.box_aspect(ref) - 1.0)


# ------------------------------------------------------------------------------------------------
# Gate builders (pure)
# ------------------------------------------------------------------------------------------------

T, QL, PF = PolicyStage.TRACK, PolicyStage.QUALITY, PolicyStage.PREFLIGHT


def track_gates(track: Track, video_id: str, case_id: str, cfg: PolicyConfig) -> list[Gate]:
    t = cfg.track
    return [
        Gate(ReasonCode.DONOR_TRACK_MISMATCH, T, track.video_id, "==", video_id, None, None,
             "Track belongs to the run's video", "Track belongs to another video"),
        Gate(ReasonCode.DONOR_TRACK_MISMATCH, T, track.case_id, "==", case_id, None, None,
             "Track belongs to the run's case", "Track belongs to another case"),
        Gate(ReasonCode.TRACK_CONFIRMED, T, str(track.state), "==", "CONFIRMED", None, None,
             "Track state is CONFIRMED", "Track is not confirmed",
             ReasonCode.TRACK_NOT_CONFIRMED),
        Gate(ReasonCode.TRACK_CONFIRMED, T, track.detector_observation_count, ">=",
             t.min_detector_observations, "observations", "track.min_detector_observations",
             "Detector-backed observations meet minimum", "Too few detector-backed observations",
             ReasonCode.TRACK_NOT_CONFIRMED),
        Gate(ReasonCode.TRACK_CONFIRMED, T, track.mean_confidence, ">=", t.min_mean_confidence,
             "confidence", "track.min_mean_confidence", "Mean detector confidence meets minimum",
             "Mean detector confidence below minimum", ReasonCode.TRACK_NOT_CONFIRMED),
    ]


def target_selection_gates(obs: TrackObservation | None, target_box: BBox, cfg: PolicyConfig
                           ) -> list[Gate]:
    source = str(obs.source) if obs is not None else "NONE"
    gates = [Gate(ReasonCode.TRACK_CONFIRMED, T, source, "!=", "NONE", None, None,
                  "Target frame has an accepted track observation",
                  "Target frame is not an accepted observation of this track",
                  ReasonCode.TRACK_NOT_CONFIRMED)]
    if obs is None:
        return gates
    gates.append(Gate(ReasonCode.IDENTITY_GEOMETRY_MISMATCH, T, source, "!=", "CSRT_BRIDGE",
                      None, None, "Target observation is not bridge-only",
                      "Target frame is only bridged; no detector or analyst box"))
    gates.append(Gate(ReasonCode.IDENTITY_GEOMETRY_MISMATCH, T, iou(target_box, obs.bbox_px),
                      ">=", cfg.track.min_iou_for_association, "IoU",
                      "track.min_iou_for_association", "Target box matches the tracked box",
                      "Target box does not match the tracked subject"))
    return gates


def target_quality_gates(box: BBox, quality: QualityScores, cfg: PolicyConfig
                         ) -> list[Gate]:
    w, h = q.box_size(box)
    t = cfg.target
    return [
        Gate(ReasonCode.TARGET_TOO_SMALL, QL, w, ">=", t.min_width_px, "px",
             "target.min_width_px", "Target width meets minimum", "Target narrower than minimum"),
        Gate(ReasonCode.TARGET_TOO_SMALL, QL, h, ">=", t.min_height_px, "px",
             "target.min_height_px", "Target height meets minimum",
             "Target shorter than minimum"),
        Gate(ReasonCode.TARGET_QUALITY_TOO_LOW, QL, quality.Q, ">=", t.min_quality, "quality",
             "target.min_quality", "Target quality above floor", "Target quality below floor"),
    ]


def donor_lineage_gates(obs: TrackObservation, track: Track, video_id: str) -> list[Gate]:
    donor_video = parse_frame_id(obs.frame_id)[0]
    member = "MEMBER" if (obs.accepted and obs in track.observations) else "NOT_MEMBER"
    return [
        Gate(ReasonCode.DONOR_TRACK_MISMATCH, PF, donor_video, "==", video_id, None, None,
             "Donor frame belongs to the run's video", "Donor frame belongs to another video"),
        Gate(ReasonCode.DONOR_TRACK_MISMATCH, PF, member, "==", "MEMBER", None, None,
             "Donor is an accepted observation of this track",
             "Donor is not an accepted observation of this track"),
        Gate(ReasonCode.IDENTITY_GEOMETRY_MISMATCH, PF, str(obs.source), "==", "DETECTOR", None,
             None, "Donor identity is detector-backed",
             "Bridged-only or analyst-only identity cannot donate pixels"),
    ]


def donor_measured_gates(m: DonorMeasure, target: SubjectMeasure, cfg: PolicyConfig
                         ) -> list[Gate]:
    t, d = cfg.track, cfg.donor
    return [
        Gate(ReasonCode.DONOR_OUTSIDE_WINDOW, PF, abs(m.dt_s), "<=", t.window_radius_s, "s",
             "track.window_radius_s", "Within temporal window", "Outside temporal window"),
        Gate(ReasonCode.IDENTITY_GEOMETRY_MISMATCH, PF, m.scale, ">=", t.box_scale_ratio_min,
             "ratio", "track.box_scale_ratio_min", "Box scale ratio in range",
             "Box scale ratio too small"),
        Gate(ReasonCode.IDENTITY_GEOMETRY_MISMATCH, PF, m.scale, "<=", t.box_scale_ratio_max,
             "ratio", "track.box_scale_ratio_max", "Box scale ratio in range",
             "Box scale ratio too large"),
        Gate(ReasonCode.IDENTITY_GEOMETRY_MISMATCH, PF, m.aspect_change, "<=",
             t.max_aspect_change, "fraction", "track.max_aspect_change",
             "Aspect change in range", "Aspect change exceeds limit"),
        Gate(ReasonCode.DONOR_OBSTRUCTED, PF, m.occluded, "<=", d.max_occluded_fraction,
             "fraction", "donor.max_occluded_fraction", "Unobstructed view",
             "Obstruction exceeds limit"),
        Gate(ReasonCode.DONOR_NOT_CLEARER, PF, m.quality.Q, ">=", d.min_quality, "quality",
             "donor.min_quality", "Donor quality above floor",
             "Donor quality below absolute floor"),
        Gate(ReasonCode.DONOR_NOT_CLEARER, PF, m.quality.Q - target.quality.Q, ">=",
             d.min_quality_gain, "quality", "donor.min_quality_gain",
             "Donor clearer than target", "Donor quality gain below minimum"),
        Gate(ReasonCode.LIGHTING_OUT_OF_RANGE, PF, m.exposure_stops, "<=",
             d.max_exposure_delta_stops, "stops", "donor.max_exposure_delta_stops",
             "Exposure difference in range", "Exposure difference exceeds limit"),
    ]


# ------------------------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------------------------


def _check_all(log: DecisionLog, gates: Sequence[Gate], subject: str) -> ReasonCode | None:
    refusal = None
    for g in gates:
        if not log.check(g, subject) and refusal is None:
            refusal = g.failure_code
    return refusal


def _load(ref: FrameReference, load: FrameLoader) -> np.ndarray:
    return load(assert_evidentiary_input(ref))


def _input_rejection(log: DecisionLog, exc: ProbityError, stage: PolicyStage, subject: str
                     ) -> ReasonCode:
    code = (ReasonCode.GENERATED_INPUT_REJECTED if isinstance(exc, NonEvidentiaryInput)
            else ReasonCode.DECODE_NOT_DETERMINISTIC)
    log.add(code, stage, subject, PolicyOutcome.REJECT, exc.message,
            observed="REJECTED", operator="==", threshold="SOURCE_FRAME")
    return code


def run_preflight(
    log: DecisionLog,
    *,
    track: Track,
    video_id: str,
    case_id: str,
    target_frame_id: str,
    target_box: BBox,
    frames: Mapping[str, FrameReference],
    load: FrameLoader,
    detections: Mapping[str, Sequence[Detection]],
    cfg: PolicyConfig,
    cancel: CancelToken | None = None,
    candidates: Sequence[TrackObservation] | None = None,
) -> PreflightResult:
    """Gate the track, the target, and every candidate donor (default: the track's accepted
    observations other than the target frame). ``detections`` maps frame_id to that frame's
    detections, used for the occluded fraction."""
    refusal = _check_all(log, track_gates(track, video_id, case_id, cfg), track.track_id)
    for o in track.observations:
        if not o.accepted:
            log.add(o.reason_code or ReasonCode.IDENTITY_GEOMETRY_MISMATCH, T, o.frame_id,
                    PolicyOutcome.REJECT, f"{o.source} observation excluded by track confirmation",
                    observed=False, operator="==", threshold=True)
    if refusal is not None:
        return PreflightResult(None, (), refusal, 1.0)

    at_target = [o for o in track.observations if o.accepted and o.frame_id == target_frame_id]
    target_obs = next((o for o in at_target if o.source is not ObservationSource.CSRT_BRIDGE),
                      at_target[0] if at_target else None)
    sel = target_selection_gates(target_obs, target_box, cfg)
    failed = first_failure(sel)
    for g in sel:
        log.check(g, target_frame_id)
        if g is failed:
            return PreflightResult(None, (), g.failure_code, 1.0)
    assert target_obs is not None

    median_aspect = median_track_aspect(track)
    by_det = {d.detection_id: d for ds in detections.values() for d in ds}

    def measure(obs: TrackObservation, box: BBox, image: np.ndarray, ref: FrameReference
                ) -> tuple[float, QualityScores]:
        own = by_det.get(obs.detection_id) if obs.detection_id else None
        occ = occluded_fraction(box, ref.width_px, ref.height_px, own,
                                detections.get(ref.frame_id, ()))
        return occ, q.measure_observation(image, box, obs.confidence, occ, median_aspect, cfg)

    target_ref = frames[target_frame_id]
    try:
        target_img = _load(target_ref, load)
    except (NonEvidentiaryInput, FrameIntegrityError) as exc:
        return PreflightResult(None, (), _input_rejection(log, exc, QL, target_frame_id),
                               median_aspect)
    t_occ, t_q = measure(target_obs, target_box, target_img, target_ref)
    target = SubjectMeasure(target_obs, target_ref, target_img, target_box, t_occ, t_q)
    refusal = _check_all(log, target_quality_gates(target_box, t_q, cfg), target_frame_id)
    if refusal is not None:
        return PreflightResult(target, (), refusal, median_aspect)

    target_luma = q.crop(q.to_luma(target_img), target_box)
    pool = candidates if candidates is not None else [
        o for o in track.observations if o.accepted]
    donors: list[DonorMeasure] = []
    for obs in sorted(pool, key=lambda o: (o.pts_us, o.frame_id, str(o.source))):
        if obs.frame_id == target_frame_id:
            continue
        if cancel is not None:
            cancel.checkpoint()
        start = len(log.rows)
        lineage = donor_lineage_gates(obs, track, video_id)
        failed = first_failure(lineage)
        if failed is not None:
            log.check(failed, obs.frame_id)
            continue
        ref = frames.get(obs.frame_id)
        if ref is None:
            raise ProbityError(f"no FrameReference supplied for track frame {obs.frame_id}")
        try:
            img = _load(ref, load)
        except (NonEvidentiaryInput, FrameIntegrityError) as exc:
            _input_rejection(log, exc, PF, obs.frame_id)
            continue
        occ, dq = measure(obs, obs.bbox_px, img, ref)
        m = DonorMeasure(
            obs=obs, frame=ref, image=img, box=obs.bbox_px, occluded=occ, quality=dq,
            dt_s=(obs.pts_us - target_obs.pts_us) / 1_000_000,
            scale=scale_ratio(obs.bbox_px, target_box),
            aspect_change=aspect_change(obs.bbox_px, target_box),
            exposure_stops=q.exposure_delta_stops(q.crop(q.to_luma(img), obs.bbox_px),
                                                  target_luma),
            decision_ids=())
        measured = donor_measured_gates(m, target, cfg)
        failed = first_failure(measured)
        if failed is not None:
            log.check(failed, obs.frame_id)
            continue
        for g in (*lineage, *measured):
            log.check(g, obs.frame_id)
        donors.append(dataclasses.replace(m, decision_ids=log.ids[start:]))
    return PreflightResult(target, tuple(donors), None, median_aspect)
