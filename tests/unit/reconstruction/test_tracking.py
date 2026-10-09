"""Track confirmation: ByteTrack-ID association, identity gates, CSRT bridging limits, seeds."""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from probity.domain.enums import (
    AssetKind,
    InferenceMode,
    ObservationSource,
    ReasonCode,
    SubjectType,
    TrackState,
)
from probity.domain.errors import ValidationFailed
from probity.domain.ids import frame_id, parse_frame_id
from probity.domain.models import AssetRef, Detection, FrameReference, Track
from probity.domain.policy import default_policy
from probity.eval import synth
from probity.eval.window import WindowBundle, load_tracker_inputs, load_window
from probity.ports import TrackRequest
from probity.reconstruction import tracking as tr
from probity.reconstruction.determinism import seeded_uuid7
from probity.reconstruction.io import NonEvidentiaryInput, json_bytes, load_frame

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "fixtures" / "synthetic" / "plate_translate_v1"
CFG = default_policy()
SCENE = synth.SceneSpec()
META = tr.TrackMeta(tracker_version=synth.TRACKER_VERSION,
                    detector_model_id=synth.DETECTOR_MODEL_ID, mode=InferenceMode.FIXTURE,
                    created_at="2026-10-09T17:00:00.200000Z")


@pytest.fixture(scope="module")
def win() -> WindowBundle:
    return load_window(ROOT)


@pytest.fixture(scope="module")
def images(win: WindowBundle) -> dict[str, np.ndarray]:
    return {ref.frame_id: load_frame(ref, win.resolver) for ref in win.frames}


def _n(fid: str) -> int:
    return parse_frame_id(fid)[1]


def _request(win: WindowBundle, *, seed_detection: bool = True, roi: tuple[int, int, int, int]
             | None = None, radius_us: int = 2_000_000) -> TrackRequest:
    seed = next(d for d in win.detections if d.frame_id == win.target_frame_id)
    return TrackRequest(
        track_id=win.track_id, case_id=win.case_id, video_id=win.video_id,
        source_sha256=win.source_sha256, subject_type=SubjectType.LICENSE_PLATE,
        seed_frame_id=win.target_frame_id, seed_bbox_px=roi or seed.bbox_px,
        seed_detection_id=seed.detection_id if seed_detection else None,
        window_radius_us=radius_us)


def _tracked(dets: Sequence[Detection], tid: int = 1) -> list[tr.TrackedDetection]:
    return [tr.TrackedDetection(d, tid) for d in dets]


def _gt_bridge(frames: Sequence[int]) -> dict[int, tuple[int, int, int, int]]:
    return {n: synth.detection_box(SCENE, n) for n in frames}


def _confirm(win: WindowBundle, images: Mapping[str, np.ndarray],
             tracked: Sequence[tr.TrackedDetection], *, request: TrackRequest | None = None,
             bridge: Mapping[int, tuple[int, int, int, int]] | None = None,
             no_bridger: bool = False,
             load: Callable[[FrameReference], np.ndarray] | None = None,
             frames: Sequence[object] | None = None) -> tr.TrackingOutcome:
    boxes = _gt_bridge(synth.GAP_FRAMES) if bridge is None else bridge
    return tr.confirm_track(
        request or _request(win), frames if frames is not None else win.frames, tracked,
        load or (lambda ref: images[ref.frame_id]), CFG, META,
        None if no_bridger else (lambda: tr.ReplayBridger(boxes)))


def _obs(track: Track, n: int, source: ObservationSource | None = None) -> list:
    return [o for o in track.observations if _n(o.frame_id) == n
            and (source is None or o.source is source)]


def _extra_detection(win: WindowBundle, n: int, box: tuple[int, int, int, int],
                     conf: float = 0.7) -> Detection:
    base = next(d for d in win.detections if _n(d.frame_id) == n)
    return base.revise(
        detection_id=seeded_uuid7(f"test:extra:{n}:{box}", 1_760_000_000_000),
        bbox_px=box, confidence=conf,
        bbox_norm=(box[0] / SCENE.width, box[1] / SCENE.height,
                   box[2] / SCENE.width, box[3] / SCENE.height))


def _shift(box: tuple[int, int, int, int], dx: int) -> tuple[int, int, int, int]:
    return box[0] + dx, box[1], box[2] + dx, box[3]


# ------------------------------------------------------------------------------------------------
# Committed fixture
# ------------------------------------------------------------------------------------------------


def test_committed_track_is_reproduced_by_tracking(win: WindowBundle,
                                                   images: Mapping[str, np.ndarray]) -> None:
    raw = load_tracker_inputs(ROOT)
    tracked = [tr.TrackedDetection(d, raw["tracker_ids"].get(d.detection_id))
               for d in win.detections]
    bridge = {int(k): (v[0], v[1], v[2], v[3]) for k, v in raw["bridge_boxes"].items()}
    out = _confirm(win, images, tracked, bridge=bridge)
    assert json_bytes(out.track) == (ROOT / "track.json").read_bytes()


def test_synthetic_track_confirms_with_bridged_gap(win: WindowBundle,
                                                   images: Mapping[str, np.ndarray]) -> None:
    track = _confirm(win, images, _tracked(win.detections)).track
    assert track.state is TrackState.CONFIRMED and track.confirmed
    assert track.reason_codes == (ReasonCode.TRACK_CONFIRMED,)
    bridged = [o for o in track.observations if o.source is ObservationSource.CSRT_BRIDGE]
    assert [_n(o.frame_id) for o in bridged] == list(synth.GAP_FRAMES)
    assert all(o.accepted and o.confidence == tr.BRIDGE_CONFIDENCE for o in bridged)
    assert track.detector_observation_count == len(win.track.observations) - len(bridged)
    assert all(track.window_start_us <= o.pts_us <= track.window_end_us
               for o in track.observations)
    for n in (synth.OBSTRUCTED_FRAME, synth.INCOMPATIBLE_FRAME):
        (o,) = _obs(track, n)
        assert o.accepted, "lighting/obstruction are preflight gates, not identity gates"


def test_obstructed_frame_keeps_an_appearance_margin(win: WindowBundle,
                                                     images: Mapping[str, np.ndarray]) -> None:
    seed = next(d for d in win.detections if d.frame_id == win.target_frame_id)
    ref_hist = tr.hs_histogram(images[seed.frame_id], seed.bbox_px)
    det = next(d for d in win.detections if _n(d.frame_id) == synth.OBSTRUCTED_FRAME)
    corr = tr.hist_correlation(tr.hs_histogram(images[det.frame_id], det.bbox_px), ref_hist)
    assert corr >= CFG.track.min_hsv_hist_correlation + 0.01


# ------------------------------------------------------------------------------------------------
# Confirmation counts: bridged observations never count
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("n_detections", "state"), [(4, TrackState.NOT_CONFIRMED),
                                                     (5, TrackState.CONFIRMED)])
def test_bridged_observations_do_not_count_toward_confirmation(
        win: WindowBundle, images: Mapping[str, np.ndarray], n_detections: int,
        state: TrackState) -> None:
    keep = set(range(synth.TARGET, synth.TARGET + n_detections))
    dets = [d for d in win.detections if _n(d.frame_id) in keep]
    last = synth.TARGET + n_detections - 1
    bridge = _gt_bridge(range(last + 1, last + 4))
    track = _confirm(win, images, _tracked(dets), bridge=bridge).track
    bridged = [o for o in track.observations if o.source is ObservationSource.CSRT_BRIDGE]
    assert len(bridged) == 3 and all(o.accepted for o in bridged)
    assert track.detector_observation_count == n_detections
    assert track.state is state


def test_low_mean_confidence_is_not_confirmed(win: WindowBundle,
                                              images: Mapping[str, np.ndarray]) -> None:
    weak = [d.revise(confidence=0.5) for d in win.detections]
    track = _confirm(win, images, _tracked(weak)).track
    assert track.mean_confidence == 0.5 < CFG.track.min_mean_confidence
    assert track.state is TrackState.NOT_CONFIRMED
    assert track.reason_codes == (ReasonCode.TRACK_NOT_CONFIRMED,)


# ------------------------------------------------------------------------------------------------
# Bridge limit
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("gap", "continues"), [(3, True), (4, False)])
def test_bridge_gap_limit(win: WindowBundle, images: Mapping[str, np.ndarray], gap: int,
                          continues: bool) -> None:
    missing = list(range(56, 56 + gap))
    dets = [d for d in win.detections if _n(d.frame_id) not in missing]
    out = _confirm(win, images, _tracked(dets), bridge=_gt_bridge([*synth.GAP_FRAMES, *missing]))
    track = out.track
    bridged = [_n(o.frame_id) for o in track.observations
               if o.source is ObservationSource.CSRT_BRIDGE and o.accepted]
    assert bridged == [*synth.GAP_FRAMES, *missing[:CFG.track.max_bridge_gap_frames]]
    after = [o for o in track.observations if _n(o.frame_id) > missing[-1]]
    assert after
    if continues:
        assert all(o.accepted for o in after)
    else:
        assert not any(o.accepted for o in after)
        assert {o.reason_code for o in after} == {ReasonCode.TRACK_NOT_CONFIRMED}
        (end,) = [a for a in out.audit if a.gate == "track.max_bridge_gap_frames"
                  and a.detection_id is None]
        assert end.frame_number == missing[-1] and end.observed == 4.0


def test_without_bridger_any_gap_ends_the_direction(win: WindowBundle,
                                                     images: Mapping[str, np.ndarray]) -> None:
    out = _confirm(win, images, _tracked(win.detections), no_bridger=True)
    track = out.track
    assert not any(o.source is ObservationSource.CSRT_BRIDGE for o in track.observations)
    after = [o for o in track.observations if _n(o.frame_id) > max(synth.GAP_FRAMES)]
    assert after and not any(o.accepted for o in after)
    assert any(a.gate == "bridge_unavailable" for a in out.audit)
    assert track.state is TrackState.CONFIRMED


def test_lost_bridge_ends_the_direction(win: WindowBundle,
                                        images: Mapping[str, np.ndarray]) -> None:
    out = _confirm(win, images, _tracked(win.detections), bridge={})
    assert any(a.gate == "bridge_lost" and a.frame_number == synth.GAP_FRAMES[0]
               for a in out.audit)
    after = [o for o in out.track.observations if _n(o.frame_id) > max(synth.GAP_FRAMES)]
    assert not any(o.accepted for o in after)


def test_rejected_bridge_box_ends_the_direction(win: WindowBundle,
                                                images: Mapping[str, np.ndarray]) -> None:
    far = {52: _shift(synth.detection_box(SCENE, 52), 150), 53: synth.detection_box(SCENE, 53)}
    track = _confirm(win, images, _tracked(win.detections), bridge=far).track
    (o,) = _obs(track, 52)
    assert o.source is ObservationSource.CSRT_BRIDGE and not o.accepted
    assert o.reason_code is ReasonCode.IDENTITY_GEOMETRY_MISMATCH
    assert not _obs(track, 53)


# ------------------------------------------------------------------------------------------------
# Identity gates
# ------------------------------------------------------------------------------------------------

PASSING = tr.IdentityMetrics(association_iou=0.9, center_step=0.01, scale=1.0, aspect=0.0,
                             appearance=0.99, competing_iou=0.0)


@pytest.mark.parametrize(("field", "inside", "outside", "key"), [
    ("association_iou", 0.30, 0.2999, "track.min_iou_for_association"),
    ("center_step", 0.25, 0.2501, "track.max_center_step_diag"),
    ("scale", 0.67, 0.6699, "track.box_scale_ratio_min"),
    ("scale", 1.50, 1.5001, "track.box_scale_ratio_max"),
    ("aspect", 0.20, 0.2001, "track.max_aspect_change"),
    ("appearance", 0.80, 0.7999, "track.min_hsv_hist_correlation"),
    ("competing_iou", 0.25, 0.2501, "track.max_competing_box_iou"),
])
def test_identity_gate_boundaries(field: str, inside: float, outside: float, key: str) -> None:
    ok = dataclasses.replace(PASSING, **{field: inside})
    bad = dataclasses.replace(PASSING, **{field: outside})
    assert tr.identity_gates(ok, CFG) is None
    failure = tr.identity_gates(bad, CFG)
    assert failure is not None and failure.gate == key and failure.observed == outside


def test_center_jump_is_rejected(win: WindowBundle, images: Mapping[str, np.ndarray]) -> None:
    n = 45
    jumped = [d.revise(bbox_px=_shift(d.bbox_px, 70)) if _n(d.frame_id) == n else d
              for d in win.detections]
    out = _confirm(win, images, _tracked(jumped), bridge=_gt_bridge([n, *synth.GAP_FRAMES]))
    (det_obs,) = _obs(out.track, n, ObservationSource.DETECTOR)
    assert not det_obs.accepted and det_obs.reason_code is ReasonCode.IDENTITY_GEOMETRY_MISMATCH
    (a,) = [a for a in out.audit if a.frame_number == n and a.source is ObservationSource.DETECTOR]
    assert a.gate == "track.max_center_step_diag"
    (bridge_obs,) = _obs(out.track, n, ObservationSource.CSRT_BRIDGE)
    assert bridge_obs.accepted
    assert all(o.accepted for o in out.track.observations if _n(o.frame_id) > n
               and o.source is ObservationSource.DETECTOR)


def test_appearance_change_is_rejected(win: WindowBundle,
                                       images: Mapping[str, np.ndarray]) -> None:
    n = 45
    det = next(d for d in win.detections if _n(d.frame_id) == n)

    def load(ref: FrameReference) -> np.ndarray:
        img = images[ref.frame_id]
        if ref.frame_number != n:
            return img
        x1, y1, x2, y2 = det.bbox_px
        out = img.copy()
        out[y1:y2, x1:x2] = (30, 30, 220)
        return out

    out = _confirm(win, images, _tracked(win.detections), load=load)
    (o,) = _obs(out.track, n)
    assert not o.accepted
    (a,) = [a for a in out.audit if a.frame_number == n
            and a.source is ObservationSource.DETECTOR]
    assert a.gate == "track.min_hsv_hist_correlation"


@pytest.mark.parametrize(("dx", "accepted"), [(132, True), (131, False)])
def test_competing_box_collision_boundary(win: WindowBundle, images: Mapping[str, np.ndarray],
                                          dx: int, accepted: bool) -> None:
    n = 45
    ours = next(d for d in win.detections if _n(d.frame_id) == n)
    rival = _extra_detection(win, n, _shift(ours.bbox_px, dx))
    assert tr.iou(rival.bbox_px, ours.bbox_px) == pytest.approx(0.25, abs=0.005)
    tracked = [*_tracked(win.detections), tr.TrackedDetection(rival, 7)]
    out = _confirm(win, images, tracked, bridge=_gt_bridge([n, *synth.GAP_FRAMES]))
    (o,) = _obs(out.track, n, ObservationSource.DETECTOR)
    assert o.accepted is accepted
    assert not any(o.detection_id == rival.detection_id for o in out.track.observations)
    if not accepted:
        (a,) = [a for a in out.audit if a.frame_number == n
                and a.source is ObservationSource.DETECTOR]
        assert a.gate == "track.max_competing_box_iou"


def test_bytetrack_id_switch_is_not_followed(win: WindowBundle,
                                             images: Mapping[str, np.ndarray]) -> None:
    switch = 45
    tracked = [tr.TrackedDetection(d, 2 if _n(d.frame_id) >= switch else 1)
               for d in win.detections]
    out = _confirm(win, images, tracked, bridge={})
    later = [o for o in out.track.observations if _n(o.frame_id) >= switch]
    assert later == []
    assert out.tracker_id == 1


def test_duplicate_same_id_box_is_rejected(win: WindowBundle,
                                           images: Mapping[str, np.ndarray]) -> None:
    n = 45
    ours = next(d for d in win.detections if _n(d.frame_id) == n)
    dup = _extra_detection(win, n, _shift(ours.bbox_px, 6))
    out = _confirm(win, images, [*_tracked(win.detections), tr.TrackedDetection(dup, 1)])
    by_id = {o.detection_id: o for o in _obs(out.track, n)}
    assert by_id[ours.detection_id].accepted
    assert not by_id[dup.detection_id].accepted


# ------------------------------------------------------------------------------------------------
# Seeds
# ------------------------------------------------------------------------------------------------


def test_analyst_roi_seed_without_detection(win: WindowBundle,
                                            images: Mapping[str, np.ndarray]) -> None:
    dets = [d for d in win.detections if d.frame_id != win.target_frame_id]
    roi = synth.detection_box(SCENE, synth.TARGET)
    req = _request(win, seed_detection=False, roi=roi)
    track = _confirm(win, images, _tracked(dets), request=req).track
    (seed,) = _obs(track, synth.TARGET)
    assert seed.source is ObservationSource.ANALYST_ROI and seed.detection_id is None
    assert seed.confidence == CFG.quality.analyst_seed_confidence == 0.50
    assert track.seed_detection_id is None
    assert track.detector_observation_count == win.track.detector_observation_count - 1
    assert track.state is TrackState.CONFIRMED


def test_analyst_roi_seed_confirmed_by_detection_in_seed_frame(
        win: WindowBundle, images: Mapping[str, np.ndarray]) -> None:
    roi = synth.detection_box(SCENE, synth.TARGET)
    req = _request(win, seed_detection=False, roi=(roi[0] + 4, roi[1] + 2, roi[2] - 4, roi[3]))
    track = _confirm(win, images, _tracked(win.detections), request=req).track
    (seed,) = _obs(track, synth.TARGET)
    assert seed.source is ObservationSource.DETECTOR
    assert track.seed_detection_id == seed.detection_id


def test_analyst_roi_on_empty_region_is_not_confirmed(win: WindowBundle,
                                                      images: Mapping[str, np.ndarray]) -> None:
    dets = [d for d in win.detections if d.frame_id != win.target_frame_id]
    req = _request(win, seed_detection=False, roi=(380, 10, 470, 40))
    out = _confirm(win, images, _tracked(dets), request=req, bridge={})
    assert out.tracker_id is None
    assert [o.source for o in out.track.observations] == [ObservationSource.ANALYST_ROI]
    assert out.track.state is TrackState.NOT_CONFIRMED


def test_colliding_seed_is_rejected(win: WindowBundle, images: Mapping[str, np.ndarray]) -> None:
    seed = next(d for d in win.detections if d.frame_id == win.target_frame_id)
    rival = _extra_detection(win, synth.TARGET, _shift(seed.bbox_px, 20))
    out = _confirm(win, images, [*_tracked(win.detections), tr.TrackedDetection(rival, 9)])
    assert out.track.state is TrackState.NOT_CONFIRMED
    assert ReasonCode.IDENTITY_GEOMETRY_MISMATCH in out.track.reason_codes
    assert len(out.track.observations) == 1 and not out.track.observations[0].accepted


# ------------------------------------------------------------------------------------------------
# Input isolation
# ------------------------------------------------------------------------------------------------


def test_detection_from_another_video_is_rejected(win: WindowBundle,
                                                  images: Mapping[str, np.ndarray]) -> None:
    other = seeded_uuid7("test:other-video", 1_760_000_000_000)
    d = win.detections[0]
    foreign = d.revise(video_id=other, frame_id=frame_id(other, _n(d.frame_id)))
    with pytest.raises(ValidationFailed):
        _confirm(win, images, [*_tracked(win.detections), tr.TrackedDetection(foreign, 1)])


def test_frame_from_another_video_is_rejected(win: WindowBundle,
                                              images: Mapping[str, np.ndarray]) -> None:
    other = seeded_uuid7("test:other-video", 1_760_000_000_000)
    f = win.frames[0]
    foreign = f.revise(video_id=other, frame_id=frame_id(other, f.frame_number),
                       lossless_png_uri=f"derived/{other}/frames/f0/frame.png")
    with pytest.raises(ValidationFailed):
        _confirm(win, images, _tracked(win.detections), frames=[*win.frames, foreign])


def test_seed_detection_must_be_in_seed_frame(win: WindowBundle,
                                              images: Mapping[str, np.ndarray]) -> None:
    other = next(d for d in win.detections if _n(d.frame_id) == 30)
    req = _request(win).model_copy(update={"seed_detection_id": other.detection_id})
    with pytest.raises(ValidationFailed):
        _confirm(win, images, _tracked(win.detections), request=req)


def test_tracker_rejects_baseline_assets(win: WindowBundle,
                                         images: Mapping[str, np.ndarray]) -> None:
    baseline = AssetRef(
        asset_id=seeded_uuid7("test:baseline", 1_760_000_000_000), case_id=win.case_id,
        kind=AssetKind.BASELINE, storage_uri=f"derived/{win.video_id}/baseline/x/out.png",
        media_type="image/png", byte_length=1, sha256="0" * 64, non_evidentiary=True,
        created_at="2026-10-09T17:00:00Z")
    with pytest.raises(NonEvidentiaryInput):
        _confirm(win, images, _tracked(win.detections), frames=[*win.frames, baseline])


# ------------------------------------------------------------------------------------------------
# Window and determinism
# ------------------------------------------------------------------------------------------------


def test_window_uses_the_narrower_radius(win: WindowBundle,
                                         images: Mapping[str, np.ndarray]) -> None:
    req = _request(win, radius_us=1_000_000)
    track = _confirm(win, images, _tracked(win.detections), request=req).track
    seed_pts = win.frame(synth.TARGET).pts_us
    assert track.window_end_us - seed_pts == 1_000_000
    assert all(abs(o.pts_us - seed_pts) <= 1_000_000 for o in track.observations)


def test_sampling_caps_rate_but_keeps_seed_and_detection_frames() -> None:
    vid = seeded_uuid7("test:30fps", 1_760_000_000_000)
    frames = [FrameReference.create(
        frame_id=frame_id(vid, n), video_id=vid, frame_number=n, pts_us=n * 1_000_000 // 30,
        is_keyframe=n == 0, width_px=64, height_px=48) for n in range(121)]
    seed = frames[61]
    with_dets = {frames[62].frame_id, frames[64].frame_id}
    sampled = tr.sample_window(frames, seed, 0, 4_000_000, 15, with_dets)
    numbers = [f.frame_number for f in sampled]
    assert 61 in numbers and 62 in numbers and 64 in numbers
    assert all((n - 61) % 2 == 0 or n in (62, 64) for n in numbers)
    assert len(numbers) == 60 + 2
    for fps, stride in ((29.97, 2), (60, 4), (15, 1), (15.1, 2)):
        dt = 1_000_000 / fps
        refs = [f.revise(pts_us=round(f.frame_number * dt)) for f in frames]
        got = tr.sample_window(refs, refs[60], 0, 10_000_000, 15, set())
        assert {f.frame_number % stride for f in got} == {0}, fps
        assert len(got) == len(range(0, 121, stride)), fps


def test_tracking_is_independent_of_detection_order(win: WindowBundle,
                                                    images: Mapping[str, np.ndarray]) -> None:
    tracked = _tracked(win.detections)
    shuffled = tracked[:]
    random.Random(7).shuffle(shuffled)
    a = _confirm(win, images, tracked).track
    b = _confirm(win, images, shuffled).track
    assert json_bytes(a) == json_bytes(b)


@pytest.mark.skipif(not tr.csrt_available(), reason="OpenCV CSRT needs opencv-contrib")
def test_real_csrt_bridger_follows_the_plate(win: WindowBundle,
                                             images: Mapping[str, np.ndarray]) -> None:
    cv2.setNumThreads(1)
    bridger = tr.CsrtBridger()
    bridger.start(37, images[win.frame(37).frame_id], synth.detection_box(SCENE, 37))
    for n in (38, 39, 40):
        box = bridger.step(n, images[win.frame(n).frame_id])
        assert box is not None and tr.iou(box, synth.detection_box(SCENE, n)) >= 0.7
