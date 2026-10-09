"""Pre-alignment hard gates: track, target, and donor preflight with rule-coded decisions."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from probity.domain.enums import (
    AssetKind,
    ObservationSource,
    PolicyOutcome,
    ReasonCode,
    TrackState,
)
from probity.domain.errors import JobCancelled
from probity.domain.ids import frame_id, parse_frame_id
from probity.domain.models import AssetRef, Detection, FrameReference, Track, TrackObservation
from probity.domain.policy import default_policy
from probity.eval import synth
from probity.eval.window import WindowBundle, load_window
from probity.reconstruction import preflight as pf
from probity.reconstruction.decisions import DecisionLog
from probity.reconstruction.determinism import FixedClock, seeded_uuid7
from probity.reconstruction.io import FrameIntegrityError, json_bytes, load_frame
from probity.reconstruction.types import QualityScores

REPO = Path(__file__).resolve().parents[3]
SYNTH = REPO / "fixtures" / "synthetic"
CFG = default_policy()


@pytest.fixture(scope="module")
def win() -> WindowBundle:
    return load_window(SYNTH / "plate_translate_v1")


@pytest.fixture(scope="module")
def images(win: WindowBundle) -> dict[str, np.ndarray]:
    return {ref.frame_id: load_frame(ref, win.resolver) for ref in win.frames}


def _n(fid: str) -> int:
    return parse_frame_id(fid)[1]


def _log() -> DecisionLog:
    return DecisionLog(seeded_uuid7("test:preflight:run", 1_760_000_000_000),
                       FixedClock("2026-10-09T17:10:00.000000Z"))


def _by_frame(dets: Sequence[Detection]) -> dict[str, list[Detection]]:
    out: dict[str, list[Detection]] = {}
    for d in dets:
        out.setdefault(d.frame_id, []).append(d)
    return out


def _run(win: WindowBundle, images: Mapping[str, np.ndarray], log: DecisionLog, *,
         track: Track | None = None, video_id: str | None = None, case_id: str | None = None,
         target_box: tuple[int, int, int, int] | None = None, target_frame_id: str | None = None,
         frames: Mapping[str, object] | None = None,
         load: Callable[[FrameReference], np.ndarray] | None = None,
         detections: Sequence[Detection] | None = None,
         candidates: Sequence[TrackObservation] | None = None,
         cancel: object | None = None) -> pf.PreflightResult:
    return pf.run_preflight(
        log, track=track or win.track, video_id=video_id or win.video_id,
        case_id=case_id or win.case_id, target_frame_id=target_frame_id or win.target_frame_id,
        target_box=target_box or win.target_bbox_px,
        frames=frames or {r.frame_id: r for r in win.frames},  # type: ignore[arg-type]
        load=load or (lambda ref: images[ref.frame_id]),
        detections=_by_frame(win.detections if detections is None else detections), cfg=CFG,
        cancel=cancel, candidates=candidates)  # type: ignore[arg-type]


def _rows(log: DecisionLog, subject: str, outcome: PolicyOutcome | None = None) -> list:
    return [r for r in log.rows if r.subject_ref == subject
            and (outcome is None or r.outcome is outcome)]


def _fid(win: WindowBundle, n: int) -> str:
    return frame_id(win.video_id, n)


# ------------------------------------------------------------------------------------------------
# Synthetic window outcomes
# ------------------------------------------------------------------------------------------------


def test_translate_window_preflight(win: WindowBundle, images: Mapping[str, np.ndarray]) -> None:
    log = _log()
    res = _run(win, images, log)
    assert res.refusal is None and res.target is not None
    accepted = [d.frame_number for d in res.donors]
    assert {synth.LEFT_SHARP_FRAME, synth.RIGHT_SHARP_FRAME, *synth.MILD_FRAMES} <= set(accepted)
    (obstructed,) = _rows(log, _fid(win, synth.OBSTRUCTED_FRAME))
    assert obstructed.rule_code is ReasonCode.DONOR_OBSTRUCTED
    assert obstructed.outcome is PolicyOutcome.REJECT
    assert (obstructed.observed, obstructed.operator, obstructed.threshold) == (0.3, "<=", 0.15)
    (dark,) = _rows(log, _fid(win, synth.INCOMPATIBLE_FRAME))
    assert dark.rule_code is ReasonCode.LIGHTING_OUT_OF_RANGE and dark.units == "stops"
    for n in synth.GAP_FRAMES:
        (bridged,) = _rows(log, _fid(win, n))
        assert bridged.rule_code is ReasonCode.IDENTITY_GEOMETRY_MISMATCH
        assert bridged.observed == "CSRT_BRIDGE"


def test_single_donor_window_keeps_exactly_one(images: Mapping[str, np.ndarray]) -> None:
    w = load_window(SYNTH / "plate_single_donor_v1")
    imgs = {ref.frame_id: load_frame(ref, w.resolver) for ref in w.frames}
    res = _run(w, imgs, _log())
    assert [d.frame_number for d in res.donors] == [synth.SINGLE_DONOR_FRAME]


def test_every_decision_is_rule_coded(win: WindowBundle, images: Mapping[str, np.ndarray]) -> None:
    log = _log()
    res = _run(win, images, log)
    assert [r.sequence for r in log.rows] == list(range(len(log.rows)))
    for r in log.rows:
        assert r.rule_code is not None and r.operator != "none"
        assert r.observed is not None and r.threshold is not None
    known = set(log.ids)
    for d in res.donors:
        assert d.decision_ids and set(d.decision_ids) <= known
        assert all(r.outcome is PolicyOutcome.ACCEPT for r in log.rows
                   if r.decision_id in d.decision_ids)


def test_preflight_is_deterministic(win: WindowBundle, images: Mapping[str, np.ndarray]) -> None:
    a, b = _log(), _log()
    _run(win, images, a)
    _run(win, images, b)
    assert json_bytes(list(a.rows)) == json_bytes(list(b.rows))


# ------------------------------------------------------------------------------------------------
# Track and target refusals
# ------------------------------------------------------------------------------------------------


def test_unconfirmed_track_refuses(win: WindowBundle, images: Mapping[str, np.ndarray]) -> None:
    track = win.track.revise(state=TrackState.NOT_CONFIRMED, confirmed=False,
                             reason_codes=(ReasonCode.TRACK_NOT_CONFIRMED,))
    log = _log()
    res = _run(win, images, log, track=track)
    assert res.refusal is ReasonCode.TRACK_NOT_CONFIRMED and res.donors == ()
    (row,) = [r for r in log.rows if r.outcome is PolicyOutcome.REJECT]
    assert row.rule_code is ReasonCode.TRACK_NOT_CONFIRMED


@pytest.mark.parametrize("field", ["video_id", "case_id"])
def test_track_from_another_video_or_case_refuses(win: WindowBundle,
                                                  images: Mapping[str, np.ndarray],
                                                  field: str) -> None:
    other = seeded_uuid7(f"test:other:{field}", 1_760_000_000_000)
    log = _log()
    res = _run(win, images, log, **{field: other})
    assert res.refusal is ReasonCode.DONOR_TRACK_MISMATCH and res.target is None


def test_target_frame_outside_track_refuses(win: WindowBundle,
                                            images: Mapping[str, np.ndarray]) -> None:
    res = _run(win, images, _log(), target_frame_id=_fid(win, 0))
    assert res.refusal is ReasonCode.TRACK_NOT_CONFIRMED


def test_bridged_only_target_refuses(win: WindowBundle, images: Mapping[str, np.ndarray]) -> None:
    n = synth.GAP_FRAMES[0]
    box = synth.detection_box(synth.SceneSpec(), n)
    res = _run(win, images, _log(), target_frame_id=_fid(win, n), target_box=box)
    assert res.refusal is ReasonCode.IDENTITY_GEOMETRY_MISMATCH


def test_target_box_must_match_tracked_box(win: WindowBundle,
                                           images: Mapping[str, np.ndarray]) -> None:
    x1, y1, x2, y2 = win.target_bbox_px
    res = _run(win, images, _log(), target_box=(x1 + 180, y1, x2 + 180, y2))
    assert res.refusal is ReasonCode.IDENTITY_GEOMETRY_MISMATCH


TARGET_Q = QualityScores(D=0.6, S=0.4, Z=0.8, E=1.0, O=1.0, P=1.0, Q=0.5)


@pytest.mark.parametrize(("box", "quality", "code"), [
    ((0, 0, 48, 20), 0.15, None),
    ((0, 0, 47, 20), 0.15, ReasonCode.TARGET_TOO_SMALL),
    ((0, 0, 48, 19), 0.15, ReasonCode.TARGET_TOO_SMALL),
    ((0, 0, 48, 20), 0.1499, ReasonCode.TARGET_QUALITY_TOO_LOW),
])
def test_target_gate_boundaries(box: tuple[int, int, int, int], quality: float,
                                code: ReasonCode | None) -> None:
    gates = pf.target_quality_gates(box, dataclasses.replace(TARGET_Q, Q=quality), CFG)
    failures = [g for g in gates if not g.ok]
    assert [g.failure_code for g in failures] == ([code] if code else [])


# ------------------------------------------------------------------------------------------------
# Donor gate boundaries (pure)
# ------------------------------------------------------------------------------------------------


def _measures() -> tuple[pf.SubjectMeasure, pf.DonorMeasure]:
    img = np.zeros((4, 4, 3), np.uint8)
    vid = seeded_uuid7("test:pure", 1_760_000_000_000)
    ref = FrameReference.create(frame_id=frame_id(vid, 1), video_id=vid, frame_number=1,
                                pts_us=1, is_keyframe=True, width_px=4, height_px=4)
    obs = TrackObservation(frame_id=ref.frame_id, pts_us=1, bbox_px=(0, 0, 2, 2),
                           source=ObservationSource.DETECTOR,
                           detection_id=seeded_uuid7("test:det", 1_760_000_000_000),
                           confidence=0.8, accepted=True)
    target = pf.SubjectMeasure(obs, ref, img, (0, 0, 2, 2), 0.0, TARGET_Q)
    donor = pf.DonorMeasure(obs, ref, img, (0, 0, 2, 2), 0.0,
                            dataclasses.replace(TARGET_Q, Q=0.70), dt_s=0.5, scale=1.0,
                            aspect_change=0.0, exposure_stops=0.0, decision_ids=())
    return target, donor


@pytest.mark.parametrize(("field", "inside", "outside", "code", "key"), [
    ("dt_s", 2.0, 2.0001, ReasonCode.DONOR_OUTSIDE_WINDOW, "track.window_radius_s"),
    ("dt_s", -2.0, -2.0001, ReasonCode.DONOR_OUTSIDE_WINDOW, "track.window_radius_s"),
    ("scale", 0.67, 0.6699, ReasonCode.IDENTITY_GEOMETRY_MISMATCH, "track.box_scale_ratio_min"),
    ("scale", 1.50, 1.5001, ReasonCode.IDENTITY_GEOMETRY_MISMATCH, "track.box_scale_ratio_max"),
    ("aspect_change", 0.20, 0.2001, ReasonCode.IDENTITY_GEOMETRY_MISMATCH,
     "track.max_aspect_change"),
    ("occluded", 0.15, 0.1501, ReasonCode.DONOR_OBSTRUCTED, "donor.max_occluded_fraction"),
    ("exposure_stops", 0.75, 0.7501, ReasonCode.LIGHTING_OUT_OF_RANGE,
     "donor.max_exposure_delta_stops"),
])
def test_donor_gate_boundaries(field: str, inside: float, outside: float, code: ReasonCode,
                               key: str) -> None:
    target, donor = _measures()
    ok = pf.donor_measured_gates(dataclasses.replace(donor, **{field: inside}), target, CFG)
    assert all(g.ok for g in ok)
    bad = pf.donor_measured_gates(dataclasses.replace(donor, **{field: outside}), target, CFG)
    (failed,) = [g for g in bad if not g.ok]
    assert failed.failure_code is code and failed.policy_key == key


@pytest.mark.parametrize(("donor_q", "ok", "key"), [
    (0.6001, True, None),
    (0.5999, False, "donor.min_quality_gain"),
    (0.55, False, "donor.min_quality_gain"),
])
def test_donor_quality_gain_boundary(donor_q: float, ok: bool, key: str | None) -> None:
    target, donor = _measures()
    gates = pf.donor_measured_gates(
        dataclasses.replace(donor, quality=dataclasses.replace(TARGET_Q, Q=donor_q)),
        dataclasses.replace(target, quality=dataclasses.replace(TARGET_Q, Q=0.50)), CFG)
    failed = [g for g in gates if not g.ok]
    assert (not failed) is ok
    if failed:
        assert failed[0].failure_code is ReasonCode.DONOR_NOT_CLEARER
        assert failed[0].policy_key == key


def test_donor_quality_floor_boundary() -> None:
    target, donor = _measures()
    low_target = dataclasses.replace(target, quality=dataclasses.replace(TARGET_Q, Q=0.30))
    for q_value, ok in ((0.55, True), (0.5499, False)):
        gates = pf.donor_measured_gates(
            dataclasses.replace(donor, quality=dataclasses.replace(TARGET_Q, Q=q_value)),
            low_target, CFG)
        failed = [g for g in gates if not g.ok]
        assert (not failed) is ok
        if failed:
            assert failed[0].policy_key == "donor.min_quality"


# ------------------------------------------------------------------------------------------------
# Lineage: never borrow across tracks or videos
# ------------------------------------------------------------------------------------------------


def test_donor_from_another_video_is_rejected_before_loading(
        win: WindowBundle, images: Mapping[str, np.ndarray]) -> None:
    other = seeded_uuid7("test:foreign-video", 1_760_000_000_000)
    foreign = TrackObservation(
        frame_id=frame_id(other, 40), pts_us=win.frame(40).pts_us, bbox_px=win.target_bbox_px,
        source=ObservationSource.DETECTOR, detection_id=seeded_uuid7("test:fd", 1), confidence=0.9,
        accepted=True)
    loaded: list[str] = []

    def load(ref: FrameReference) -> np.ndarray:
        loaded.append(ref.frame_id)
        return images[ref.frame_id]

    log = _log()
    res = _run(win, images, log, candidates=[foreign], load=load)
    assert res.donors == ()
    (row,) = _rows(log, foreign.frame_id)
    assert row.rule_code is ReasonCode.DONOR_TRACK_MISMATCH and row.observed == other
    assert foreign.frame_id not in loaded


def test_donor_from_another_track_is_rejected(win: WindowBundle,
                                              images: Mapping[str, np.ndarray]) -> None:
    real = next(o for o in win.track.observations if _n(o.frame_id) == synth.LEFT_SHARP_FRAME)
    impostor = real.model_copy(update={"detection_id": seeded_uuid7("test:other-track", 2)})
    log = _log()
    res = _run(win, images, log, candidates=[impostor, real])
    assert [d.frame_number for d in res.donors] == [synth.LEFT_SHARP_FRAME]
    rejects = _rows(log, real.frame_id, PolicyOutcome.REJECT)
    assert [r.rule_code for r in rejects] == [ReasonCode.DONOR_TRACK_MISMATCH]
    assert rejects[0].observed == "NOT_MEMBER"


def test_rejected_track_observation_cannot_donate(win: WindowBundle,
                                                  images: Mapping[str, np.ndarray]) -> None:
    real = next(o for o in win.track.observations if _n(o.frame_id) == synth.LEFT_SHARP_FRAME)
    rejected = real.model_copy(update={"accepted": False,
                                       "reason_code": ReasonCode.IDENTITY_GEOMETRY_MISMATCH})
    res = _run(win, images, _log(), candidates=[rejected])
    assert res.donors == ()


# ------------------------------------------------------------------------------------------------
# Generated / tampered inputs
# ------------------------------------------------------------------------------------------------


def _baseline_asset(win: WindowBundle) -> AssetRef:
    return AssetRef(
        asset_id=seeded_uuid7("test:baseline-asset", 1_760_000_000_000), case_id=win.case_id,
        kind=AssetKind.BASELINE, storage_uri=f"derived/{win.video_id}/baseline/r/out.png",
        media_type="image/png", byte_length=1, sha256="0" * 64, non_evidentiary=True,
        created_at="2026-10-09T17:00:00Z")


def test_reconstructor_refuses_baseline_as_target(win: WindowBundle,
                                                  images: Mapping[str, np.ndarray]) -> None:
    frames: dict[str, object] = {r.frame_id: r for r in win.frames}
    frames[win.target_frame_id] = _baseline_asset(win)
    log = _log()
    res = _run(win, images, log, frames=frames)
    assert res.refusal is ReasonCode.GENERATED_INPUT_REJECTED
    assert log.rows[-1].rule_code is ReasonCode.GENERATED_INPUT_REJECTED


def test_reconstructor_rejects_baseline_as_donor(win: WindowBundle,
                                                 images: Mapping[str, np.ndarray]) -> None:
    donor_fid = _fid(win, synth.LEFT_SHARP_FRAME)
    frames: dict[str, object] = {r.frame_id: r for r in win.frames}
    frames[donor_fid] = _baseline_asset(win)
    log = _log()
    res = _run(win, images, log, frames=frames)
    assert synth.LEFT_SHARP_FRAME not in [d.frame_number for d in res.donors]
    (row,) = _rows(log, donor_fid)
    assert row.rule_code is ReasonCode.GENERATED_INPUT_REJECTED


def test_tampered_donor_pixels_are_rejected(win: WindowBundle,
                                            images: Mapping[str, np.ndarray]) -> None:
    def load(ref: FrameReference) -> np.ndarray:
        if ref.frame_number == synth.RIGHT_SHARP_FRAME:
            raise FrameIntegrityError("pixel_sha256 mismatch")
        return images[ref.frame_id]

    log = _log()
    res = _run(win, images, log, load=load)
    assert synth.RIGHT_SHARP_FRAME not in [d.frame_number for d in res.donors]
    (row,) = _rows(log, _fid(win, synth.RIGHT_SHARP_FRAME))
    assert row.rule_code is ReasonCode.DECODE_NOT_DETERMINISTIC


# ------------------------------------------------------------------------------------------------
# Occlusion measure and cancellation
# ------------------------------------------------------------------------------------------------


def _det(win: WindowBundle, box: tuple[int, int, int, int], label: str,
         occlusion: float | None = None) -> Detection:
    base = win.detections[0]
    return base.revise(detection_id=seeded_uuid7(f"test:occ:{label}", 3), bbox_px=box,
                       bbox_norm=(box[0] / 480, box[1] / 288, box[2] / 480, box[3] / 288),
                       occlusion_score=occlusion)


def test_occluded_fraction_counts_foreground_boxes_and_frame_edges(win: WindowBundle) -> None:
    box = (100, 100, 200, 140)
    own = _det(win, box, "own")
    assert pf.occluded_fraction(box, 480, 288, own, [own]) == 0.0
    half = _det(win, (150, 90, 260, 150), "half")
    assert pf.occluded_fraction(box, 480, 288, own, [own, half]) == pytest.approx(0.5)
    carrier = _det(win, (50, 50, 300, 250), "car")
    assert pf.occluded_fraction(box, 480, 288, own, [own, carrier]) == 0.0
    assert pf.occluded_fraction((-20, 100, 80, 140), 480, 288, None, []) == pytest.approx(0.2)
    reported = _det(win, box, "reported", occlusion=0.4)
    assert pf.occluded_fraction(box, 480, 288, reported, [reported]) == pytest.approx(0.4)


def test_overlapping_detection_obstructs_a_donor(win: WindowBundle,
                                                 images: Mapping[str, np.ndarray]) -> None:
    n = synth.LEFT_SHARP_FRAME
    obs = next(o for o in win.track.observations if _n(o.frame_id) == n)
    x1, y1, x2, y2 = obs.bbox_px
    occluder = _det(win, (x1, y1 - 10, x1 + (x2 - x1) // 4, y2 + 10), "pole").revise(
        frame_id=obs.frame_id, pts_us=obs.pts_us)
    log = _log()
    res = _run(win, images, log, detections=[*win.detections, occluder])
    assert n not in [d.frame_number for d in res.donors]
    (row,) = _rows(log, obs.frame_id)
    assert row.rule_code is ReasonCode.DONOR_OBSTRUCTED and row.observed > 0.15


def test_cancellation_is_checked_per_donor(win: WindowBundle,
                                           images: Mapping[str, np.ndarray]) -> None:
    class Cancel:
        calls = 0

        def is_cancelled(self) -> bool:
            return self.calls >= 3

        def checkpoint(self) -> None:
            self.calls += 1
            if self.is_cancelled():
                raise JobCancelled("cancelled")

    token = Cancel()
    with pytest.raises(JobCancelled):
        _run(win, images, _log(), cancel=token)
    assert token.calls == 3
