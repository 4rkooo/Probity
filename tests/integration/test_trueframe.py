"""Integrator: TrueFrame over committed synthetic windows (section 9)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from probity.domain.enums import (
    ObservationSource,
    PolicyOutcome,
    PolicyStage,
    ProvenanceClass,
    ReasonCode,
    ReconstructionState,
)
from probity.domain.ids import parse_frame_id
from probity.domain.models import (
    PixelProvenance,
    PolicyDecision,
    ReconstructionRun,
    integrity_score_0_100,
)
from probity.domain.policy import default_policy
from probity.eval import synth
from probity.eval.window import load_window
from probity.reconstruction.integrity import audit_completeness
from probity.reconstruction.io import load_frame, read_json, read_png
from probity.reconstruction.provenance import decode_provenance, unsupported_changed_pixel_rate
from probity.reconstruction.trueframe import NeverCancelled, TrueFrame, request_from_window
from probity.reconstruction.types import ProvenanceArrays

REPO = Path(__file__).resolve().parents[2]
SYNTH = REPO / "fixtures" / "synthetic"
COMPLETED = SYNTH / "plate_translate_v1" / "reconstructions" / "completed"
CFG = default_policy()


def _window(name: str):
    return load_window(SYNTH / name)


def _run(window, *, label: str = "trueframe", donor_candidates=None):
    engine = TrueFrame.from_window(window, CFG)
    return engine.run(
        request_from_window(window, CFG, run_label=label),
        NeverCancelled(),
        donor_candidates=donor_candidates,
    )


def _obs_at(window, frame_number: int):
    return next(
        o for o in window.track.observations
        if parse_frame_id(o.frame_id)[1] == frame_number
        and o.accepted and o.source is ObservationSource.DETECTOR
    )


def _reject_for_frame(decisions, frame_number: int):
    return [
        d for d in decisions
        if d.outcome is PolicyOutcome.REJECT and d.subject_ref.endswith(f":f{frame_number}")
    ]


def test_golden_window_aligns_at_least_two_same_track_donors() -> None:
    """Designed sharp donors (same track/video, never the target) pass AKAZE.

    Bounded color then rejects them (flat car-body ring, ill-conditioned IRLS; NOTES #52),
    so this window currently REFUSES with INSUFFICIENT_COMPATIBLE_DONORS rather than borrowing.
    Gates are not loosened.
    """
    window = _window("plate_translate_v1")
    built = _run(window)
    run = built.result.run
    detector_frames = {
        o.frame_id for o in window.track.observations
        if o.accepted and o.source is ObservationSource.DETECTOR
    }
    aligned = {
        d.subject_ref
        for d in built.result.decisions
        if d.stage is PolicyStage.ALIGN
        and d.outcome is PolicyOutcome.ACCEPT
        and d.rule_code is ReasonCode.ALIGNMENT_AKAZE_ACCEPTED
        and d.policy_key == "alignment.min_valid_coverage"
    }
    assert len(aligned) >= CFG.donor.min_count
    for fid in aligned:
        assert parse_frame_id(fid)[0] == window.video_id
        assert fid in detector_frames
        assert fid != run.target_frame_id
    assert run.state is ReconstructionState.REFUSED
    assert run.refusal_reasons == (ReasonCode.INSUFFICIENT_COMPATIBLE_DONORS,)
    color_rejects = [
        d for d in built.result.decisions
        if d.stage is PolicyStage.COLOR and d.outcome is PolicyOutcome.REJECT
        and d.rule_code is ReasonCode.LIGHTING_OUT_OF_RANGE
        and d.policy_key == "color.gain_min"
    ]
    assert len(color_rejects) >= CFG.donor.min_count


def test_committed_golden_has_zero_unsupported_changed_pixels() -> None:
    """Hand-built completed golden: >=2 same-track donors, winner-take-all, no unsupported pixels.

    TrueFrame itself does not yet emit a SUCCEEDED run on this window (NOTES #52).
    """
    window = _window("plate_translate_v1")
    run = ReconstructionRun.model_validate(read_json(COMPLETED / "run.json"))
    arrays = decode_provenance((COMPLETED / "provenance.npz").read_bytes())
    target = load_frame(window.frame(parse_frame_id(run.target_frame_id)[1]), window.resolver)
    result = read_png(COMPLETED / "result.png")
    assert unsupported_changed_pixel_rate(result, target, arrays) == 0.0
    x1, y1, x2, y2 = run.target_bbox_px
    tile = CFG.fusion.tile_px
    used: set[int] = set()
    for ty in range(y1, y2, tile):
        for tx in range(x1, x2, tile):
            cls = arrays.cls[ty:min(ty + tile, y2), tx:min(tx + tile, x2)]
            idx = arrays.source_index[ty:min(ty + tile, y2), tx:min(tx + tile, x2)]
            assert int(np.unique(cls).size) == 1 and int(np.unique(idx).size) == 1
            if int(cls.flat[0]) == int(ProvenanceClass.BORROWED):
                used.add(int(idx.flat[0]))
    assert len(used) >= 2
    outside = np.ones(arrays.cls.shape, dtype=bool)
    outside[y1:y2, x1:x2] = False
    assert (arrays.cls[outside] == ProvenanceClass.ORIGINAL).all()


def test_obstructed_window_returns_refused_with_donor_obstructed() -> None:
    window = _window("plate_translate_v1")
    built = _run(
        window, label="obstructed",
        donor_candidates=(_obs_at(window, synth.OBSTRUCTED_FRAME),),
    )
    run = built.result.run
    assert run.state is ReconstructionState.REFUSED
    assert ReasonCode.INSUFFICIENT_COMPATIBLE_DONORS in run.refusal_reasons
    rejected = _reject_for_frame(built.result.decisions, synth.OBSTRUCTED_FRAME)
    assert rejected and rejected[0].rule_code is ReasonCode.DONOR_OBSTRUCTED


def test_incompatible_window_returns_refused_with_lighting_code() -> None:
    window = _window("plate_translate_v1")
    built = _run(
        window, label="incompatible",
        donor_candidates=(_obs_at(window, synth.INCOMPATIBLE_FRAME),),
    )
    run = built.result.run
    assert run.state is ReconstructionState.REFUSED
    assert ReasonCode.INSUFFICIENT_COMPATIBLE_DONORS in run.refusal_reasons
    rejected = _reject_for_frame(built.result.decisions, synth.INCOMPATIBLE_FRAME)
    assert rejected and rejected[0].rule_code is ReasonCode.LIGHTING_OUT_OF_RANGE


def test_single_donor_window_returns_refused_with_insufficient() -> None:
    window = _window("plate_single_donor_v1")
    built = _run(window)
    run = built.result.run
    assert run.state is ReconstructionState.REFUSED
    assert run.refusal_reasons == (ReasonCode.INSUFFICIENT_COMPATIBLE_DONORS,)
    assert run.result_png_uri is None and run.provenance is None and run.integrity is None
    (gate,) = [
        d for d in built.result.decisions
        if d.rule_code is ReasonCode.INSUFFICIENT_COMPATIBLE_DONORS
    ]
    assert gate.outcome is PolicyOutcome.REJECT
    assert gate.policy_key == "donor.min_count" and gate.units == "donors"
    assert gate.observed < gate.threshold


def test_integrity_matches_golden_value_and_is_order_invariant() -> None:
    run = ReconstructionRun.model_validate(read_json(COMPLETED / "run.json"))
    decisions = tuple(PolicyDecision.model_validate(r) for r in read_json(COMPLETED / "decisions.json"))
    prov = PixelProvenance.model_validate(read_json(COMPLETED / "provenance.json"))
    raw = decode_provenance((COMPLETED / "provenance.npz").read_bytes())
    arrays = ProvenanceArrays(
        cls=raw.cls, source_index=raw.source_index, source_x=raw.source_x,
        source_y=raw.source_y, lut=prov.source_lut,
    )
    integrity = run.integrity
    assert integrity is not None
    golden = integrity_score_0_100(
        supported_coverage=integrity.supported_coverage,
        mean_source_confidence=integrity.mean_source_confidence,
        mean_alignment_confidence=integrity.mean_alignment_confidence,
        track_continuity=integrity.track_continuity,
        audit_completeness=integrity.audit_completeness,
        semantic_generated_pct=integrity.semantic_generated_pct,
    )
    assert integrity.score_0_100 == golden
    assert integrity.semantic_generated_pct == 0.0
    assert integrity.score_0_100 >= CFG.integrity.min_score
    first_au = audit_completeness(decisions, arrays)
    shuffled = list(decisions)
    random.Random(20261009).shuffle(shuffled)
    assert audit_completeness(tuple(shuffled), arrays) == first_au
    assert first_au == integrity.audit_completeness
