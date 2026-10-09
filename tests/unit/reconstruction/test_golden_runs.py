"""Step-1 golden runs (HAND-BUILT; step 8 regenerates them from the real pipeline)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from probity.domain.enums import (
    AssetKind,
    ObservationSource,
    PolicyOutcome,
    ProvenanceClass,
    ReasonCode,
    ReconstructionState,
    SourceRole,
)
from probity.domain.ids import parse_frame_id
from probity.domain.models import (
    AssetRef,
    PixelProvenance,
    PolicyDecision,
    ReconstructionRun,
    integrity_score_0_100,
)
from probity.domain.policy import default_policy
from probity.eval.handbuilt_runs import build_handbuilt_run
from probity.eval.window import WindowBundle, load_truth, load_window
from probity.reconstruction.io import (
    array_sha256,
    file_sha256,
    json_bytes,
    load_frame,
    pixel_sha256,
    read_json,
    read_png,
)
from probity.reconstruction.provenance import (
    decode_provenance,
    unsupported_changed_pixel_rate,
    validate_arrays,
)

REPO = Path(__file__).resolve().parents[3]
SYNTH = REPO / "fixtures" / "synthetic"
COMPLETED = SYNTH / "plate_translate_v1" / "reconstructions" / "completed"
REFUSED = SYNTH / "plate_single_donor_v1" / "reconstructions" / "refused"
CFG = default_policy()


def _run(d: Path) -> ReconstructionRun:
    return ReconstructionRun.model_validate(read_json(d / "run.json"))


def _decisions(d: Path) -> list[PolicyDecision]:
    return [PolicyDecision.model_validate(r) for r in read_json(d / "decisions.json")]


@pytest.fixture(scope="module")
def window() -> WindowBundle:
    return load_window(SYNTH / "plate_translate_v1")


@pytest.fixture(scope="module")
def completed() -> tuple[ReconstructionRun, list[PolicyDecision], PixelProvenance]:
    prov = PixelProvenance.model_validate(read_json(COMPLETED / "provenance.json"))
    return _run(COMPLETED), _decisions(COMPLETED), prov


# ------------------------------------------------------------------------------------------------
# Completed run
# ------------------------------------------------------------------------------------------------


def test_completed_run_is_schema_valid_and_succeeded(completed: tuple) -> None:
    run, decisions, prov = completed
    assert run.state is ReconstructionState.SUCCEEDED
    assert run.iteration_count <= CFG.iteration.max_passes
    assert len(run.accepted_donor_frame_ids) >= CFG.donor.min_count
    assert prov.run_id == run.run_id and prov.coverage_complete
    assert prov.coverage_counts.GENERATED_BLEND == 0
    assert run.provenance is not None and run.provenance.source_lut == prov.source_lut


def test_artifact_hashes_agree(completed: tuple) -> None:
    run, _, prov = completed
    npz_sha = file_sha256(COMPLETED / "provenance.npz")
    assert npz_sha == run.provenance_sha256 == prov.artifact_sha256
    assert file_sha256(COMPLETED / "result.png") == run.result_png_sha256
    assets = [AssetRef.model_validate(a) for a in read_json(COMPLETED / "assets.json")]
    assert {a.kind for a in assets} == {AssetKind.RESULT_PNG, AssetKind.PROVENANCE_NPZ,
                                         AssetKind.PROVENANCE_LUT}
    for a in assets:
        assert (REPO / a.storage_uri).is_file()
        assert file_sha256(REPO / a.storage_uri) == a.sha256
    assert run.result_png_uri == f"asset://{assets[0].asset_id}"
    assert run.provenance_uri == f"asset://{assets[1].asset_id}"


def test_decision_log_is_complete_and_ordered(completed: tuple) -> None:
    run, decisions, prov = completed
    assert [d.sequence for d in decisions] == list(range(len(decisions)))
    assert tuple(d.decision_id for d in decisions) == run.policy_decision_ids
    assert all(d.run_id == run.run_id for d in decisions)
    known = set(run.policy_decision_ids)
    for entry in prov.source_lut:
        assert set(entry.decision_ids) <= known
    for d in decisions:
        if d.outcome in (PolicyOutcome.ACCEPT, PolicyOutcome.REJECT) and d.policy_key:
            assert d.operator != "none" and d.units is not None


def test_provenance_arrays_validate_with_zero_generated(window: WindowBundle,
                                                        completed: tuple) -> None:
    run, _, prov = completed
    arrays = decode_provenance((COMPLETED / "provenance.npz").read_bytes())
    sizes = {e.index: (prov.width_px, prov.height_px) for e in prov.source_lut}
    counts = validate_arrays(arrays, prov.width_px, prov.height_px, prov.source_lut, sizes)
    assert counts == prov.coverage_counts
    assert int((arrays.cls == ProvenanceClass.GENERATED_BLEND).sum()) == 0
    target_n = parse_frame_id(run.target_frame_id)[1]
    target = load_frame(window.frame(target_n), window.resolver)
    result = read_png(COMPLETED / "result.png")
    assert unsupported_changed_pixel_rate(result, target, arrays) == 0.0


def test_borrows_from_at_least_two_donors_winner_take_all(completed: tuple) -> None:
    run, _, prov = completed
    arrays = decode_provenance((COMPLETED / "provenance.npz").read_bytes())
    x1, y1, x2, y2 = run.target_bbox_px
    tile = CFG.fusion.tile_px
    tiles_by_row: dict[int, int] = {}
    for ty in range(y1, y2, tile):
        for tx in range(x1, x2, tile):
            cls = arrays.cls[ty:min(ty + tile, y2), tx:min(tx + tile, x2)]
            idx = arrays.source_index[ty:min(ty + tile, y2), tx:min(tx + tile, x2)]
            assert np.unique(cls).size == 1 and np.unique(idx).size == 1, "tiles never mix"
            if cls.flat[0] == ProvenanceClass.BORROWED:
                tiles_by_row[int(idx.flat[0])] = tiles_by_row.get(int(idx.flat[0]), 0) + 1
    assert len([r for r, n in tiles_by_row.items() if n >= 1]) >= 2
    outside = np.ones(arrays.cls.shape, dtype=bool)
    outside[y1:y2, x1:x2] = False
    assert (arrays.cls[outside] == ProvenanceClass.ORIGINAL).all()


def test_donors_belong_to_the_same_track_and_video(window: WindowBundle, completed: tuple) -> None:
    run, _, prov = completed
    assert run.track_id == window.track.track_id and run.video_id == window.video_id
    detector_frames = {o.frame_id for o in window.track.observations
                       if o.accepted and o.source is ObservationSource.DETECTOR}
    for entry in prov.source_lut:
        assert parse_frame_id(entry.frame_id)[0] == window.video_id
        if entry.role is SourceRole.DONOR:
            assert entry.frame_id in detector_frames and entry.frame_id != run.target_frame_id
            assert entry.frame_id in run.accepted_donor_frame_ids


def test_borrowed_pixels_are_donor_samples_at_recorded_coordinates(window: WindowBundle,
                                                                    completed: tuple) -> None:
    _, _, prov = completed
    arrays = decode_provenance((COMPLETED / "provenance.npz").read_bytes())
    result = read_png(COMPLETED / "result.png")
    for entry in prov.source_lut:
        if entry.role is not SourceRole.DONOR:
            continue
        sel = (arrays.cls == ProvenanceClass.BORROWED) & (arrays.source_index == entry.index)
        donor = load_frame(window.frame(entry.frame_number), window.resolver)
        sampled = cv2.remap(donor, arrays.source_x, arrays.source_y, cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_REFLECT_101)
        assert np.array_equal(sampled[sel], result[sel])


def test_integrity_matches_formula_and_minimum(completed: tuple) -> None:
    run, _, _ = completed
    i = run.integrity
    assert i is not None
    assert i.semantic_generated_pct == 0.0
    assert i.score_0_100 == integrity_score_0_100(
        supported_coverage=i.supported_coverage, mean_source_confidence=i.mean_source_confidence,
        mean_alignment_confidence=i.mean_alignment_confidence,
        track_continuity=i.track_continuity, audit_completeness=i.audit_completeness,
        semantic_generated_pct=i.semantic_generated_pct)
    assert i.score_0_100 >= CFG.integrity.min_score


def _rejects(decisions: list[PolicyDecision], frame: int) -> list[PolicyDecision]:
    return [d for d in decisions if d.outcome is PolicyOutcome.REJECT
            and d.subject_ref.endswith(f":f{frame}")]


@pytest.mark.parametrize("run_dir", [COMPLETED, REFUSED], ids=["completed", "refused"])
def test_obstructed_and_incompatible_frames_are_rule_coded(run_dir: Path) -> None:
    decisions = _decisions(run_dir)
    (obstructed,) = _rejects(decisions, 34)
    assert obstructed.rule_code is ReasonCode.DONOR_OBSTRUCTED
    assert obstructed.policy_key == "donor.max_occluded_fraction"
    assert obstructed.observed > obstructed.threshold
    (incompatible,) = _rejects(decisions, 40)
    assert incompatible.rule_code is ReasonCode.LIGHTING_OUT_OF_RANGE
    assert incompatible.units == "stops" and incompatible.observed > incompatible.threshold
    for n in (52, 53):
        (bridged,) = _rejects(decisions, n)
        assert bridged.rule_code is ReasonCode.IDENTITY_GEOMETRY_MISMATCH


# ------------------------------------------------------------------------------------------------
# Refused run
# ------------------------------------------------------------------------------------------------


def test_refused_run_is_successful_refusal_with_rule_code() -> None:
    run = _run(REFUSED)
    assert run.state is ReconstructionState.REFUSED
    assert run.refusal_reasons == (ReasonCode.INSUFFICIENT_COMPATIBLE_DONORS,)
    assert run.result_png_uri is None and run.provenance is None and run.integrity is None
    assert not (REFUSED / "result.png").exists() and not (REFUSED / "provenance.npz").exists()
    decisions = _decisions(REFUSED)
    assert tuple(d.decision_id for d in decisions) == run.policy_decision_ids
    (gate,) = [d for d in decisions if d.rule_code is ReasonCode.INSUFFICIENT_COMPATIBLE_DONORS]
    assert gate.outcome is PolicyOutcome.REJECT
    assert (gate.observed, gate.operator, gate.threshold) == (1, ">=", 2)
    assert gate.policy_key == "donor.min_count" and gate.units == "donors"


# ------------------------------------------------------------------------------------------------
# Determinism
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("fixture_id", "label"), [("plate_translate_v1", "completed"),
                                                   ("plate_single_donor_v1", "refused")])
def test_regeneration_is_deterministic(fixture_id: str, label: str) -> None:
    win = load_window(SYNTH / fixture_id)
    built = build_handbuilt_run(win, CFG, label)
    out = SYNTH / fixture_id / "reconstructions" / label
    assert json_bytes(list(built.decisions)) == (out / "decisions.json").read_bytes()
    if built.provenance is None:
        assert json_bytes(built.run) == (out / "run.json").read_bytes()
        return
    assert built.arrays is not None and built.result_png is not None
    committed = decode_provenance((out / "provenance.npz").read_bytes())
    assert array_sha256(built.arrays.as_dict()) == array_sha256(committed.as_dict())
    rebuilt = cv2.imdecode(np.frombuffer(built.result_png, np.uint8), cv2.IMREAD_COLOR)
    assert pixel_sha256(rebuilt) == pixel_sha256(read_png(out / "result.png"))
    if load_truth(win.root)["opencv_version"] == cv2.__version__:
        assert json_bytes(built.run) == (out / "run.json").read_bytes()
        assert built.npz == (out / "provenance.npz").read_bytes()
