"""Branch guards: must pass on EVERY Person 2 branch at all times. FROZEN.

They pin the golden runs (schema, section 10 provenance invariants, rule-coded refusal), the
fixture bytes (scripts/golden_hashes.json), and the frozen contract plus lane signatures
(scripts/frozen_hashes.json).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from probity.domain.enums import (
    AlignmentMethod,
    AssetKind,
    Interpolation,
    ObservationSource,
    PolicyOutcome,
    ProvenanceClass,
    ReasonCode,
    ReconstructionState,
    SourceRole,
)
from probity.domain.ids import parse_frame_id
from probity.domain.models import integrity_score_0_100
from probity.eval.window import WindowBundle
from probity.reconstruction.io import assert_evidentiary_input, pixel_sha256, sha256_bytes
from probity.reconstruction.provenance import (
    assert_supported_changes,
    coverage_pct,
    subject_coverage_pct,
    validate_arrays,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GoldenRun = Any  # conftest.GoldenRun; conftest is not importable as a module


def _checker() -> ModuleType:
    path = REPO_ROOT / "scripts" / "check_lane.py"
    spec = importlib.util.spec_from_file_location("probity_check_lane", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("probity_check_lane", module)
    spec.loader.exec_module(module)
    return module


CHECK = _checker()


# ------------------------------------------------------------------------------------------------
# Hash pins
# ------------------------------------------------------------------------------------------------


def test_fixture_files_match_golden_hashes() -> None:
    problems = CHECK.golden_problems(REPO_ROOT)
    assert not problems, "fixtures differ from scripts/golden_hashes.json:\n" + "\n".join(problems)


def test_frozen_contract_and_signatures_match() -> None:
    problems = CHECK.frozen_problems(REPO_ROOT)
    assert not problems, "frozen files differ from scripts/frozen_hashes.json:\n" + "\n".join(
        problems)


def test_hash_normalization_ignores_crlf(tmp_path: Path) -> None:
    lf, crlf, png = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.png"
    lf.write_bytes(b'{\n  "a": 1\n}\n')
    crlf.write_bytes(b'{\r\n  "a": 1\r\n}\r\n')
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert CHECK.normalized_sha256(lf) == CHECK.normalized_sha256(crlf)
    assert CHECK.normalized_sha256(png) == sha256_bytes(b"\x89PNG\r\n\x1a\n")


# ------------------------------------------------------------------------------------------------
# Golden runs: schema
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["golden_completed", "golden_refused"])
def test_goldens_validate_against_frozen_schema(name: str, request: pytest.FixtureRequest
                                                ) -> None:
    golden: GoldenRun = request.getfixturevalue(name)
    run = golden.run
    assert run.content_sha256 == run.compute_content_sha256()
    assert run.algorithm_version == "trueframe-tile-v1"
    assert run.policy_profile == "demo-conservative-v1"
    assert [d.decision_id for d in golden.decisions] == list(run.policy_decision_ids)
    assert [d.sequence for d in golden.decisions] == list(range(len(golden.decisions)))
    assert {d.run_id for d in golden.decisions} == {run.run_id}
    for d in golden.decisions:
        assert d.content_sha256 == d.compute_content_sha256()


# ------------------------------------------------------------------------------------------------
# Golden completed run: section 10 provenance invariants
# ------------------------------------------------------------------------------------------------


def test_completed_golden_provenance_invariants(golden_completed: GoldenRun,
                                                translate_window: WindowBundle,
                                                translate_images: dict[int, np.ndarray]) -> None:
    g, win = golden_completed, translate_window
    run, prov, arrays, result = g.run, g.provenance, g.arrays, g.result
    assert run.state is ReconstructionState.SUCCEEDED and run.refusal_reasons == ()
    assert prov is not None and arrays is not None and result is not None and g.npz_bytes
    assert run.provenance is not None and run.integrity is not None

    # Artifact hashes bind run, record, and bytes.
    assert sha256_bytes(g.npz_bytes) == prov.artifact_sha256 == run.provenance_sha256
    assert sha256_bytes((g.root / "result.png").read_bytes()) == run.result_png_sha256
    assert prov.run_id == run.run_id and run.provenance.source_lut == prov.source_lut

    # Full-frame arrays, every pixel resolves, no class 2, coverage exactly W*H.
    H, W = result.shape[:2]
    target_n = parse_frame_id(run.target_frame_id)[1]
    target = translate_images[target_n]
    assert (prov.width_px, prov.height_px) == (W, H) and target.shape == result.shape
    sizes = {e.index: (W, H) for e in prov.source_lut}
    counts = validate_arrays(arrays, W, H, prov.source_lut, sizes)
    assert counts == prov.coverage_counts and counts.GENERATED_BLEND == 0
    assert not (arrays.cls == ProvenanceClass.GENERATED_BLEND).any()
    assert prov.coverage_complete and run.provenance.coverage_complete
    assert coverage_pct(counts, W * H) == prov.coverage_pct == run.provenance.coverage_pct
    assert subject_coverage_pct(arrays, prov.subject_bbox_px) == prov.subject_coverage_pct
    assert prov.subject_bbox_px == run.target_bbox_px
    assert prov.encoding_version == "npz-pixel-v1" and prov.tile_size_px == 8

    # Every changed pixel is BORROWED; ORIGINAL pixels are the untouched target.
    assert_supported_changes(result, target, arrays)
    original = arrays.cls == ProvenanceClass.ORIGINAL
    assert np.array_equal(result[original], target[original])
    borrowed = arrays.cls == ProvenanceClass.BORROWED
    assert borrowed.any(), "the completed golden must borrow real pixels"
    x1, y1, x2, y2 = prov.subject_bbox_px
    outside = np.ones_like(borrowed)
    outside[y1:y2, x1:x2] = False
    assert not (borrowed & outside).any(), "borrowing is confined to the subject box"

    # LUT: row 0 target identity; donors from the same track and video, evidentiary inputs only.
    lut = prov.source_lut
    assert lut[0].role is SourceRole.TARGET and lut[0].frame_id == run.target_frame_id
    assert lut[0].alignment_method is AlignmentMethod.IDENTITY
    donors = lut[1:]
    assert len(donors) >= 2
    assert tuple(e.frame_id for e in donors) == run.accepted_donor_frame_ids
    members = {o.frame_id: o for o in win.track.observations if o.accepted}
    known_ids = {d.decision_id for d in g.decisions}
    refs = {r.frame_id: r for r in win.frames}
    assert win.track.track_id == run.track_id and win.video_id == run.video_id
    for e in donors:
        assert e.role is SourceRole.DONOR and e.frame_id != run.target_frame_id
        assert parse_frame_id(e.frame_id)[0] == run.video_id
        assert members[e.frame_id].source is ObservationSource.DETECTOR
        assert e.interpolation is Interpolation.LANCZOS4 and e.matrix is not None
        assert e.decision_ids and set(e.decision_ids) <= known_ids
        ref = assert_evidentiary_input(refs[e.frame_id])
        assert ref.pixel_sha256 == pixel_sha256(translate_images[ref.frame_number])
    used = set(np.unique(arrays.source_index[borrowed]).tolist())
    assert used <= {e.index for e in donors} and len(used) >= 1

    # Assets are evidentiary reconstruction outputs, never baselines.
    kinds = {a.kind for a in g.assets}
    assert kinds == {AssetKind.RESULT_PNG, AssetKind.PROVENANCE_NPZ, AssetKind.PROVENANCE_LUT}
    assert all(not a.non_evidentiary for a in g.assets)
    for a in g.assets:
        assert sha256_bytes((REPO_ROOT / a.storage_uri).read_bytes()) == a.sha256

    # Integrity formula and gates.
    i = run.integrity
    assert i.semantic_generated_pct == 0.0 and i.supported_coverage == 1.0
    assert i.score_0_100 == integrity_score_0_100(
        i.supported_coverage, i.mean_source_confidence, i.mean_alignment_confidence,
        i.track_continuity, i.audit_completeness, i.semantic_generated_pct)
    codes = {(d.rule_code, d.outcome) for d in g.decisions}
    assert (ReasonCode.PROVENANCE_COMPLETE, PolicyOutcome.ACCEPT) in codes
    assert (ReasonCode.INTEGRITY_BELOW_MINIMUM, PolicyOutcome.ACCEPT) in codes
    assert (ReasonCode.BORROW_TILE_ACCEPTED, PolicyOutcome.ACCEPT) in codes


# ------------------------------------------------------------------------------------------------
# Golden refused run: rule-coded refusal, nothing emitted
# ------------------------------------------------------------------------------------------------


def test_refused_golden_is_rule_coded(golden_refused: GoldenRun) -> None:
    g = golden_refused
    run = g.run
    assert run.state is ReconstructionState.REFUSED and run.refusal_reasons
    assert run.result_png_uri is None and run.provenance_uri is None and run.provenance is None
    assert run.integrity is None and run.result_png_sha256 is None
    assert g.provenance is None and g.result is None
    for reason in run.refusal_reasons:
        rows = [d for d in g.decisions
                if d.rule_code is reason and d.outcome is PolicyOutcome.REJECT]
        assert rows, f"refusal {reason} has no REJECT decision"
        for d in rows:
            assert d.operator != "none" and d.observed is not None and d.threshold is not None
            assert d.units and d.policy_key


@pytest.mark.parametrize("name", ["golden_completed", "golden_refused"])
def test_every_material_decision_is_rule_coded(name: str, request: pytest.FixtureRequest) -> None:
    golden: GoldenRun = request.getfixturevalue(name)
    for d in golden.decisions:
        if d.outcome in {PolicyOutcome.ACCEPT, PolicyOutcome.REJECT}:
            assert d.operator != "none", d
            assert d.observed is not None and d.threshold is not None, d
    assert all(d.rule_code is not ReasonCode.GENERATED_SEMANTIC_PIXEL
               or d.outcome is PolicyOutcome.ACCEPT for d in golden.decisions)
