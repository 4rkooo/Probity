"""Every Lane D policy threshold: one case just inside and one just outside, exact reason code."""

from __future__ import annotations

import pytest

from probity.domain.enums import ProvenanceClass, ReasonCode
from probity.domain.models import IntegrityScore, integrity_score_0_100
from probity.domain.policy import PolicyConfig
from probity.reconstruction.integrity import compute_integrity, integrity_gate
from probity.reconstruction.provenance import (
    GeneratedPixelError,
    ProvenanceIncomplete,
    build_provenance_arrays,
    build_source_lut,
    provenance_record,
    validate_arrays,
)
from probity.reconstruction.io import sha256_bytes
from probity.reconstruction.provenance import encode_provenance

from ._builders import (
    H,
    RUN,
    START,
    SUBJECT,
    W,
    build_scene,
    donor_map,
    npz_uri,
)


def _arrays(built):
    lut = build_source_lut(built.target, built.donors, built.ids)
    return build_provenance_arrays(built.target, built.fusion, built.donors, lut)


def test_required_provenance_coverage_just_inside_and_outside(rng: np.random.Generator,
                                                              policy: PolicyConfig) -> None:
    """integrity.required_provenance_coverage = 1.00: every pixel or PROVENANCE_INCOMPLETE."""
    assert policy.integrity.required_provenance_coverage == 1.00
    built = build_scene(rng, policy)
    arrays = _arrays(built)
    counts = validate_arrays(arrays, W, H, arrays.lut, {e.index: (W, H) for e in arrays.lut})
    assert counts.ORIGINAL + counts.BORROWED == W * H
    assert counts.GENERATED_BLEND == 0

    arrays.cls[0, 0] = 3  # one-pixel hole: coverage = (W*H-1)/(W*H) just below 1.00
    with pytest.raises(ProvenanceIncomplete, match="no valid provenance class") as exc:
        validate_arrays(arrays, W, H, arrays.lut, {e.index: (W, H) for e in arrays.lut})
    assert exc.value.reason_code is ReasonCode.PROVENANCE_INCOMPLETE
    with pytest.raises(ProvenanceIncomplete) as rec_exc:
        provenance_record(arrays, run_id=RUN, subject_box=SUBJECT, npz_uri=npz_uri(),
                          npz_sha256=sha256_bytes(encode_provenance(arrays)), created_at=START)
    assert rec_exc.value.reason_code is ReasonCode.PROVENANCE_INCOMPLETE


def test_max_semantic_generated_fraction_just_inside_and_outside(rng: np.random.Generator,
                                                                 policy: PolicyConfig) -> None:
    """integrity.max_semantic_generated_fraction = 0.00: zero class-2 or GENERATED_SEMANTIC_PIXEL."""
    assert policy.integrity.max_semantic_generated_fraction == 0.00
    built = build_scene(rng, policy)
    arrays = _arrays(built)
    assert int((arrays.cls == ProvenanceClass.GENERATED_BLEND).sum()) == 0
    compute_integrity(arrays, SUBJECT, donor_map(built.donors), 1.0, built.decisions)

    arrays.cls[SUBJECT[1], SUBJECT[0]] = ProvenanceClass.GENERATED_BLEND
    with pytest.raises(GeneratedPixelError) as exc:
        validate_arrays(arrays, W, H, arrays.lut, {e.index: (W, H) for e in arrays.lut})
    assert exc.value.reason_code is ReasonCode.GENERATED_SEMANTIC_PIXEL
    with pytest.raises(GeneratedPixelError) as comp_exc:
        compute_integrity(arrays, SUBJECT, donor_map(built.donors), 1.0, built.decisions)
    assert comp_exc.value.reason_code is ReasonCode.GENERATED_SEMANTIC_PIXEL


def test_min_score_just_inside_70_and_just_outside_69(policy: PolicyConfig) -> None:
    """integrity.min_score = 70: score 70 accepts, score 69 is INTEGRITY_BELOW_MINIMUM."""
    assert policy.integrity.min_score == 70
    inside_kw = dict(supported_coverage=1.0, mean_source_confidence=0.4,
                     mean_alignment_confidence=0.25, track_continuity=1.0,
                     audit_completeness=1.0, semantic_generated_pct=0.0)
    outside_kw = dict(inside_kw, mean_alignment_confidence=0.20)
    inside = IntegrityScore(score_0_100=integrity_score_0_100(**inside_kw), **inside_kw)
    outside = IntegrityScore(score_0_100=integrity_score_0_100(**outside_kw), **outside_kw)
    assert inside.score_0_100 == 70
    assert outside.score_0_100 == 69
    assert integrity_gate(inside, policy).ok
    bad = integrity_gate(outside, policy)
    assert not bad.ok
    assert bad.rule_code is ReasonCode.INTEGRITY_BELOW_MINIMUM
    assert bad.failure_code is ReasonCode.INTEGRITY_BELOW_MINIMUM
