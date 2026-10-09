"""Lane D integrity-v1: components, order-independence, and refusal with no borrowed pixels."""

from __future__ import annotations

import random

import numpy as np
import pytest

from probity.domain.enums import ProvenanceClass, ReasonCode, SourceRole
from probity.domain.models import IntegrityScore, integrity_score_0_100
from probity.domain.policy import PolicyConfig
from probity.reconstruction.integrity import (
    RUN_STAGE_EXPECTED,
    IntegrityRefusal,
    audit_completeness,
    compute_integrity,
    integrity_gate,
)
from probity.reconstruction.provenance import (
    ProvenanceIncomplete,
    build_provenance_arrays,
    build_source_lut,
)
from ._builders import (
    SUBJECT,
    build_scene,
    donor_map,
    single_donor,
)


def _arrays(built):
    lut = build_source_lut(built.target, built.donors, built.ids)
    return build_provenance_arrays(built.target, built.fusion, built.donors, lut)


def _score(**components: float) -> IntegrityScore:
    return IntegrityScore(score_0_100=integrity_score_0_100(**components), **components)


def test_qd_ad_are_borrowed_pixel_weighted(rng: np.random.Generator,
                                           policy: PolicyConfig) -> None:
    built = build_scene(rng, policy)
    arrays = _arrays(built)
    borrowed = arrays.cls == ProvenanceClass.BORROWED
    idx = arrays.source_index
    n = int(borrowed.sum())
    assert n > 0
    qd = ad = 0.0
    for k, donor in donor_map(built.donors).items():
        n_k = int(((idx == k) & borrowed).sum())
        qd += n_k * donor.quality.Q
        ad += n_k * donor.alignment.A
    score = compute_integrity(arrays, SUBJECT, donor_map(built.donors), 1.0, built.decisions)
    assert score.mean_source_confidence == round(qd / n, 6)
    assert score.mean_alignment_confidence == round(ad / n, 6)
    assert score.supported_coverage == 1.0
    assert score.semantic_generated_pct == 0.0
    assert score.audit_completeness == 1.0
    assert score.formula_version == "integrity-v1"
    assert score.score_0_100 == integrity_score_0_100(
        score.supported_coverage, score.mean_source_confidence, score.mean_alignment_confidence,
        score.track_continuity, score.audit_completeness, score.semantic_generated_pct)


def test_no_borrowed_pixels_refuses_without_inventing_qd(rng: np.random.Generator,
                                                         policy: PolicyConfig) -> None:
    built = build_scene(rng, policy, borrow={})
    arrays = _arrays(built)
    assert not (arrays.cls == ProvenanceClass.BORROWED).any()
    with pytest.raises(IntegrityRefusal, match="no borrowed pixels") as exc:
        compute_integrity(arrays, SUBJECT, donor_map(built.donors), 1.0, built.decisions)
    assert exc.value.reason_code is ReasonCode.NO_TILE_IMPROVED


def test_shuffle_decisions_and_donors_is_order_independent(rng: np.random.Generator,
                                                           policy: PolicyConfig) -> None:
    built = build_scene(rng, policy)
    arrays = _arrays(built)
    donors = donor_map(built.donors)
    first = compute_integrity(arrays, SUBJECT, donors, 0.84, built.decisions)
    shuffled = list(built.decisions)
    random.Random(20261009).shuffle(shuffled)
    reversed_donors = {k: donors[k] for k in reversed(list(donors))}
    second = compute_integrity(arrays, SUBJECT, reversed_donors, 0.84, tuple(shuffled))
    assert first == second
    assert first.score_0_100 == second.score_0_100


def test_audit_completeness_is_expected_fraction(rng: np.random.Generator,
                                                 policy: PolicyConfig) -> None:
    built = build_scene(rng, policy)
    arrays = _arrays(built)
    assert audit_completeness(built.decisions, arrays) == 1.0
    donor_ids = {did for e in arrays.lut if e.role is SourceRole.DONOR for did in e.decision_ids}
    n = len(donor_ids) + len(RUN_STAGE_EXPECTED)
    missing_one = tuple(d for d in built.decisions if d.decision_id != next(iter(donor_ids)))
    assert audit_completeness(missing_one, arrays) == pytest.approx((n - 1) / n)
    extra = built.decisions + built.decisions
    assert audit_completeness(extra, arrays) == 1.0
    shuffled = list(built.decisions)
    random.Random(7).shuffle(shuffled)
    assert audit_completeness(tuple(shuffled), arrays) == 1.0


def test_au_on_readonly_golden_is_complete(golden_completed) -> None:
    g = golden_completed
    assert g.arrays is not None
    assert audit_completeness(g.decisions, g.arrays) == 1.0


def test_missing_borrowed_donor_is_incomplete(rng: np.random.Generator,
                                              policy: PolicyConfig) -> None:
    built = build_scene(rng, policy)
    arrays = _arrays(built)
    donors = donor_map(built.donors)
    donors.pop(1)
    with pytest.raises(ProvenanceIncomplete, match="no donor provided") as exc:
        compute_integrity(arrays, SUBJECT, donors, 1.0, built.decisions)
    assert exc.value.reason_code is ReasonCode.PROVENANCE_INCOMPLETE


def test_integrity_gate_just_inside_and_outside_min_score(policy: PolicyConfig) -> None:
    inside = _score(supported_coverage=1.0, mean_source_confidence=0.4,
                    mean_alignment_confidence=0.25, track_continuity=1.0,
                    audit_completeness=1.0, semantic_generated_pct=0.0)
    outside = _score(supported_coverage=1.0, mean_source_confidence=0.4,
                     mean_alignment_confidence=0.20, track_continuity=1.0,
                     audit_completeness=1.0, semantic_generated_pct=0.0)
    assert inside.score_0_100 == policy.integrity.min_score == 70
    assert outside.score_0_100 == policy.integrity.min_score - 1 == 69
    ok = integrity_gate(inside, policy)
    bad = integrity_gate(outside, policy)
    assert ok.rule_code is ReasonCode.INTEGRITY_BELOW_MINIMUM
    assert ok.ok and ok.observed == 70 and ok.threshold == 70
    assert not bad.ok
    assert bad.failure_code is ReasonCode.INTEGRITY_BELOW_MINIMUM
    assert bad.observed == 69 and bad.threshold == 70


def test_compute_integrity_crosses_min_score_via_alignment(rng: np.random.Generator,
                                                           policy: PolicyConfig) -> None:
    inside_built = single_donor(rng, policy, Q=0.4, A=0.25)
    outside_built = single_donor(rng, policy, Q=0.4, A=0.20)
    inside = compute_integrity(_arrays(inside_built), SUBJECT, donor_map(inside_built.donors),
                               1.0, inside_built.decisions)
    outside = compute_integrity(_arrays(outside_built), SUBJECT, donor_map(outside_built.donors),
                                1.0, outside_built.decisions)
    assert inside.score_0_100 == 70
    assert outside.score_0_100 == 69
    assert integrity_gate(inside, policy).ok
    assert not integrity_gate(outside, policy).ok
    assert integrity_gate(outside, policy).failure_code is ReasonCode.INTEGRITY_BELOW_MINIMUM
