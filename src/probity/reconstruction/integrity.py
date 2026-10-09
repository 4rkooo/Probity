"""Lane D: integrity-v1 components and score (section 9 "Integrity"). SIGNATURES FROZEN.

``Integrity = round(100 * (0.30C + 0.25Qd + 0.20Ad + 0.15Tc + 0.10Au) * (1 - G))`` via the frozen
``probity.domain.models.integrity_score_0_100``. Qd and Ad are weighted by borrowed-pixel count per
LUT row. The result must not depend on the order of ``decisions`` or ``donors`` (shuffle test).
No borrowed pixels means refusal (``NO_TILE_IMPROVED``), never an artificial Qd.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from probity.domain.enums import PolicyStage, ProvenanceClass, ReasonCode, SourceRole
from probity.domain.errors import ProbityError
from probity.domain.models import IntegrityScore, PolicyDecision, integrity_score_0_100
from probity.domain.policy import PolicyConfig
from probity.reconstruction.provenance import ProvenanceIncomplete, validate_arrays
from probity.reconstruction.types import AlignedDonor, BBox, Gate, ProvenanceArrays

LANE = "lane D"
INTEGRITY_WEIGHTS = {"C": 0.30, "Qd": 0.25, "Ad": 0.20, "Tc": 0.15, "Au": 0.10}
_COMPONENT_DP = 6

# Run-stage material rows a completed reconstruction must record. INTEGRITY is omitted so Au is
# not circular (the integrity gate is logged after this score). MODE / FIXTURE disclosure is not
# a reconstruction gate. Presence is enough; outcome is not scored.
RUN_STAGE_EXPECTED: frozenset[tuple[PolicyStage, ReasonCode]] = frozenset({
    (PolicyStage.INPUT, ReasonCode.SOURCE_HASH_VERIFIED),
    (PolicyStage.TRACK, ReasonCode.TRACK_CONFIRMED),
    (PolicyStage.QUALITY, ReasonCode.TARGET_TOO_SMALL),
    (PolicyStage.QUALITY, ReasonCode.TARGET_QUALITY_TOO_LOW),
    (PolicyStage.RANK, ReasonCode.INSUFFICIENT_COMPATIBLE_DONORS),
    (PolicyStage.FUSE, ReasonCode.KEEP_ORIGINAL_CLEAR),
    (PolicyStage.PROVENANCE, ReasonCode.PROVENANCE_COMPLETE),
    (PolicyStage.PROVENANCE, ReasonCode.GENERATED_SEMANTIC_PIXEL),
})


class IntegrityRefusal(ProbityError):
    """No integrity score is produced: zero borrowed pixels would require an invented Qd."""

    reason_code = ReasonCode.NO_TILE_IMPROVED

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems), details={"problems": list(problems)})
        self.problems = tuple(problems)


def _unit(name: str, value: float) -> float:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ProvenanceIncomplete([f"{name} {value} is not in [0, 1]"])
    return float(value)


def audit_completeness(decisions: Sequence[PolicyDecision], arrays: ProvenanceArrays) -> float:
    """Au: fraction of expected material decisions present (per LUT donor and per run stage)."""
    expected_ids = {did for e in arrays.lut if e.role is SourceRole.DONOR for did in e.decision_ids}
    present_ids = {d.decision_id for d in decisions}
    present_stages = {(d.stage, d.rule_code) for d in decisions}
    n_expected = len(expected_ids) + len(RUN_STAGE_EXPECTED)
    if n_expected == 0:
        return 0.0
    n_found = len(expected_ids & present_ids) + len(RUN_STAGE_EXPECTED & present_stages)
    return n_found / n_expected


def compute_integrity(arrays: ProvenanceArrays, subject_box: BBox,
                      donors: Mapping[int, AlignedDonor], track_continuity: float,
                      decisions: Sequence[PolicyDecision]) -> IntegrityScore:
    """``donors`` is keyed by LUT index (>= 1); ``arrays.lut`` must be populated."""
    lut = tuple(arrays.lut)
    if not lut:
        raise ProvenanceIncomplete(["arrays carry no source LUT"])
    if arrays.cls.ndim != 2:
        raise ProvenanceIncomplete([f"class map must be 2-D, got shape {arrays.cls.shape}"])
    height, width = arrays.cls.shape
    x1, y1, x2, y2 = subject_box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ProvenanceIncomplete(
            [f"subject box {subject_box} is not a non-empty box inside {width}x{height}"])
    validate_arrays(arrays, width, height, lut, {e.index: (width, height) for e in lut})
    tc = _unit("track_continuity", float(track_continuity))

    borrowed = arrays.cls == int(ProvenanceClass.BORROWED)
    n_borrowed = int(borrowed.sum())
    if n_borrowed == 0:
        raise IntegrityRefusal(["no borrowed pixels; refuse rather than invent Qd"])

    idx = arrays.source_index
    rows = [int(k) for k in np.unique(idx[borrowed])]
    missing = sorted(k for k in rows if k not in donors)
    if missing:
        raise ProvenanceIncomplete([f"no donor provided for borrowed LUT rows {missing}"])

    qd_num = 0.0
    ad_num = 0.0
    for k in rows:
        n_k = int(((idx == k) & borrowed).sum())
        donor = donors[k]
        qd_num += n_k * _unit(f"LUT row {k} Q", float(donor.quality.Q))
        ad_num += n_k * _unit(f"LUT row {k} A", float(donor.alignment.A))
    qd = qd_num / n_borrowed
    ad = ad_num / n_borrowed

    n_original = int((arrays.cls == int(ProvenanceClass.ORIGINAL)).sum())
    supported = (n_original + n_borrowed) / (width * height)
    subject = arrays.cls[y1:y2, x1:x2]
    generated = int((subject == int(ProvenanceClass.GENERATED_BLEND)).sum())
    semantic_generated_pct = 100.0 * generated / subject.size
    au = audit_completeness(decisions, arrays)

    components = dict(
        supported_coverage=_unit("supported_coverage", supported),
        mean_source_confidence=round(qd, _COMPONENT_DP),
        mean_alignment_confidence=round(ad, _COMPONENT_DP),
        track_continuity=tc,
        audit_completeness=round(au, _COMPONENT_DP),
        semantic_generated_pct=semantic_generated_pct,
    )
    score = integrity_score_0_100(**components)
    return IntegrityScore(score_0_100=score, **components)


def integrity_gate(score: IntegrityScore, cfg: PolicyConfig) -> Gate:
    """``INTEGRITY_BELOW_MINIMUM``: score_0_100 >= integrity.min_score (points)."""
    return Gate(
        rule_code=ReasonCode.INTEGRITY_BELOW_MINIMUM,
        stage=PolicyStage.INTEGRITY,
        observed=score.score_0_100,
        operator=">=",
        threshold=cfg.integrity.min_score,
        units="points",
        policy_key="integrity.min_score",
        accept_reason="Integrity meets minimum",
        reject_reason="Integrity below minimum",
    )
