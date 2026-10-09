"""Lane D: integrity-v1 components and score (section 9 "Integrity"). SIGNATURES FROZEN.

``Integrity = round(100 * (0.30C + 0.25Qd + 0.20Ad + 0.15Tc + 0.10Au) * (1 - G))`` via the frozen
``probity.domain.models.integrity_score_0_100``. Qd and Ad are weighted by borrowed-pixel count per
LUT row. The result must not depend on the order of ``decisions`` or ``donors`` (shuffle test).
No borrowed pixels means refusal (``NO_TILE_IMPROVED``), never an artificial Qd.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from probity.domain.models import IntegrityScore, PolicyDecision
from probity.domain.policy import PolicyConfig
from probity.reconstruction.types import AlignedDonor, BBox, Gate, ProvenanceArrays

LANE = "lane D"
INTEGRITY_WEIGHTS = {"C": 0.30, "Qd": 0.25, "Ad": 0.20, "Tc": 0.15, "Au": 0.10}


def audit_completeness(decisions: Sequence[PolicyDecision], arrays: ProvenanceArrays) -> float:
    """Au: fraction of expected material decisions present (per LUT donor and per run stage)."""
    raise NotImplementedError(LANE)


def compute_integrity(arrays: ProvenanceArrays, subject_box: BBox,
                      donors: Mapping[int, AlignedDonor], track_continuity: float,
                      decisions: Sequence[PolicyDecision]) -> IntegrityScore:
    """``donors`` is keyed by LUT index (>= 1); ``arrays.lut`` must be populated."""
    raise NotImplementedError(LANE)


def integrity_gate(score: IntegrityScore, cfg: PolicyConfig) -> Gate:
    """``INTEGRITY_BELOW_MINIMUM``: score_0_100 >= integrity.min_score (points)."""
    raise NotImplementedError(LANE)
