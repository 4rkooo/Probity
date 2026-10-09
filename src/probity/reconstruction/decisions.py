"""Append-only, sequenced PolicyDecision log for one reconstruction run, and declarative gates."""

from __future__ import annotations

from collections.abc import Iterable

from probity.domain.enums import PolicyOutcome, PolicyStage, ReasonCode
from probity.domain.models import PolicyDecision, ScalarValue
from probity.reconstruction.determinism import FixedClock, seeded_uuid7
from probity.reconstruction.types import Gate, GateResult, Operator, passes

__all__ = ["DecisionLog", "Gate", "GateResult", "Operator", "first_failure", "passes"]


def first_failure(gates: Iterable[Gate]) -> Gate | None:
    return next((g for g in gates if not g.ok), None)


def _normalize(value: ScalarValue) -> ScalarValue:
    if isinstance(value, bool) or value is None or isinstance(value, int | str):
        return value
    return float(value)


class DecisionLog:
    def __init__(self, run_id: str, clock: FixedClock) -> None:
        self.run_id = run_id
        self._clock = clock
        self._rows: list[PolicyDecision] = []

    @property
    def rows(self) -> tuple[PolicyDecision, ...]:
        return tuple(self._rows)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(r.decision_id for r in self._rows)

    def add(
        self,
        rule_code: ReasonCode,
        stage: PolicyStage,
        subject_ref: str,
        outcome: PolicyOutcome,
        reason: str,
        *,
        observed: ScalarValue = None,
        operator: Operator = "none",
        threshold: ScalarValue = None,
        units: str | None = None,
        policy_key: str | None = None,
    ) -> PolicyDecision:
        seq = len(self._rows)
        row = PolicyDecision.create(
            created_at=self._clock.at(seq),
            decision_id=seeded_uuid7(f"{self.run_id}:decision:{seq}", self._clock.unix_ms + seq),
            run_id=self.run_id,
            sequence=seq,
            rule_code=rule_code,
            stage=stage,
            subject_ref=subject_ref,
            outcome=outcome,
            reason=reason,
            observed=_normalize(observed),
            operator=operator,
            threshold=_normalize(threshold),
            units=units,
            policy_key=policy_key,
        )
        self._rows.append(row)
        return row

    def check(self, gate: Gate, subject_ref: str) -> bool:
        """Record ``gate`` as ACCEPT or REJECT and return whether it passed."""
        ok = gate.ok
        self.add(
            gate.rule_code if ok else gate.failure_code,
            gate.stage,
            subject_ref,
            PolicyOutcome.ACCEPT if ok else PolicyOutcome.REJECT,
            gate.accept_reason if ok else gate.reject_reason,
            observed=gate.observed,
            operator=gate.operator,
            threshold=gate.threshold,
            units=gate.units,
            policy_key=gate.policy_key,
        )
        return ok

    def apply(self, gates: Iterable[Gate], subject_ref: str, *, all_gates: bool = False
              ) -> GateResult:
        """Record gates in order. By default stop after the first failure (one REJECT row);
        with ``all_gates`` record every gate. Accepted iff every recorded gate passed."""
        start = len(self._rows)
        accepted = True
        for g in gates:
            accepted = self.check(g, subject_ref) and accepted
            if not accepted and not all_gates:
                break
        return GateResult(accepted, tuple(self._rows[start:]))

    def gate(
        self,
        rule_code: ReasonCode,
        stage: PolicyStage,
        subject_ref: str,
        *,
        observed: float,
        operator: Operator,
        threshold: float,
        units: str | None,
        policy_key: str,
        accept_reason: str,
        reject_reason: str,
        reject_code: ReasonCode | None = None,
    ) -> bool:
        """Evaluate ``observed operator threshold``, record it, and return whether it passed."""
        return self.check(Gate(rule_code, stage, observed, operator, threshold, units, policy_key,
                               accept_reason, reject_reason, reject_code), subject_ref)
