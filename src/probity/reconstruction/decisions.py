"""Append-only, sequenced PolicyDecision log for one reconstruction run, and declarative gates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from probity.domain.enums import PolicyOutcome, PolicyStage, ReasonCode
from probity.domain.models import PolicyDecision, ScalarValue
from probity.reconstruction.determinism import FixedClock, seeded_uuid7

Operator = Literal["<", "<=", ">", ">=", "==", "!=", "in", "none"]


def passes(observed: ScalarValue, operator: Operator, threshold: ScalarValue) -> bool:
    if operator == "==":
        return observed == threshold
    if operator == "!=":
        return observed != threshold
    if not isinstance(observed, int | float) or not isinstance(threshold, int | float):
        raise ValueError(f"operator {operator!r} needs numeric operands")
    if operator == "<":
        return observed < threshold
    if operator == "<=":
        return observed <= threshold
    if operator == ">":
        return observed > threshold
    if operator == ">=":
        return observed >= threshold
    raise ValueError(f"operator {operator!r} is not a comparison")


@dataclass(frozen=True)
class Gate:
    """One material comparison. ``reject_code`` defaults to ``rule_code``."""

    rule_code: ReasonCode
    stage: PolicyStage
    observed: ScalarValue
    operator: Operator
    threshold: ScalarValue
    units: str | None
    policy_key: str | None
    accept_reason: str
    reject_reason: str
    reject_code: ReasonCode | None = None

    @property
    def ok(self) -> bool:
        return passes(self.observed, self.operator, self.threshold)

    @property
    def failure_code(self) -> ReasonCode:
        return self.reject_code or self.rule_code


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
