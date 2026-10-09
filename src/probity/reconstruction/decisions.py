"""Append-only, sequenced PolicyDecision log for one reconstruction run."""

from __future__ import annotations

from typing import Literal

from probity.domain.enums import PolicyOutcome, PolicyStage, ReasonCode
from probity.domain.models import PolicyDecision, ScalarValue
from probity.reconstruction.determinism import FixedClock, seeded_uuid7

Operator = Literal["<", "<=", ">", ">=", "==", "!=", "in", "none"]


def passes(observed: float, operator: Operator, threshold: float) -> bool:
    if operator == "<":
        return observed < threshold
    if operator == "<=":
        return observed <= threshold
    if operator == ">":
        return observed > threshold
    if operator == ">=":
        return observed >= threshold
    if operator == "==":
        return observed == threshold
    if operator == "!=":
        return observed != threshold
    raise ValueError(f"operator {operator!r} is not numeric")


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
            observed=observed,
            operator=operator,
            threshold=threshold,
            units=units,
            policy_key=policy_key,
        )
        self._rows.append(row)
        return row

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
        record_accept: bool = True,
    ) -> bool:
        """Evaluate ``observed operator threshold``, record it, and return whether it passed."""
        ok = passes(observed, operator, threshold)
        if ok and not record_accept:
            return True
        self.add(
            rule_code,
            stage,
            subject_ref,
            PolicyOutcome.ACCEPT if ok else PolicyOutcome.REJECT,
            accept_reason if ok else reject_reason,
            observed=observed if isinstance(observed, int) else float(observed),
            operator=operator,
            threshold=threshold,
            units=units,
            policy_key=policy_key,
        )
        return ok
