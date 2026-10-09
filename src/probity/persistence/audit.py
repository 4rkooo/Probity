"""Append-only, per-case sequenced, hash-chained audit log (section 10).

Tamper-evident within the demo database, not a trusted external ledger. Rows are chained per
``chain_key`` (the case ID, or ``""`` for writes whose case cannot be resolved).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Connection, Engine, func, insert, select

from probity.domain.ids import canonical_json, canonical_sha256
from probity.persistence.db import read_transaction
from probity.persistence.schema import audit_log

GENESIS_HASH = "0" * 64
UNSCOPED_CHAIN = ""


@dataclass(frozen=True, slots=True)
class AuditRow:
    chain_key: str
    sequence: int
    occurred_at: str
    actor: str
    action: str
    entity_type: str
    entity_id: str
    correlation_id: str | None
    reason_code: str | None
    params: str
    previous_row_hash: str
    row_hash: str

    def hashed_fields(self) -> dict[str, Any]:
        return {
            "chain_key": self.chain_key,
            "sequence": self.sequence,
            "occurred_at": self.occurred_at,
            "actor": self.actor,
            "action": self.action,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "correlation_id": self.correlation_id,
            "reason_code": self.reason_code,
            "params": self.params,
            "previous_row_hash": self.previous_row_hash,
        }


@dataclass(frozen=True, slots=True)
class ChainVerification:
    chain_key: str
    rows: int
    valid: bool
    first_broken_sequence: int | None


def append_audit(
    conn: Connection,
    *,
    case_id: str | None,
    occurred_at: str,
    actor: str,
    action: str,
    entity_type: str,
    entity_id: str,
    correlation_id: str | None = None,
    reason_code: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> AuditRow:
    """Append one row inside the caller's write transaction (``BEGIN IMMEDIATE`` serializes)."""
    chain_key = case_id or UNSCOPED_CHAIN
    last = conn.execute(
        select(audit_log.c.sequence, audit_log.c.row_hash)
        .where(audit_log.c.chain_key == chain_key)
        .order_by(audit_log.c.sequence.desc())
        .limit(1)
    ).first()
    sequence = 0 if last is None else last.sequence + 1
    previous = GENESIS_HASH if last is None else last.row_hash
    draft = AuditRow(
        chain_key=chain_key,
        sequence=sequence,
        occurred_at=occurred_at,
        actor=actor[:128],
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        correlation_id=correlation_id,
        reason_code=reason_code,
        params=canonical_json(dict(params or {})).decode("utf-8"),
        previous_row_hash=previous,
        row_hash="",
    )
    row_hash = canonical_sha256(draft.hashed_fields())
    values = {**draft.hashed_fields(), "row_hash": row_hash}
    conn.execute(insert(audit_log).values(**values))
    return AuditRow(**values)


def _row(r: Any) -> AuditRow:
    return AuditRow(
        chain_key=r.chain_key,
        sequence=r.sequence,
        occurred_at=r.occurred_at,
        actor=r.actor,
        action=r.action,
        entity_type=r.entity_type,
        entity_id=r.entity_id,
        correlation_id=r.correlation_id,
        reason_code=r.reason_code,
        params=r.params,
        previous_row_hash=r.previous_row_hash,
        row_hash=r.row_hash,
    )


def audit_rows(engine: Engine, case_id: str | None) -> Sequence[AuditRow]:
    with read_transaction(engine) as conn:
        rows = conn.execute(
            select(audit_log)
            .where(audit_log.c.chain_key == (case_id or UNSCOPED_CHAIN))
            .order_by(audit_log.c.sequence)
        ).all()
    return [_row(r) for r in rows]


def verify_chain(engine: Engine, case_id: str | None) -> ChainVerification:
    """Recompute every row hash and link; report the first broken sequence number."""
    rows = audit_rows(engine, case_id)
    previous = GENESIS_HASH
    for expected_seq, row in enumerate(rows):
        recomputed = canonical_sha256(row.hashed_fields())
        if (
            row.sequence != expected_seq
            or row.previous_row_hash != previous
            or row.row_hash != recomputed
        ):
            return ChainVerification(row.chain_key, len(rows), False, row.sequence)
        previous = row.row_hash
    return ChainVerification(case_id or UNSCOPED_CHAIN, len(rows), True, None)


def chain_keys(engine: Engine) -> Sequence[str]:
    with read_transaction(engine) as conn:
        return list(conn.execute(select(func.distinct(audit_log.c.chain_key))).scalars())
