"""SQLite idempotency store: unique (route, case, key), 24h TTL, body-hash conflict."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy import Engine, delete, insert, select, update

from probity.domain.errors import IdempotencyConflict
from probity.domain.ids import canonical_sha256, parse_utc
from probity.persistence.db import read_transaction, to_epoch_us, utc_clock, write_transaction
from probity.persistence.schema import idempotency_records
from probity.ports import IdempotencyRecord

Clock = Callable[[], datetime]
EMPTY_CASE = ""


def request_fingerprint(route: str, case_id: str | None, body: Any) -> str:
    """SHA-256 of canonical JSON ``{route, case_id, body}``."""
    return canonical_sha256({"route": route, "case_id": case_id or EMPTY_CASE, "body": body})


def _case_scope(case_id: str | None) -> str:
    return case_id or EMPTY_CASE


def _to_record(row: Any) -> IdempotencyRecord:
    case_id = None if row.case_scope == EMPTY_CASE else row.case_scope
    return IdempotencyRecord(
        route=row.route,
        case_id=case_id,
        key=row.idem_key,
        request_sha256=row.request_sha256,
        status_code=row.status_code,
        response_body=row.response_body,
        created_at=row.created_at,
        expires_at=row.expires_at,
    )


def _expired(row: Any, now: datetime) -> bool:
    return parse_utc(row.expires_at) <= now


class SqlIdempotencyStore:
    """``IdempotencyStore`` over SQLite. ``lookup`` ignores expired rows."""

    def __init__(
        self, engine: Engine, ttl_hours: int = 24, clock: Clock = utc_clock
    ) -> None:
        self._engine = engine
        self._ttl_hours = ttl_hours
        self._clock = clock

    @property
    def ttl_hours(self) -> int:
        return self._ttl_hours

    def lookup(self, route: str, case_id: str | None, key: str) -> IdempotencyRecord | None:
        now = self._clock()
        with read_transaction(self._engine) as conn:
            row = conn.execute(
                select(idempotency_records).where(
                    idempotency_records.c.route == route,
                    idempotency_records.c.case_scope == _case_scope(case_id),
                    idempotency_records.c.idem_key == key,
                )
            ).first()
        if row is None or _expired(row, now):
            return None
        return _to_record(row)

    def save(self, record: IdempotencyRecord) -> IdempotencyRecord:
        """Insert-or-return-existing; ``IdempotencyConflict`` when ``request_sha256`` differs."""
        scope = _case_scope(record.case_id)
        values = {
            "route": record.route,
            "case_scope": scope,
            "idem_key": record.key,
            "request_sha256": record.request_sha256,
            "status_code": record.status_code,
            "response_body": record.response_body,
            "created_at": record.created_at,
            "expires_at": record.expires_at,
            "expires_us": to_epoch_us(parse_utc(record.expires_at)),
        }
        now = self._clock()
        with write_transaction(self._engine) as conn:
            existing = conn.execute(
                select(idempotency_records).where(
                    idempotency_records.c.route == record.route,
                    idempotency_records.c.case_scope == scope,
                    idempotency_records.c.idem_key == record.key,
                )
            ).first()
            if existing is not None and not _expired(existing, now):
                if existing.request_sha256 != record.request_sha256:
                    raise IdempotencyConflict(
                        "Idempotency-Key was reused with a different request body",
                        details={
                            "route": record.route,
                            "key": record.key,
                            "stored_sha256": existing.request_sha256,
                        },
                    )
                return _to_record(existing)
            if existing is not None:
                conn.execute(
                    update(idempotency_records)
                    .where(idempotency_records.c.id == existing.id)
                    .values(**values)
                )
                return record
            conn.execute(insert(idempotency_records).values(**values))
        return record

    def purge_expired(self) -> int:
        now_us = to_epoch_us(self._clock())
        with write_transaction(self._engine) as conn:
            result = conn.execute(
                delete(idempotency_records).where(idempotency_records.c.expires_us <= now_us)
            )
        return int(result.rowcount or 0)
