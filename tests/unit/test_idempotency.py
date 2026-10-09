"""Idempotency replay, conflict, 24h clock expiry, and Hypothesis invariants."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from probity.domain.errors import IdempotencyConflict
from probity.domain.ids import format_utc, new_uuid7
from probity.persistence.db import create_engine_for, run_migrations
from probity.persistence.idempotency import SqlIdempotencyStore, request_fingerprint
from probity.ports import IdempotencyRecord


class _Clock:
    def __init__(self) -> None:
        self.t = datetime(2026, 10, 9, 16, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.t

    def advance(self, hours: float) -> None:
        self.t = self.t + timedelta(hours=hours)


def _store(tmp_path: Path, clock: _Clock | None = None) -> SqlIdempotencyStore:
    url = f"sqlite:///{tmp_path / 'cases.db'}"
    engine = create_engine_for(url)
    run_migrations(url, engine=engine)
    return SqlIdempotencyStore(engine, ttl_hours=24, clock=clock or _Clock())


def _record(
    clock: _Clock,
    *,
    key: str = "idem-key-1",
    body: object = {"n": 1},
    route: str = "POST /v1/cases",
    case_id: str | None = None,
) -> IdempotencyRecord:
    created = format_utc(clock())
    expires = format_utc(clock() + timedelta(hours=24))
    return IdempotencyRecord(
        route=route,
        case_id=case_id,
        key=key,
        request_sha256=request_fingerprint(route, case_id, body),
        status_code=201,
        response_body="{}",
        created_at=created,
        expires_at=expires,
    )


def test_replay_same_key_and_hash_returns_original(tmp_path: Path) -> None:
    clock = _Clock()
    store = _store(tmp_path, clock)
    first = store.save(_record(clock, body={"a": 1}))
    again = store.save(_record(clock, body={"a": 1}))
    assert again.created_at == first.created_at
    assert again.request_sha256 == first.request_sha256
    looked = store.lookup("POST /v1/cases", None, "idem-key-1")
    assert looked is not None
    assert looked.request_sha256 == first.request_sha256


def test_same_key_different_hash_conflicts(tmp_path: Path) -> None:
    clock = _Clock()
    store = _store(tmp_path, clock)
    store.save(_record(clock, body={"a": 1}))
    with pytest.raises(IdempotencyConflict):
        store.save(_record(clock, body={"a": 2}))


def test_lookup_ignores_expired_and_save_replaces(tmp_path: Path) -> None:
    clock = _Clock()
    store = _store(tmp_path, clock)
    store.save(_record(clock, body={"a": 1}))
    clock.advance(24)
    assert store.lookup("POST /v1/cases", None, "idem-key-1") is None
    replaced = store.save(_record(clock, body={"a": 2}))
    assert replaced.request_sha256 == request_fingerprint("POST /v1/cases", None, {"a": 2})
    assert store.purge_expired() >= 0


def test_case_scope_and_empty_case_are_distinct(tmp_path: Path) -> None:
    clock = _Clock()
    store = _store(tmp_path, clock)
    case_id = new_uuid7()
    store.save(_record(clock, case_id=None, body={"x": 1}))
    store.save(_record(clock, case_id=case_id, body={"x": 2}))
    none_rec = store.lookup("POST /v1/cases", None, "idem-key-1")
    case_rec = store.lookup("POST /v1/cases", case_id, "idem-key-1")
    assert none_rec is not None and case_rec is not None
    assert none_rec.request_sha256 != case_rec.request_sha256


def test_ttl_hours_default_is_24(tmp_path: Path) -> None:
    assert _store(tmp_path).ttl_hours == 24


@given(
    key=st.from_regex(r"[A-Za-z0-9_-]{8,32}", fullmatch=True),
    n=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=40, deadline=None)
def test_fingerprint_stable_and_conflict_property(key: str, n: int) -> None:
    route = "POST /v1/videos/{video_id}/searches"
    case_id = "01a12164-8c00-7bfc-9416-0a1e8674dac6"
    body = {"query": "plate", "n": n}
    a = request_fingerprint(route, case_id, body)
    b = request_fingerprint(route, case_id, dict(body))
    assert a == b
    assert a != request_fingerprint(route, case_id, {"query": "plate", "n": n + 1})
    assert len(a) == 64
    assert len(key) >= 8
