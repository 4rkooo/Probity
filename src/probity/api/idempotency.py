"""POST idempotency: fingerprint, 2xx replay, 24h TTL, conflict on different hash."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from fastapi import Request
from fastapi.responses import Response
from pydantic import BaseModel

from probity.api.deps import AppServices, parse_clock
from probity.domain.api import CORRELATION_HEADER
from probity.domain.errors import IdempotencyConflict
from probity.domain.ids import canonical_json, canonical_sha256
from probity.ports import IdempotencyRecord

REPLAY_HEADER = "Idempotent-Replay"


def request_route(request: Request) -> str:
    """Concrete path, not the route template, so each resource has its own key scope."""

    return f"{request.method} {request.url.path}"


def fingerprint(
    *,
    route: str,
    case_id: str | None,
    body: dict[str, Any] | None = None,
    upload_sha256: str | None = None,
    query: dict[str, str | None] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "body": body,
        "case_id": case_id,
        "query": query,
        "route": route,
        "upload_sha256": upload_sha256,
    }
    return canonical_sha256(payload)


def _expired(record: IdempotencyRecord, now: str) -> bool:
    return parse_clock(record.expires_at) <= parse_clock(now)


def lookup_replay(
    services: AppServices,
    *,
    route: str,
    case_id: str | None,
    key: str,
    request_sha256: str,
) -> IdempotencyRecord | None:
    record = services.idempotency.lookup(route, case_id, key)
    if record is None:
        return None
    now = services.clock()
    if _expired(record, now):
        return None
    if record.request_sha256 != request_sha256:
        raise IdempotencyConflict("Idempotency-Key was reused with a different request.")
    return record


def replay_response(record: IdempotencyRecord, correlation_id: str) -> Response:
    return Response(
        content=record.response_body.encode("utf-8"),
        status_code=record.status_code,
        media_type="application/json",
        headers={REPLAY_HEADER: "true", CORRELATION_HEADER: correlation_id},
    )


def save_success(
    services: AppServices,
    *,
    route: str,
    case_id: str | None,
    key: str,
    request_sha256: str,
    status_code: int,
    body: BaseModel | dict[str, Any],
) -> None:
    if not (200 <= status_code < 300):
        return
    now = services.clock()
    expires = parse_clock(now) + timedelta(hours=services.policy.idempotency.ttl_hours)
    if isinstance(body, BaseModel):
        dumped = body.model_dump(mode="json")
    else:
        dumped = body
    record = IdempotencyRecord(
        route=route,
        case_id=case_id,
        key=key,
        request_sha256=request_sha256,
        status_code=status_code,
        response_body=canonical_json(dumped).decode("utf-8"),
        created_at=now,
        expires_at=expires.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    )
    saved = services.idempotency.save(record)
    if saved.request_sha256 != request_sha256:
        raise IdempotencyConflict("Idempotency-Key was reused with a different request.")
