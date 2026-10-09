"""Correlation ID echo and JSON access logs (no bodies/headers/filenames/query text)."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from probity.domain.api import CORRELATION_HEADER
from probity.domain.ids import is_uuid7, new_uuid7, utc_now

logger = logging.getLogger("probity.api")


def resolve_correlation_id(request: Request) -> str:
    raw = request.headers.get(CORRELATION_HEADER)
    if raw and is_uuid7(raw):
        return raw
    return new_uuid7()


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    return request.url.path


class CorrelationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        cid = resolve_correlation_id(request)
        request.state.correlation_id = cid
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            latency_ms = int((time.perf_counter() - started) * 1000)
            _access_log(request, status=500, latency_ms=latency_ms, correlation_id=cid)
            raise
        response.headers[CORRELATION_HEADER] = cid
        latency_ms = int((time.perf_counter() - started) * 1000)
        _access_log(request, status=status, latency_ms=latency_ms, correlation_id=cid)
        return response


def _access_log(request: Request, *, status: int, latency_ms: int, correlation_id: str) -> None:
    payload = {
        "ts": utc_now(),
        "level": "INFO",
        "component": "api",
        "event": "http_access",
        "method": request.method,
        "path": _route_path(request),
        "status": status,
        "latency_ms": latency_ms,
        "correlation_id": correlation_id,
    }
    logger.info(json.dumps(payload, separators=(",", ":"), sort_keys=True))
