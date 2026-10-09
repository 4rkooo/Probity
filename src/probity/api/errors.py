"""Normalized ErrorEnvelope handlers. Never echo bodies, paths, or exception text."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from probity.domain.api import CORRELATION_HEADER, ErrorBody, ErrorEnvelope
from probity.domain.enums import RETRYABLE_ERROR_CODES, ErrorCode
from probity.domain.errors import ProbityError
from probity.domain.ids import is_uuid7, new_uuid7

GENERIC_INTERNAL = "An internal error occurred."
IDEMPOTENCY_HEADER = "Idempotency-Key"


def correlation_id_of(request: Request) -> str:
    cid = getattr(request.state, "correlation_id", None)
    if isinstance(cid, str) and is_uuid7(cid):
        return cid
    raw = request.headers.get(CORRELATION_HEADER)
    if raw and is_uuid7(raw):
        return raw
    return new_uuid7()


def envelope(
    *,
    code: ErrorCode,
    message: str,
    correlation_id: str,
    retryable: bool | None = None,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    retry = code in RETRYABLE_ERROR_CODES if retryable is None else retryable
    body = ErrorEnvelope(
        error=ErrorBody(
            code=code,
            message=message[:500],
            retryable=retry,
            correlation_id=correlation_id,
            details=details or {},
        )
    )
    return body.model_dump(mode="json")


def json_error(
    request: Request,
    *,
    status_code: int,
    code: ErrorCode,
    message: str,
    retryable: bool | None = None,
    details: dict[str, object] | None = None,
) -> JSONResponse:
    cid = correlation_id_of(request)
    return JSONResponse(
        status_code=status_code,
        content=envelope(
            code=code, message=message, correlation_id=cid, retryable=retryable, details=details
        ),
        headers={CORRELATION_HEADER: cid},
    )


def _missing_idempotency_key(exc: RequestValidationError) -> bool:
    for err in exc.errors():
        loc = tuple(err.get("loc", ()))
        err_type = str(err.get("type", ""))
        if loc == ("header", IDEMPOTENCY_HEADER) and err_type == "missing":
            return True
    return False


def _sanitized_validation(exc: RequestValidationError) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for err in exc.errors():
        loc = err.get("loc", ())
        out.append(
            {
                "loc": [str(part) for part in loc],
                "msg": str(err.get("msg", "invalid")),
                "type": str(err.get("type", "value_error")),
            }
        )
    return out


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProbityError)
    async def probity_error_handler(request: Request, exc: ProbityError) -> JSONResponse:
        return json_error(
            request,
            status_code=exc.http_status,
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
            details=dict(exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        if _missing_idempotency_key(exc):
            return json_error(
                request,
                status_code=422,
                code=ErrorCode.IDEMPOTENCY_KEY_REQUIRED,
                message="Idempotency-Key header is required for mutating requests.",
            )
        return json_error(
            request,
            status_code=422,
            code=ErrorCode.VALIDATION_FAILED,
            message="Request validation failed.",
            details={"errors": _sanitized_validation(exc)},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        status = exc.status_code
        if status == 404:
            return json_error(
                request, status_code=404, code=ErrorCode.NOT_FOUND, message="Record not found."
            )
        if status == 405:
            return json_error(
                request,
                status_code=405,
                code=ErrorCode.VALIDATION_FAILED,
                message="Method not allowed.",
            )
        if status == 416:
            return json_error(
                request,
                status_code=416,
                code=ErrorCode.VALIDATION_FAILED,
                message="Requested range is not satisfiable.",
            )
        if status == 413:
            return json_error(
                request,
                status_code=413,
                code=ErrorCode.PAYLOAD_TOO_LARGE,
                message="Payload exceeds the configured maximum.",
            )
        if status == 415:
            return json_error(
                request,
                status_code=415,
                code=ErrorCode.UNSUPPORTED_MEDIA,
                message="Unsupported media type.",
            )
        return json_error(
            request, status_code=500, code=ErrorCode.INTERNAL_ERROR, message=GENERIC_INTERNAL
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, _exc: Exception) -> JSONResponse:
        return json_error(
            request, status_code=500, code=ErrorCode.INTERNAL_ERROR, message=GENERIC_INTERNAL
        )
