"""Typed application errors. The API maps these to the normalized error envelope."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from probity.domain.enums import ERROR_HTTP_STATUS, RETRYABLE_ERROR_CODES, ErrorCode


class ProbityError(Exception):
    code: ErrorCode = ErrorCode.INTERNAL_ERROR

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        details: Mapping[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        self.message = message
        self.details: Mapping[str, Any] = MappingProxyType(dict(details or {}))
        self.retryable = self.code in RETRYABLE_ERROR_CODES if retryable is None else retryable

    @property
    def http_status(self) -> int:
        return ERROR_HTTP_STATUS[self.code]


class ValidationFailed(ProbityError):
    code = ErrorCode.VALIDATION_FAILED


class NotFound(ProbityError):
    code = ErrorCode.NOT_FOUND


class IdempotencyConflict(ProbityError):
    code = ErrorCode.IDEMPOTENCY_CONFLICT


class InvalidStateTransition(ProbityError):
    code = ErrorCode.INVALID_STATE_TRANSITION


class ArtifactHashConflict(ProbityError):
    code = ErrorCode.ARTIFACT_HASH_CONFLICT


class PayloadTooLarge(ProbityError):
    code = ErrorCode.PAYLOAD_TOO_LARGE


class UnsupportedMedia(ProbityError):
    code = ErrorCode.UNSUPPORTED_MEDIA


class SourceHashMismatch(ProbityError):
    code = ErrorCode.SOURCE_HASH_MISMATCH


class SourcePathViolation(ProbityError):
    """A write or derived path targeted the immutable source tree or escaped the data root."""

    code = ErrorCode.INTERNAL_ERROR


class SponsorTimeout(ProbityError):
    code = ErrorCode.SPONSOR_TIMEOUT


class SponsorUnavailable(ProbityError):
    code = ErrorCode.SPONSOR_UNAVAILABLE


class SponsorMalformedResponse(ProbityError):
    code = ErrorCode.SPONSOR_MALFORMED_RESPONSE


class FixtureNotFound(ProbityError):
    code = ErrorCode.FIXTURE_NOT_FOUND


class JobCancelled(ProbityError):
    """Raised by ``CancelToken.checkpoint()`` after cooperative cancellation was requested."""

    code = ErrorCode.CANCELLED
