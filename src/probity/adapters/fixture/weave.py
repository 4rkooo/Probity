"""Fixture trace sink for Weave observability (Person 3).

Implements TraceSink and SpanContext protocols using a local in-memory/JSONL sink.
Strictly sanitizes all span attributes to enforce privacy guarantees.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any

from probity.domain.models import ScalarValue
from probity.ports import EvaluationRow, SpanContext, TraceSink

FORBIDDEN_KEY_SUBSTRINGS = (
    "video",
    "crop",
    "image",
    "frame_bytes",
    "secret",
    "token",
    "auth",
    "password",
    "notes",
    "comment",
    "api_key",
)


def sanitize_attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    """Strip any video, image, crop, secret, or sensitive user note attributes."""
    safe: dict[str, Any] = {}
    for k, v in attrs.items():
        k_lower = k.lower()
        if any(bad in k_lower for bad in FORBIDDEN_KEY_SUBSTRINGS):
            continue
        if isinstance(v, (bytes, bytearray)):
            continue
        if isinstance(v, str) and len(v) > 500:
            # Drop excessively large blobs/strings that might be base64 images
            continue
        safe[k] = v
    return safe


class FixtureSpanContext(SpanContext):
    """Context manager for local trace spans."""

    def __init__(self, name: str, safe_attributes: dict[str, object], sink: WeaveFixtureAdapter) -> None:
        self.name = name
        self.attributes = sanitize_attributes(safe_attributes)
        self.sink = sink

    def set_attribute(self, key: str, value: ScalarValue) -> None:
        safe_update = sanitize_attributes({key: value})
        if safe_update:
            self.attributes.update(safe_update)

    def __enter__(self) -> FixtureSpanContext:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_val is not None:
            self.attributes["error"] = str(exc_val)
        self.sink.record_span(self.name, self.attributes)


class WeaveFixtureAdapter(TraceSink):
    """Local fixture implementation of Weave tracing."""

    def __init__(self, log_path: Path | str | None = None) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.spans: list[dict[str, Any]] = []
        self.evaluations: list[EvaluationRow] = []

    def span(self, name: str, safe_attributes: dict[str, object]) -> SpanContext:
        return FixtureSpanContext(name, safe_attributes, self)

    def record_span(self, name: str, attributes: dict[str, Any]) -> None:
        record = {"span_name": name, "attributes": attributes}
        self.spans.append(record)
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    async def log_evaluation(self, row: EvaluationRow) -> None:
        self.evaluations.append(row)
