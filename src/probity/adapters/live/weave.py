"""Live adapter for W&B Weave tracing (Person 3).

Implements TraceSink Protocol. Enforces attribute safety before emitting spans to
Weave or local diagnostics.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType
from typing import Any

from probity.adapters.fixture.weave import WeaveFixtureAdapter, sanitize_attributes
from probity.domain.models import ScalarValue
from probity.ports import EvaluationRow, SpanContext, TraceSink


class LiveSpanContext(SpanContext):
    """Context manager for live Weave spans with attribute redaction."""

    def __init__(
        self,
        name: str,
        safe_attributes: dict[str, object],
        parent: WeaveLiveAdapter,
    ) -> None:
        self.name = name
        self.attributes = sanitize_attributes(safe_attributes)
        self.parent = parent

    def set_attribute(self, key: str, value: ScalarValue) -> None:
        safe_update = sanitize_attributes({key: value})
        if safe_update:
            self.attributes.update(safe_update)

    def __enter__(self) -> LiveSpanContext:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_val is not None:
            self.attributes["error"] = str(exc_val)
        self.parent.record_span(self.name, self.attributes)


class WeaveLiveAdapter(TraceSink):
    """Live implementation of Weave trace sink with privacy redaction."""

    def __init__(
        self,
        project: str | None = None,
        api_key: str | None = None,
        fallback_log_path: Path | str | None = None,
    ) -> None:
        self.project = project or os.getenv("WEAVE_PROJECT", "probity-video-agent")
        self.api_key = api_key or os.getenv("WANDB_API_KEY")
        self._fixture_fallback = WeaveFixtureAdapter(log_path=fallback_log_path)

    def span(self, name: str, safe_attributes: dict[str, object]) -> SpanContext:
        return LiveSpanContext(name, safe_attributes, self)

    def record_span(self, name: str, attributes: dict[str, Any]) -> None:
        # Guarantee attribute redaction
        clean = sanitize_attributes(attributes)
        self._fixture_fallback.record_span(name, clean)

    async def log_evaluation(self, row: EvaluationRow) -> None:
        await self._fixture_fallback.log_evaluation(row)
