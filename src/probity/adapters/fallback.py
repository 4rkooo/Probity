"""AUTO/LIVE/FIXTURE resolution for VideoUnderstanding and EvidenceStore."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

import anyio

from probity.domain.enums import AdapterMode, HealthStatus, OperatingMode
from probity.domain.errors import SponsorMalformedResponse, SponsorTimeout, SponsorUnavailable
from probity.domain.models import FrozenModel
from probity.ports import ClipReference, SearchRequest

HEALTH_BUDGET_S = 2.0
_FALLBACK_ERRORS = (SponsorTimeout, SponsorUnavailable, SponsorMalformedResponse)
T = TypeVar("T")


def snapshot_value(value: Any) -> Any:
    """Canonical snapshot used to prove frozen inputs are unchanged."""
    if isinstance(value, FrozenModel):
        dumped = value.model_dump(mode="json")
        return (type(value).__name__, dumped, dumped.get("content_sha256"))
    if isinstance(value, (list, tuple)):
        return [snapshot_value(item) for item in value]
    if isinstance(value, dict):
        return {key: snapshot_value(item) for key, item in value.items()}
    return value


def assert_unchanged(label: str, before: Any, after: Any) -> None:
    if before != after:
        raise RuntimeError(f"fallback adapter mutated frozen input {label}")


class FallbackVideoUnderstanding:
    """Resolves Cosmos live vs fixture once, on the first health check."""

    schema_version = "1.0"

    def __init__(
        self,
        primary: Any,
        fixture: Any,
        mode: OperatingMode,
        health_budget_s: float = HEALTH_BUDGET_S,
    ) -> None:
        self._primary = primary
        self._fixture = fixture
        self._operating_mode = mode
        self._health_budget_s = health_budget_s
        self._resolved: Any | None = None

    @property
    def adapter_name(self) -> str:
        target = self._resolved or (
            self._fixture if self._operating_mode is OperatingMode.FIXTURE else self._primary
        )
        return getattr(target, "adapter_name", "cosmos-fallback")

    @property
    def model_id(self) -> str | None:
        target = self._resolved or self._fixture
        return getattr(target, "model_id", None)

    @property
    def mode(self) -> AdapterMode:
        if self._resolved is not None:
            return self._resolved.mode
        if self._operating_mode is OperatingMode.FIXTURE:
            return AdapterMode.FIXTURE
        if self._operating_mode is OperatingMode.LIVE:
            return AdapterMode.LIVE
        return AdapterMode.DEGRADED

    async def resolve(self) -> Any:
        if self._resolved is not None:
            return self._resolved
        if self._operating_mode is OperatingMode.FIXTURE:
            self._resolved = self._fixture
            return self._resolved
        if self._operating_mode is OperatingMode.LIVE:
            self._resolved = self._primary
            return self._resolved
        try:
            with anyio.fail_after(self._health_budget_s):
                health = await self._primary.health()
            if health.status is HealthStatus.OK:
                self._resolved = self._primary
                return self._resolved
        except Exception:
            self._resolved = self._fixture
            return self._resolved
        self._resolved = self._fixture
        return self._resolved

    async def health(self) -> Any:
        target = await self.resolve()
        return await target.health()

    async def _invoke(self, method: str, *args: Any, **kwargs: Any) -> Any:
        target = await self.resolve()
        snapshots = [snapshot_value(arg) for arg in args]
        kw_snaps = {key: snapshot_value(val) for key, val in kwargs.items()}
        call: Callable[..., Awaitable[Any]] = getattr(target, method)
        error: BaseException | None = None
        result: Any = None
        try:
            result = await call(*args, **kwargs)
        except _FALLBACK_ERRORS as exc:
            error = exc
        self._assert_args(method, args, snapshots, kwargs, kw_snaps)
        if error is None:
            return result
        if (
            self._operating_mode is OperatingMode.LIVE
            or self._operating_mode is OperatingMode.FIXTURE
            or target is self._fixture
        ):
            raise error
        replay: Callable[..., Awaitable[Any]] = getattr(self._fixture, method)
        result = await replay(*args, **kwargs)
        self._resolved = self._fixture
        self._assert_args(method, args, snapshots, kwargs, kw_snaps)
        return result

    def _assert_args(
        self,
        method: str,
        args: tuple[Any, ...],
        snapshots: list[Any],
        kwargs: dict[str, Any],
        kw_snaps: dict[str, Any],
    ) -> None:
        for arg, before in zip(args, snapshots, strict=True):
            assert_unchanged(method, before, snapshot_value(arg))
        for key, before in kw_snaps.items():
            assert_unchanged(key, before, snapshot_value(kwargs[key]))

    async def describe(self, clip: ClipReference, prompt: str) -> Any:
        return await self._invoke("describe", clip, prompt)

    async def embed_segments(self, items: Sequence[Any]) -> Any:
        return await self._invoke("embed_segments", items)

    async def embed_query(self, query: str) -> Any:
        return await self._invoke("embed_query", query)


class FallbackEvidenceStore:
    """Resolves VAST live vs fixture once. Replay one failed request on fixture.

    Health is fixed at the first check. A later write does not move to fixture after
    a live write in the same store has already committed.
    """

    schema_version = "1.0"

    def __init__(
        self,
        primary: Any,
        fixture: Any,
        mode: OperatingMode,
        health_budget_s: float = HEALTH_BUDGET_S,
    ) -> None:
        self._primary = primary
        self._fixture = fixture
        self._operating_mode = mode
        self._health_budget_s = health_budget_s
        self._resolved: Any | None = None
        self._live_write_committed = False

    @property
    def adapter_name(self) -> str:
        target = self._resolved or (
            self._fixture if self._operating_mode is OperatingMode.FIXTURE else self._primary
        )
        return getattr(target, "adapter_name", "vast-fallback")

    @property
    def model_id(self) -> str | None:
        target = self._resolved or self._fixture
        return getattr(target, "model_id", None)

    @property
    def mode(self) -> AdapterMode:
        if self._resolved is not None:
            return self._resolved.mode
        if self._operating_mode is OperatingMode.FIXTURE:
            return AdapterMode.FIXTURE
        if self._operating_mode is OperatingMode.LIVE:
            return AdapterMode.LIVE
        return AdapterMode.DEGRADED

    async def resolve(self) -> Any:
        if self._resolved is not None:
            return self._resolved
        if self._operating_mode is OperatingMode.FIXTURE:
            self._resolved = self._fixture
            return self._resolved
        if self._operating_mode is OperatingMode.LIVE:
            self._resolved = self._primary
            return self._resolved
        try:
            with anyio.fail_after(self._health_budget_s):
                health = await self._primary.health()
            if health.status is HealthStatus.OK:
                self._resolved = self._primary
                return self._resolved
        except Exception:
            self._resolved = self._fixture
            return self._resolved
        self._resolved = self._fixture
        return self._resolved

    async def health(self) -> Any:
        target = await self.resolve()
        return await target.health()

    async def _invoke(self, method: str, mutating: bool, *args: Any) -> Any:
        target = await self.resolve()
        snapshots = [snapshot_value(arg) for arg in args]
        error: BaseException | None = None
        result: Any = None
        call: Callable[..., Awaitable[Any]] = getattr(target, method)
        try:
            result = await call(*args)
        except _FALLBACK_ERRORS as exc:
            error = exc
        for arg, before in zip(args, snapshots, strict=True):
            assert_unchanged(method, before, snapshot_value(arg))
        if error is None:
            if mutating and target is self._primary:
                self._live_write_committed = True
            return result
        # A committed live write must not continue on fixture. Replay is only the
        # same failed request, and only when live has not already committed.
        if (
            self._operating_mode is OperatingMode.LIVE
            or self._operating_mode is OperatingMode.FIXTURE
            or target is self._fixture
            or self._live_write_committed
        ):
            raise error
        result = await getattr(self._fixture, method)(*args)
        self._resolved = self._fixture
        for arg, before in zip(args, snapshots, strict=True):
            assert_unchanged(method, before, snapshot_value(arg))
        return result

    async def put_source(self, video: Any, local_path: str) -> str:
        return await self._invoke("put_source", True, video, local_path)

    async def upsert_segments(self, segments: Sequence[Any], embeddings: Sequence[Any]) -> None:
        await self._invoke("upsert_segments", True, segments, embeddings)

    async def upsert_detections(self, detections: Sequence[Any]) -> None:
        await self._invoke("upsert_detections", True, detections)

    async def search(self, request: SearchRequest) -> Any:
        return await self._invoke("search", False, request)

    async def put_lineage(self, record: Any) -> None:
        await self._invoke("put_lineage", True, record)

    async def put_report(self, report: Any) -> None:
        await self._invoke("put_report", True, report)
