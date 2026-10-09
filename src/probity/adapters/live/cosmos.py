"""Live Cosmos adapter shell.

No official Cosmos/CoreWeave client, endpoint contract, or SDK is bound here. An optional
``CosmosTransport`` may be injected so the official client can be wired once event docs exist.
Sponsor-specific types never leave this module.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol

import anyio
from pydantic import SecretStr, ValidationError

from probity.domain.enums import AdapterMode, HealthStatus
from probity.domain.errors import SponsorMalformedResponse, SponsorTimeout, SponsorUnavailable
from probity.domain.ids import utc_now
from probity.domain.models import AdapterHealth, Embedding, SegmentDescription
from probity.ports import ClipReference

SleepFn = Callable[[float], Awaitable[None]]
READ_RETRY_DELAYS_S: tuple[float, ...] = (2.0, 5.0)
COSMOS_TIMEOUT_S = 90.0
HEALTH_BUDGET_S = 2.0


class CosmosTransport(Protocol):
    """Probity-owned injection point. Bind the official client once docs exist."""

    async def execute(self, operation: str, payload: Mapping[str, Any]) -> Any:
        """Return JSON-compatible data for a Probity protocol operation.

        May raise ``TimeoutError``, ``SponsorTimeout``, ``SponsorUnavailable``, or
        ``SponsorMalformedResponse``. Operation names are Probity protocol methods
        (``health``, ``describe``, ``embed_segments``, ``embed_query``), not vendor APIs.
        """
        ...


class LiveCosmosUnderstanding:
    """VideoUnderstanding shell. Disabled unless ``enabled`` and a transport is bound."""

    adapter_name = "cosmos-live"
    schema_version = "1.0"

    def __init__(
        self,
        *,
        enabled: bool,
        endpoint: str | None = None,
        token: SecretStr | None = None,
        transport: CosmosTransport | None = None,
        model_id: str | None = None,
        timeout_s: float = COSMOS_TIMEOUT_S,
        read_retry_delays_s: tuple[float, ...] = READ_RETRY_DELAYS_S,
        sleep: SleepFn | None = None,
    ) -> None:
        self._enabled = enabled
        self._endpoint = endpoint
        self._token = token
        self._transport = transport
        self._model_id = model_id
        self._timeout_s = timeout_s
        self._read_retry_delays_s = read_retry_delays_s
        self._sleep = sleep

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def mode(self) -> AdapterMode:
        if not self._enabled or self._transport is None:
            return AdapterMode.DISABLED
        return AdapterMode.LIVE

    def _bound(self) -> bool:
        return self._enabled and self._transport is not None

    async def _sleep_for(self, seconds: float) -> None:
        if self._sleep is not None:
            await self._sleep(seconds)
            return
        await anyio.sleep(seconds)

    async def _call(self, operation: str, payload: Mapping[str, Any], timeout_s: float) -> Any:
        if not self._bound() or self._transport is None:
            raise SponsorUnavailable("cosmos live adapter is disabled or has no transport bound")
        with anyio.fail_after(timeout_s):
            return await self._transport.execute(operation, payload)

    async def _read(self, operation: str, payload: Mapping[str, Any], timeout_s: float) -> Any:
        delays = self._read_retry_delays_s
        attempts = len(delays) + 1
        last_timeout = False
        last_unavailable: SponsorUnavailable | None = None
        for attempt in range(attempts):
            try:
                return await self._call(operation, payload, timeout_s)
            except SponsorMalformedResponse:
                raise
            except TimeoutError:
                last_timeout = True
            except SponsorTimeout:
                last_timeout = True
            except SponsorUnavailable as exc:
                last_unavailable = exc
            if attempt < len(delays):
                await self._sleep_for(delays[attempt])
        if last_unavailable is not None and not last_timeout:
            raise last_unavailable
        raise SponsorTimeout(f"cosmos {operation} exhausted read retry budget")

    async def health(self) -> AdapterHealth:
        started = time.perf_counter()
        if not self._enabled:
            return AdapterHealth(
                adapter_name=self.adapter_name,
                model_id=self.model_id,
                mode=AdapterMode.DISABLED,
                status=HealthStatus.DISABLED,
                checked_at=utc_now(),
                latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
                detail="PROBITY_COSMOS_ENABLED is false",
            )
        if self._transport is None:
            return AdapterHealth(
                adapter_name=self.adapter_name,
                model_id=self.model_id,
                mode=AdapterMode.DISABLED,
                status=HealthStatus.UNAVAILABLE,
                checked_at=utc_now(),
                latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
                detail="no transport bound; official Cosmos client is not configured",
            )
        try:
            await self._call("health", {}, HEALTH_BUDGET_S)
            status = HealthStatus.OK
            detail = None
            mode = AdapterMode.LIVE
        except (TimeoutError, SponsorTimeout, SponsorUnavailable, SponsorMalformedResponse) as exc:
            status = HealthStatus.UNAVAILABLE
            detail = type(exc).__name__
            mode = AdapterMode.LIVE
        return AdapterHealth(
            adapter_name=self.adapter_name,
            model_id=self.model_id,
            mode=mode,
            status=status,
            checked_at=utc_now(),
            latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
            detail=detail,
        )

    async def describe(self, clip: ClipReference, prompt: str) -> SegmentDescription:
        raw = await self._read(
            "describe",
            {"clip": clip.model_dump(mode="json"), "prompt": prompt},
            self._timeout_s,
        )
        try:
            parsed = SegmentDescription.model_validate(raw)
        except ValidationError as exc:
            raise SponsorMalformedResponse(
                "cosmos describe returned an invalid SegmentDescription"
            ) from exc
        return parsed

    async def embed_segments(self, items: Sequence[SegmentDescription]) -> Sequence[Embedding]:
        out: list[Embedding] = []
        for item in items:
            raw = await self._read(
                "embed_segments",
                {"items": [item.model_dump(mode="json")]},
                self._timeout_s,
            )
            try:
                if isinstance(raw, Mapping) and "vector" in raw:
                    parsed = Embedding.model_validate(raw)
                elif isinstance(raw, Sequence) and raw:
                    parsed = Embedding.model_validate(raw[0])
                else:
                    raise SponsorMalformedResponse(
                        "cosmos embed_segments returned an invalid Embedding"
                    )
            except ValidationError as exc:
                raise SponsorMalformedResponse(
                    "cosmos embed_segments returned an invalid Embedding"
                ) from exc
            out.append(parsed)
        return tuple(out)

    async def embed_query(self, query: str) -> Embedding:
        raw = await self._read("embed_query", {"query": query}, self._timeout_s)
        try:
            if not isinstance(raw, Mapping):
                raise SponsorMalformedResponse("cosmos embed_query returned an invalid Embedding")
            return Embedding.model_validate(raw)
        except ValidationError as exc:
            raise SponsorMalformedResponse(
                "cosmos embed_query returned an invalid Embedding"
            ) from exc
