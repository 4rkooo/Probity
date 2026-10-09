"""Application service bundle and default-wiring hook.

Coordinator (or tests) inject ``AppServices`` via ``create_app(services=...)``.
``build_default_services`` is the production hook; until persistence/search/media
are wired it raises so the API process still imports without a database.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from fastapi import Request

from probity.config import Settings, get_settings
from probity.domain.errors import SponsorUnavailable
from probity.domain.fixtures import FixtureCatalog
from probity.domain.ids import new_uuid7, utc_now
from probity.domain.models import AdapterHealth
from probity.domain.policy import PolicyConfig, load_policy
from probity.ports import (
    IdempotencyStore,
    JobStore,
    MediaProber,
    Repository,
    SearchService,
    SourceStore,
)


class Clock(Protocol):
    def __call__(self) -> str: ...


class IdFactory(Protocol):
    def __call__(self, unix_ms: int | None = None) -> str: ...


class AdapterHealthProbe(Protocol):
    async def __call__(self) -> Sequence[AdapterHealth]: ...


@dataclass(frozen=True, slots=True)
class AppServices:
    """Ports and process config consumed by the FastAPI layer.

    Fields
    ------
    settings: process settings (``PROBITY_*``).
    policy: parsed ``config/policy.demo.yaml``.
    repository: case/video/search/track/run/review/report/asset store.
    jobs: job state machine.
    idempotency: 24h POST replay cache.
    source_store: immutable source bytes.
    prober: ffprobe-equivalent media probe.
    search: Agent C search service.
    fixture_catalog: verified demo catalog.
    fixture_root: ``fixtures/demo`` directory.
    adapter_health: async callable returning current adapter health rows.
    clock: RFC 3339 UTC factory (injectable in tests).
    new_id: lowercase UUIDv7 factory (injectable in tests).
    """

    settings: Settings
    policy: PolicyConfig
    repository: Repository
    jobs: JobStore
    idempotency: IdempotencyStore
    source_store: SourceStore
    prober: MediaProber
    search: SearchService
    fixture_catalog: FixtureCatalog
    fixture_root: Path
    adapter_health: AdapterHealthProbe
    clock: Clock
    new_id: IdFactory


class ServicesNotWired(SponsorUnavailable):
    """Raised by ``build_default_services`` until the coordinator wires real ports."""

    def __init__(self, message: str = "Application services are not wired") -> None:
        super().__init__(message)


def load_fixture_catalog(path: Path) -> FixtureCatalog:
    return FixtureCatalog.model_validate_json(path.read_text(encoding="utf-8"))


def build_default_services() -> AppServices:
    """Production wiring hook.

    Must not open a database or import live adapters as a side effect of
    ``from probity.api.main import app``. Until Agent B/C/A implementations are
    composed here, this raises ``ServicesNotWired`` (HTTP 503 retryable).
    """

    get_settings()
    load_policy()
    raise ServicesNotWired(
        "Default AppServices are not wired; inject via create_app(services=...) "
        "or implement build_default_services in the coordinator merge"
    )


def optional_services(request: Request) -> AppServices | None:
    return getattr(request.app.state, "services", None)


def require_services(request: Request) -> AppServices:
    services = optional_services(request)
    if services is None:
        raise ServicesNotWired()
    return services


def default_clock() -> str:
    return utc_now()


def default_new_id(unix_ms: int | None = None) -> str:
    return new_uuid7(unix_ms)


def parse_clock(value: str) -> datetime:
    from probity.domain.ids import parse_utc

    return parse_utc(value)
