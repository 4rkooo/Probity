"""Process composition for the API and the leased worker.

Live VAST and Cosmos shells stay disabled unless the matching ``PROBITY_*_ENABLED``
flag is set and an endpoint is configured. This module does not invent sponsor URLs.
"""

from __future__ import annotations

from collections.abc import Sequence

from probity.adapters.fallback import FallbackEvidenceStore, FallbackVideoUnderstanding
from probity.adapters.fixture.cosmos import FixtureVideoUnderstanding
from probity.adapters.fixture.vast import FixtureEvidenceStore
from probity.adapters.live.cosmos import LiveCosmosUnderstanding
from probity.adapters.live.vast import LiveVastEvidenceStore
from probity.api.deps import AppServices, default_clock, default_new_id, load_fixture_catalog
from probity.api.mocks import mock_handlers
from probity.config import Settings, get_settings
from probity.domain.enums import InferenceMode, OperatingMode
from probity.domain.models import AdapterHealth
from probity.domain.policy import load_policy
from probity.ingest import IngestHandler
from probity.jobs import SqlJobStore
from probity.media.ffprobe import FfprobeMediaProber
from probity.media.segmentation import DeterministicSegmenter
from probity.media.source_store import LocalSourceStore
from probity.media.thumbnails import FfmpegThumbnailExtractor
from probity.persistence import (
    SqlIdempotencyStore,
    SqlRepository,
    create_engine_for,
    run_migrations,
    utc_clock,
)
from probity.ports import JobHandler
from probity.search.service import LocalSearchService
from probity.search.sqlite_store import SqlEvidenceStore


def _live_ready(enabled: bool, endpoint: str | None) -> bool:
    return bool(enabled and endpoint)


def build_services(settings: Settings | None = None) -> AppServices:
    """Open the configured database and share one evidence store with search."""
    settings = settings or get_settings()
    policy = load_policy(settings.policy_path)
    catalog = load_fixture_catalog(settings.fixture_manifest)
    source_store = LocalSourceStore(settings.data_dir)
    url = settings.resolved_database_url
    run_migrations(url)
    engine = create_engine_for(url)
    repository = SqlRepository(engine, clock=utc_clock)
    job_mode = InferenceMode.LIVE if settings.mode is OperatingMode.LIVE else InferenceMode.FIXTURE
    jobs = SqlJobStore(engine, clock=utc_clock, lease_seconds=settings.lease_seconds, mode=job_mode)
    idempotency = SqlIdempotencyStore(engine, clock=utc_clock)

    fixture_understanding = FixtureVideoUnderstanding(
        settings.fixture_root, settings.fixture_manifest
    )
    live_understanding = LiveCosmosUnderstanding(
        enabled=_live_ready(settings.cosmos_enabled, settings.cosmos_endpoint),
        endpoint=settings.cosmos_endpoint,
        token=settings.cosmos_token,
        model_id=settings.cosmos_model_id,
    )
    understanding = FallbackVideoUnderstanding(
        live_understanding, fixture_understanding, settings.mode
    )

    fixture_index = FixtureEvidenceStore()
    searchable = SqlEvidenceStore(repository, fixture_index)
    live_store = LiveVastEvidenceStore(
        enabled=_live_ready(settings.vast_enabled, settings.vast_endpoint),
        endpoint=settings.vast_endpoint,
        token=settings.vast_token,
    )
    evidence = FallbackEvidenceStore(live_store, searchable, settings.mode)
    search = LocalSearchService(understanding, evidence, policy)

    async def adapter_health() -> tuple[AdapterHealth, ...]:
        return (await understanding.health(), await evidence.health())

    return AppServices(
        settings=settings,
        policy=policy,
        repository=repository,
        jobs=jobs,
        idempotency=idempotency,
        source_store=source_store,
        prober=FfprobeMediaProber(policy),
        search=search,
        fixture_catalog=catalog,
        fixture_root=settings.fixture_root,
        adapter_health=adapter_health,
        clock=default_clock,
        new_id=default_new_id,
        understanding=understanding,
        evidence_store=evidence,
    )


def handler_factory(settings: Settings) -> Sequence[JobHandler]:
    """INGEST plus the fixture TRACK / RECONSTRUCT / REPORT mocks, on this process's DB."""
    services = build_services(settings)
    evidence = services.evidence_store
    understanding = services.understanding
    if evidence is None or understanding is None:
        raise RuntimeError("default services did not attach search adapters")
    mocks = mock_handlers(
        repository=services.repository,
        source_store=services.source_store,
        fixture_root=services.fixture_root,
        clock=services.clock,
        new_id=services.new_id,
        policy=services.policy,
        settings=services.settings,
    )
    ingest = IngestHandler(
        repository=services.repository,
        source_store=services.source_store,
        prober=services.prober,
        segmenter=DeterministicSegmenter(services.policy),
        thumbnails=FfmpegThumbnailExtractor(services.policy),
        understanding=understanding,
        evidence_store=evidence,
        new_id=services.new_id,
    )
    return [ingest, *mocks.values()]
