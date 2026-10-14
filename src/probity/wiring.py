"""Process composition for the API and the leased worker.

Live VAST and Cosmos shells stay disabled unless the matching ``PROBITY_*_ENABLED``
flag is set and an endpoint is configured. This module does not invent sponsor URLs.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import SecretStr

from probity.adapters.fallback import FallbackEvidenceStore, FallbackVideoUnderstanding
from probity.adapters.fixture.cosmos import FixtureVideoUnderstanding
from probity.adapters.fixture.vast import FixtureEvidenceStore
from probity.adapters.live.cosmos import LiveCosmosUnderstanding
from probity.adapters.live.vast import LiveVastEvidenceStore
from probity.adapters.live.wandb import WandbLiveAdapter
from probity.adapters.live.workshop_cosmos import WorkshopCosmosTransport
from probity.adapters.live.workshop_vast import WorkshopVastTransport
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


def _secret_value(value: SecretStr | None) -> str | None:
    if value is None:
        return None
    return value.get_secret_value()


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
        settings.fixture_root,
        settings.fixture_manifest,
        allow_synthetic=settings.synthetic_descriptions,
    )

    cosmos_token = _secret_value(settings.cosmos_token)
    cosmos_transport = None
    if (
        _live_ready(settings.cosmos_enabled, settings.cosmos_endpoint)
        and settings.cosmos_embed_endpoint
    ):
        cosmos_transport = WorkshopCosmosTransport(
            reason_url=settings.cosmos_endpoint or "",
            embed_url=settings.cosmos_embed_endpoint,
            token=cosmos_token,
            data_dir=settings.data_dir,
            reason_model_id=settings.cosmos_model_id,
            embed_model_id=settings.cosmos_embed_model_id,
        )

    live_understanding = LiveCosmosUnderstanding(
        enabled=_live_ready(settings.cosmos_enabled, settings.cosmos_endpoint)
        and cosmos_transport is not None,
        endpoint=settings.cosmos_endpoint,
        token=settings.cosmos_token,
        transport=cosmos_transport,
        model_id=settings.cosmos_model_id or None,
    )
    understanding = FallbackVideoUnderstanding(
        live_understanding, fixture_understanding, settings.mode
    )

    fixture_index = FixtureEvidenceStore()
    searchable = SqlEvidenceStore(repository, fixture_index)

    vast_transport = None
    if _live_ready(settings.vast_enabled, settings.vast_endpoint):
        vast_transport = WorkshopVastTransport(
            endpoint=settings.vast_endpoint or "",
            token=_secret_value(settings.vast_token),
            store=searchable,
        )

    live_store = LiveVastEvidenceStore(
        enabled=_live_ready(settings.vast_enabled, settings.vast_endpoint)
        and vast_transport is not None,
        endpoint=settings.vast_endpoint,
        token=settings.vast_token,
        transport=vast_transport,
    )
    evidence = FallbackEvidenceStore(live_store, searchable, settings.mode)
    reasoner = WandbLiveAdapter() if settings.wandb_enabled else None
    search = LocalSearchService(understanding, evidence, policy, reasoner=reasoner)

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
