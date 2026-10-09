from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from probity.api.deps import AppServices, load_fixture_catalog
from probity.api.main import create_app
from probity.api.mocks import mock_handlers
from probity.config import Settings
from probity.domain.enums import AdapterMode, HealthStatus, JobKind
from probity.domain.ids import new_uuid7, utc_now
from probity.domain.models import AdapterHealth
from probity.domain.policy import load_policy
from probity.ports import JobOutcome
from tests.api.fakes import (
    FakeIdempotencyStore,
    FakeJobStore,
    FakeMediaProber,
    FakeRepository,
    FakeSearchService,
    FakeSourceStore,
)

WORKTREE = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = WORKTREE / "fixtures" / "demo"


class NullCancel:
    def is_cancelled(self) -> bool:
        return False

    def checkpoint(self) -> None:
        return None


class TestJobContext:
    def __init__(self) -> None:
        self.cancel = NullCancel()
        self.stages: list[tuple[str, int, int]] = []

    def progress(self, stage, completed: int, total: int) -> None:
        self.stages.append((stage, completed, total))


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    return root


@pytest.fixture
def services(data_dir: Path) -> AppServices:
    settings = Settings(
        data_dir=data_dir,
        fixture_root=FIXTURE_ROOT,
        fixture_manifest=FIXTURE_ROOT / "manifest.json",
        policy_path=WORKTREE / "config" / "policy.demo.yaml",
    )
    policy = load_policy(settings.policy_path)
    repository = FakeRepository()
    jobs = FakeJobStore(clock=utc_now)
    clock = utc_now

    async def adapter_health() -> tuple[AdapterHealth, ...]:
        return (
            AdapterHealth(
                adapter_name="fixture-catalog",
                model_id=None,
                mode=AdapterMode.FIXTURE,
                status=HealthStatus.OK,
                checked_at=clock(),
                latency_ms=1,
            ),
        )

    return AppServices(
        settings=settings,
        policy=policy,
        repository=repository,
        jobs=jobs,
        idempotency=FakeIdempotencyStore(),
        source_store=FakeSourceStore(data_dir),
        prober=FakeMediaProber(),
        search=FakeSearchService(new_id=new_uuid7, clock=clock),
        fixture_catalog=load_fixture_catalog(settings.fixture_manifest),
        fixture_root=FIXTURE_ROOT,
        adapter_health=adapter_health,
        clock=clock,
        new_id=new_uuid7,
    )


@pytest.fixture
def client(services: AppServices) -> TestClient:
    return TestClient(create_app(services), raise_server_exceptions=False)


@pytest.fixture
def handlers(services: AppServices) -> dict[JobKind, object]:
    return mock_handlers(
        repository=services.repository,
        source_store=services.source_store,
        fixture_root=services.fixture_root,
        clock=services.clock,
        new_id=services.new_id,
        policy=services.policy,
        settings=services.settings,
    )


async def run_job(services: AppServices, handlers: dict, job_id: str) -> JobOutcome:
    claimed = services.jobs.claim(job_id, "test-worker")  # type: ignore[attr-defined]
    handler = handlers[claimed.payload.kind]
    outcome = await handler.run(claimed, TestJobContext())
    return services.jobs.complete(claimed.job.job_id, "test-worker", outcome)


def key(name: str = "idem-key-01") -> dict[str, str]:
    token = name if len(name) >= 8 else f"{name}-idem01"
    return {"Idempotency-Key": token}


def cid(value: str | None = None) -> dict[str, str]:
    if value is None:
        return {}
    return {"X-Correlation-ID": value}
