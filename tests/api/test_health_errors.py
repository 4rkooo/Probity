from __future__ import annotations

from fastapi.testclient import TestClient

from probity.api.deps import AppServices
from probity.api.main import create_app
from probity.domain.api import CORRELATION_HEADER
from probity.domain.ids import is_uuid7, new_uuid7
from tests.api.conftest import key
from tests.api.fakes import ExplodingRepository


def test_health_ok(client: TestClient) -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["process"] == "api"
    assert body["database"] == "ok"
    assert body["fixtures"]
    assert is_uuid7(response.headers[CORRELATION_HEADER])


def test_health_unwired_is_degraded_without_db() -> None:
    response = TestClient(create_app(), raise_server_exceptions=False).get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["database"] == "unavailable"
    assert body["status"] == "degraded"


def test_unwired_mutating_returns_503() -> None:
    client = TestClient(create_app(), raise_server_exceptions=False)
    response = client.post(
        "/v1/cases",
        json={"display_name": "Fictional Case"},
        headers=key(),
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "SPONSOR_UNAVAILABLE"
    assert response.json()["error"]["retryable"] is True
    assert is_uuid7(response.headers[CORRELATION_HEADER])


def test_missing_idempotency_key_422(client: TestClient) -> None:
    response = client.post("/v1/cases", json={"display_name": "Fictional Case"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "IDEMPOTENCY_KEY_REQUIRED"
    assert "input" not in error["details"]
    assert is_uuid7(response.headers[CORRELATION_HEADER])


def test_validation_sanitizes_loc_msg_type(client: TestClient) -> None:
    response = client.post("/v1/cases", json={"display_name": ""}, headers=key())
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_FAILED"
    details = error["details"]["errors"]
    assert details
    for item in details:
        assert set(item) <= {"loc", "msg", "type"}
        assert "input" not in item


def test_correlation_accepts_uuid7_and_rejects_other(client: TestClient) -> None:
    good = new_uuid7()
    ok = client.get("/v1/health", headers={CORRELATION_HEADER: good})
    assert ok.headers[CORRELATION_HEADER] == good
    bad = client.get("/v1/health", headers={CORRELATION_HEADER: "not-a-uuid"})
    echoed = bad.headers[CORRELATION_HEADER]
    assert echoed != "not-a-uuid"
    assert is_uuid7(echoed)


def test_404_and_405_envelopes(client: TestClient) -> None:
    missing = client.get(f"/v1/cases/{new_uuid7()}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "NOT_FOUND"
    assert is_uuid7(missing.headers[CORRELATION_HEADER])

    wrong = client.get("/v1/cases")
    assert wrong.status_code == 405
    assert wrong.json()["error"]["code"] == "VALIDATION_FAILED"
    assert is_uuid7(wrong.headers[CORRELATION_HEADER])


def test_500_sanitizes_secret_path(services: AppServices) -> None:
    exploding = ExplodingRepository()
    object.__setattr__(services, "repository", exploding)  # frozen dataclass
    services = AppServices(
        settings=services.settings,
        policy=services.policy,
        repository=exploding,
        jobs=services.jobs,
        idempotency=services.idempotency,
        source_store=services.source_store,
        prober=services.prober,
        search=services.search,
        fixture_catalog=services.fixture_catalog,
        fixture_root=services.fixture_root,
        adapter_health=services.adapter_health,
        clock=services.clock,
        new_id=services.new_id,
    )
    client = TestClient(create_app(services), raise_server_exceptions=False)
    response = client.get(f"/v1/cases/{new_uuid7()}")
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert "/secret/keys/prod.token" not in response.text
    assert "prod.token" not in response.text
    assert is_uuid7(response.headers[CORRELATION_HEADER])
