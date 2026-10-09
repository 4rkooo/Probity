from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from probity.api.deps import AppServices
from probity.domain.errors import PayloadTooLarge
from tests.api.conftest import FIXTURE_ROOT, key
from tests.api.fakes import CountingStream, FakeSourceStore

DEMO_MP4 = FIXTURE_ROOT / "source" / "demo-plate-90s.mp4"


def _create_case(client: TestClient, name: str = "Fictional Case 0420") -> str:
    response = client.post("/v1/cases", json={"display_name": name}, headers=key("case-key-1"))
    assert response.status_code == 201, response.text
    return response.json()["case_id"]


def test_create_and_get_case(client: TestClient) -> None:
    created = client.post(
        "/v1/cases",
        json={"display_name": "Fictional Case", "owner_alias": "analyst-1"},
        headers=key(),
    )
    assert created.status_code == 201
    body = created.json()
    assert body["purpose"] == "DEMO_RESEARCH"
    assert body["status"] == "ACTIVE"
    fetched = client.get(f"/v1/cases/{body['case_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["case_id"] == body["case_id"]


def test_case_idempotent_replay_does_not_create_second(
    client: TestClient, services: AppServices
) -> None:
    headers = key("same-case-key")
    payload = {"display_name": "Replay Case"}
    first = client.post("/v1/cases", json=payload, headers=headers)
    second = client.post("/v1/cases", json=payload, headers=headers)
    assert first.status_code == 201
    assert second.status_code == 201
    assert second.headers.get("Idempotent-Replay") == "true"
    assert first.json()["case_id"] == second.json()["case_id"]
    assert len(services.repository.cases) == 1


def test_case_idempotent_conflict_on_different_body(client: TestClient) -> None:
    headers = key("conflict-case")
    client.post("/v1/cases", json={"display_name": "One"}, headers=headers)
    conflict = client.post("/v1/cases", json={"display_name": "Two"}, headers=headers)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"


def test_fixture_select_upload(client: TestClient) -> None:
    case_id = _create_case(client)
    response = client.post(
        f"/v1/cases/{case_id}/videos",
        params={"fixture_id": "demo-plate-90s"},
        headers={**key("fixture-up"), "Content-Type": "application/octet-stream"},
        content=b"",
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["sha256"] == "9806895754d3af16f50ba66a99514f4b7c1b6659b278fdf7f8260cfd12bf05e5"
    video = client.get(f"/v1/videos/{body['video_id']}")
    assert video.status_code == 200
    assert video.json()["ingest_state"] == "STORED"
    assert video.json()["fixture_id"] == "demo-plate-90s"


def test_unknown_fixture_404(client: TestClient) -> None:
    case_id = _create_case(client)
    response = client.post(
        f"/v1/cases/{case_id}/videos",
        params={"fixture_id": "missing-clip"},
        headers={**key("missing-fx"), "Content-Type": "application/octet-stream"},
        content=b"",
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "FIXTURE_NOT_FOUND"


def test_raw_mp4_upload_and_duplicate_bytes(client: TestClient) -> None:
    case_id = _create_case(client)
    payload = DEMO_MP4.read_bytes()
    first = client.post(
        f"/v1/cases/{case_id}/videos",
        headers={**key("raw-1"), "Content-Type": "video/mp4"},
        content=payload,
    )
    second = client.post(
        f"/v1/cases/{case_id}/videos",
        headers={**key("raw-2"), "Content-Type": "video/mp4"},
        content=payload,
    )
    assert first.status_code == 202, first.text
    assert second.status_code == 202, second.text
    assert first.json()["sha256"] == second.json()["sha256"]
    assert first.json()["video_id"] != second.json()["video_id"]
    assert second.json()["deduplicated"] is True
    v1 = client.get(f"/v1/videos/{first.json()['video_id']}").json()
    v2 = client.get(f"/v1/videos/{second.json()['video_id']}").json()
    assert v1["storage_uri"] == v2["storage_uri"]


def test_upload_idempotent_replay_discards_staging(
    client: TestClient, services: AppServices
) -> None:
    case_id = _create_case(client)
    payload = DEMO_MP4.read_bytes()
    headers = {**key("up-replay"), "Content-Type": "video/mp4"}
    first = client.post(f"/v1/cases/{case_id}/videos", headers=headers, content=payload)
    jobs_before = len(services.jobs.jobs)
    second = client.post(f"/v1/cases/{case_id}/videos", headers=headers, content=payload)
    assert first.status_code == 202
    assert second.status_code == 202
    assert second.headers.get("Idempotent-Replay") == "true"
    assert first.json()["job_id"] == second.json()["job_id"]
    assert len(services.jobs.jobs) == jobs_before
    staging = Path(services.settings.data_dir) / "staging"
    leftovers = [p for p in staging.glob("*") if p.is_file()] if staging.exists() else []
    assert leftovers == []


def test_content_length_over_limit_413_does_not_read(
    client: TestClient, services: AppServices
) -> None:
    from probity.api.deps import AppServices as Services
    from probity.api.main import create_app

    tight_video = services.policy.video.model_copy(update={"max_bytes": 64})
    tight_policy = services.policy.model_copy(update={"video": tight_video})
    tight = Services(
        settings=services.settings,
        policy=tight_policy,
        repository=services.repository,
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
    local = TestClient(create_app(tight), raise_server_exceptions=False)
    case_id = _create_case(local)
    store = tight.source_store
    assert isinstance(store, FakeSourceStore)
    store.bytes_read = 0
    store.stage_calls = 0
    response = local.post(
        f"/v1/cases/{case_id}/videos",
        headers={**key("huge"), "Content-Type": "video/mp4"},
        content=b"x" * 128,
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"
    assert store.stage_calls == 0
    assert store.bytes_read == 0


def test_store_stops_reading_past_max_bytes(services: AppServices) -> None:
    store = services.source_store
    assert isinstance(store, FakeSourceStore)
    limit = 1024
    payload = b"a" * (limit + 5000)
    stream = CountingStream(payload)
    with pytest.raises(PayloadTooLarge):
        store.stage_upload(stream, limit)
    assert stream.bytes_read <= limit + 65536
    assert store.bytes_read <= limit


def test_unsupported_media_415(client: TestClient) -> None:
    case_id = _create_case(client)
    response = client.post(
        f"/v1/cases/{case_id}/videos",
        headers={**key("json-up"), "Content-Type": "application/json"},
        content=b"{}",
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA"


def test_garbage_mp4_415(client: TestClient) -> None:
    case_id = _create_case(client)
    response = client.post(
        f"/v1/cases/{case_id}/videos",
        headers={**key("garbage"), "Content-Type": "video/mp4"},
        content=b"not-mp4!",
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA"


def test_missing_case_upload_404(client: TestClient) -> None:
    from probity.domain.ids import new_uuid7

    response = client.post(
        f"/v1/cases/{new_uuid7()}/videos",
        params={"fixture_id": "demo-plate-90s"},
        headers={**key("no-case"), "Content-Type": "application/octet-stream"},
        content=b"",
    )
    assert response.status_code == 404
