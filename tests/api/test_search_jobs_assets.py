from __future__ import annotations

from fastapi.testclient import TestClient

from probity.api.deps import AppServices
from probity.domain.enums import IngestState
from probity.domain.ids import new_uuid7
from probity.domain.models import TimeRangeUs
from tests.api.conftest import key
from tests.api.test_cases_videos import DEMO_MP4, _create_case


def _stored_video(client: TestClient, services: AppServices) -> tuple[str, str]:
    case_id = _create_case(client)
    uploaded = client.post(
        f"/v1/cases/{case_id}/videos",
        params={"fixture_id": "demo-plate-90s"},
        headers={**key("vid-1"), "Content-Type": "application/octet-stream"},
        content=b"",
    )
    assert uploaded.status_code == 202, uploaded.text
    video_id = uploaded.json()["video_id"]
    job_id = uploaded.json()["job_id"]
    return video_id, job_id


def _searchable(services: AppServices, video_id: str) -> None:
    video = services.repository.get_video(video_id)
    services.repository.put_video(
        video.revise(
            ingest_state=IngestState.SEARCHABLE,
            indexed_range=TimeRangeUs(start_pts_us=0, end_pts_us=video.duration_us),
        )
    )


def test_same_idempotency_key_cancels_each_job(client: TestClient) -> None:
    case_id = _create_case(client)
    first_upload = client.post(
        f"/v1/cases/{case_id}/videos",
        params={"fixture_id": "demo-plate-90s"},
        headers={**key("vid-a"), "Content-Type": "application/octet-stream"},
        content=b"",
    )
    second_upload = client.post(
        f"/v1/cases/{case_id}/videos",
        params={"fixture_id": "demo-plate-90s"},
        headers={**key("vid-b"), "Content-Type": "application/octet-stream"},
        content=b"",
    )
    assert first_upload.status_code == 202, first_upload.text
    assert second_upload.status_code == 202, second_upload.text
    job_a = first_upload.json()["job_id"]
    job_b = second_upload.json()["job_id"]

    first = client.post(f"/v1/jobs/{job_a}/cancel", headers=key("same-cancel-key"))
    second = client.post(f"/v1/jobs/{job_b}/cancel", headers=key("same-cancel-key"))
    assert first.status_code == 202, first.text
    assert second.status_code == 202, second.text
    assert first.headers.get("Idempotent-Replay") is None
    assert second.headers.get("Idempotent-Replay") is None
    assert first.json()["job_id"] == job_a
    assert second.json()["job_id"] == job_b
    assert client.get(f"/v1/jobs/{job_b}").json()["state"] == "CANCELLING"

    replay = client.post(f"/v1/jobs/{job_a}/cancel", headers=key("same-cancel-key"))
    assert replay.status_code == 202
    assert replay.headers.get("Idempotent-Replay") == "true"
    assert replay.json()["job_id"] == job_a


def test_get_job_and_cancel(client: TestClient, services: AppServices) -> None:
    _video_id, job_id = _stored_video(client, services)
    fetched = client.get(f"/v1/jobs/{job_id}")
    assert fetched.status_code == 200
    assert fetched.json()["state"] == "QUEUED"
    cancelled = client.post(f"/v1/jobs/{job_id}/cancel", headers=key("cancel-1"))
    assert cancelled.status_code == 202
    assert cancelled.json()["state"] == "CANCELLING"


def test_search_not_searchable_409(client: TestClient, services: AppServices) -> None:
    video_id, _ = _stored_video(client, services)
    response = client.post(
        f"/v1/videos/{video_id}/searches",
        json={"query": "Find the blue sedan"},
        headers=key("search-early"),
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "VIDEO_NOT_SEARCHABLE"


def test_search_ok_and_clarification(client: TestClient, services: AppServices) -> None:
    video_id, _ = _stored_video(client, services)
    _searchable(services, video_id)
    ok = client.post(
        f"/v1/videos/{video_id}/searches",
        json={"query": "Find the blue sedan when its rear plate is most visible"},
        headers=key("search-ok"),
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["status"] == "OK"
    assert ok.json()["results"]
    fetched = client.get(f"/v1/searches/{ok.json()['search_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["search_id"] == ok.json()["search_id"]

    clarify = client.post(
        f"/v1/videos/{video_id}/searches",
        json={"query": "Who is the guilty driver and why did they flee?"},
        headers=key("search-clarify"),
    )
    assert clarify.status_code == 200
    assert clarify.json()["status"] == "NEEDS_CLARIFICATION"
    assert clarify.json()["results"] == []


def test_asset_source_and_ranges(client: TestClient, services: AppServices) -> None:
    video_id, _ = _stored_video(client, services)
    video = client.get(f"/v1/videos/{video_id}").json()
    asset_id = video["source_asset_uri"].removeprefix("asset://")
    full = client.get(f"/v1/assets/{asset_id}")
    assert full.status_code == 200
    assert full.headers["Cache-Control"] == "no-store"
    assert full.headers["Accept-Ranges"] == "bytes"
    assert full.content == DEMO_MP4.read_bytes()

    ranged = client.get(f"/v1/assets/{asset_id}", headers={"Range": "bytes=0-15"})
    assert ranged.status_code == 206
    assert ranged.content == DEMO_MP4.read_bytes()[:16]
    assert ranged.headers["Content-Range"].startswith("bytes 0-15/")

    unsat = client.get(f"/v1/assets/{asset_id}", headers={"Range": "bytes=99999999-99999999"})
    assert unsat.status_code == 416
    assert unsat.json()["error"]["code"] == "VALIDATION_FAILED"

    missing = client.get(f"/v1/assets/{new_uuid7()}")
    assert missing.status_code == 404


def test_asset_path_traversal_is_sanitized_500(client: TestClient, services: AppServices) -> None:
    video_id, _ = _stored_video(client, services)
    video = client.get(f"/v1/videos/{video_id}").json()
    asset_id = video["source_asset_uri"].removeprefix("asset://")
    asset = services.repository.get_asset(asset_id)
    services.repository.put_asset(
        asset.model_copy(update={"storage_uri": "source/sha256/../secret.mp4"})
    )
    response = client.get(f"/v1/assets/{asset_id}")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "secret" not in response.text
    assert "Traceback" not in response.text


def test_search_sponsor_unavailable_is_retryable_503(
    client: TestClient, services: AppServices
) -> None:
    from dataclasses import replace

    from probity.api.main import create_app
    from probity.domain.errors import SponsorUnavailable

    video_id, _ = _stored_video(client, services)
    _searchable(services, video_id)

    class DownSearch:
        async def search(self, video, query, correlation_id):  # noqa: ANN001
            raise SponsorUnavailable("Sponsor retry budget exhausted.")

    local = TestClient(
        create_app(replace(services, search=DownSearch())), raise_server_exceptions=False
    )
    response = local.post(
        f"/v1/videos/{video_id}/searches",
        json={"query": "Find the blue sedan"},
        headers=key("search-down"),
    )
    assert response.status_code == 503
    body = response.json()["error"]
    assert body["code"] == "SPONSOR_UNAVAILABLE"
    assert body["retryable"] is True
    assert "Traceback" not in response.text


def test_segments_404_then_empty_list(client: TestClient, services: AppServices) -> None:
    video_id, _ = _stored_video(client, services)
    listed = client.get(f"/v1/videos/{video_id}/segments")
    assert listed.status_code == 200
    assert listed.json() == []
    missing = client.get(f"/v1/videos/{new_uuid7()}/segments")
    assert missing.status_code == 404
