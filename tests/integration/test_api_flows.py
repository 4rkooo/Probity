"""End-to-end API flow against in-memory fakes and fixture mock handlers."""

from __future__ import annotations

import anyio
from fastapi.testclient import TestClient
from tests.api.conftest import client, data_dir, handlers, key, run_job, services  # noqa: F401
from tests.api.test_search_jobs_assets import _searchable

from probity.api.deps import AppServices
from probity.api.mocks import PLATE_BBOX


def test_fixture_search_track_reconstruct_review_report(
    client: TestClient,  # noqa: F811
    services: AppServices,  # noqa: F811
    handlers,  # noqa: F811
) -> None:
    case = client.post(
        "/v1/cases", json={"display_name": "Fictional Case 0420"}, headers=key("flow-case")
    )
    assert case.status_code == 201
    case_id = case.json()["case_id"]

    video = client.post(
        f"/v1/cases/{case_id}/videos",
        params={"fixture_id": "demo-plate-90s"},
        headers={**key("flow-vid"), "Content-Type": "application/octet-stream"},
        content=b"",
    )
    assert video.status_code == 202, video.text
    video_id = video.json()["video_id"]
    job = client.get(f"/v1/jobs/{video.json()['job_id']}")
    assert job.status_code == 200

    stored = client.get(f"/v1/videos/{video_id}")
    assert stored.json()["ingest_state"] == "STORED"
    _searchable(services, video_id)

    search = client.post(
        f"/v1/videos/{video_id}/searches",
        json={"query": "Find the blue sedan when its rear plate is most visible"},
        headers=key("flow-search"),
    )
    assert search.status_code == 200
    assert search.json()["status"] == "OK"

    track = client.post(
        f"/v1/videos/{video_id}/tracks",
        json={
            "subject_type": "LICENSE_PLATE",
            "seed_frame_id": f"{video_id}:f417",
            "seed_bbox_px": list(PLATE_BBOX),
        },
        headers=key("flow-track"),
    )
    assert track.status_code == 202
    anyio.run(run_job, services, handlers, track.json()["job_id"])
    confirmed = client.get(f"/v1/tracks/{track.json()['track_id']}")
    assert confirmed.json()["confirmed"] is True

    recon = client.post(
        "/v1/reconstructions",
        json={
            "track_id": track.json()["track_id"],
            "target_frame_id": f"{video_id}:f417",
            "target_bbox_px": list(PLATE_BBOX),
        },
        headers=key("flow-recon"),
    )
    assert recon.status_code == 202
    anyio.run(run_job, services, handlers, recon.json()["job_id"])
    run = client.get(f"/v1/reconstructions/{recon.json()['run_id']}").json()
    assert run["state"] == "SUCCEEDED"

    origin = client.get(
        f"/v1/reconstructions/{run['run_id']}/provenance", params={"x": 219, "y": 481}
    )
    assert origin.status_code == 200

    review = client.post(
        f"/v1/reconstructions/{run['run_id']}/reviews",
        json={
            "reviewer_alias": "analyst-1",
            "decision": "APPROVE",
            "comment": "Inspected donors.",
            "reviewed_result_sha256": run["result_png_sha256"],
            "reviewed_provenance_sha256": run["provenance_sha256"],
        },
        headers=key("flow-review"),
    )
    assert review.status_code == 201

    report = client.post(
        "/v1/reports",
        json={"run_id": run["run_id"], "review_id": review.json()["review_id"]},
        headers=key("flow-report"),
    )
    assert report.status_code == 202
    anyio.run(run_job, services, handlers, report.json()["job_id"])
    fetched = client.get(f"/v1/reports/{report.json()['report_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["source_verified"] is True

    asset_id = stored.json()["source_asset_uri"].removeprefix("asset://")
    asset = client.get(f"/v1/assets/{asset_id}", headers={"Range": "bytes=0-7"})
    assert asset.status_code == 206
