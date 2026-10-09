from __future__ import annotations

import hashlib

import anyio
from fastapi.testclient import TestClient

from probity.api.deps import AppServices
from probity.api.mocks import PLATE_BBOX, seed_confirms_plate
from tests.api.conftest import FIXTURE_ROOT, key, run_job
from tests.api.test_search_jobs_assets import _stored_video

CONTRACTS = FIXTURE_ROOT.parent / "contracts"
RESULT_SHA = hashlib.sha256((CONTRACTS / "artifacts" / "result.png").read_bytes()).hexdigest()
PROV_SHA = hashlib.sha256((CONTRACTS / "artifacts" / "provenance.npz").read_bytes()).hexdigest()


def _seed(video_id: str, frame: int = 417, bbox=PLATE_BBOX) -> dict:
    return {
        "subject_type": "LICENSE_PLATE",
        "seed_frame_id": f"{video_id}:f{frame}",
        "seed_bbox_px": list(bbox),
    }


def test_track_confirm_and_reject(client: TestClient, services: AppServices, handlers) -> None:
    video_id, _ = _stored_video(client, services)
    confirmed = client.post(
        f"/v1/videos/{video_id}/tracks",
        json=_seed(video_id, 417),
        headers=key("track-yes"),
    )
    assert confirmed.status_code == 202, confirmed.text
    anyio.run(run_job, services, handlers, confirmed.json()["job_id"])
    track = client.get(f"/v1/tracks/{confirmed.json()['track_id']}")
    assert track.status_code == 200
    assert track.json()["confirmed"] is True
    assert track.json()["mode"] == "FIXTURE"
    assert track.json()["video_id"] == video_id

    rejected = client.post(
        f"/v1/videos/{video_id}/tracks",
        json=_seed(video_id, 10, (0, 0, 10, 10)),
        headers=key("track-no"),
    )
    assert rejected.status_code == 202
    anyio.run(run_job, services, handlers, rejected.json()["job_id"])
    other = client.get(f"/v1/tracks/{rejected.json()['track_id']}")
    assert other.json()["confirmed"] is False
    assert other.json()["state"] == "NOT_CONFIRMED"


def test_reconstruction_gates_and_provenance(
    client: TestClient, services: AppServices, handlers
) -> None:
    video_id, _ = _stored_video(client, services)
    pending = client.post(
        f"/v1/videos/{video_id}/tracks",
        json=_seed(video_id, 10, (0, 0, 10, 10)),
        headers=key("trk-unconf"),
    )
    anyio.run(run_job, services, handlers, pending.json()["job_id"])
    blocked = client.post(
        "/v1/reconstructions",
        json={
            "track_id": pending.json()["track_id"],
            "target_frame_id": f"{video_id}:f417",
            "target_bbox_px": list(PLATE_BBOX),
        },
        headers=key("recon-blocked"),
    )
    assert blocked.status_code == 409

    seeded = client.post(
        f"/v1/videos/{video_id}/tracks",
        json=_seed(video_id),
        headers=key("trk-conf"),
    )
    anyio.run(run_job, services, handlers, seeded.json()["job_id"])
    track_id = seeded.json()["track_id"]

    refused = client.post(
        "/v1/reconstructions",
        json={
            "track_id": track_id,
            "target_frame_id": f"{video_id}:f400",
            "target_bbox_px": list(PLATE_BBOX),
        },
        headers=key("recon-no"),
    )
    assert refused.status_code == 202
    anyio.run(run_job, services, handlers, refused.json()["job_id"])
    refused_run = client.get(f"/v1/reconstructions/{refused.json()['run_id']}")
    assert refused_run.json()["state"] == "REFUSED"

    ok = client.post(
        "/v1/reconstructions",
        json={
            "track_id": track_id,
            "target_frame_id": f"{video_id}:f417",
            "target_bbox_px": list(PLATE_BBOX),
        },
        headers=key("recon-yes"),
    )
    assert ok.status_code == 202
    anyio.run(run_job, services, handlers, ok.json()["job_id"])
    run_id = ok.json()["run_id"]
    run = client.get(f"/v1/reconstructions/{run_id}")
    assert run.status_code == 200
    assert run.json()["state"] == "SUCCEEDED"
    assert run.json()["result_png_sha256"] == RESULT_SHA
    decisions = client.get(f"/v1/reconstructions/{run_id}/decisions")
    assert decisions.status_code == 200
    assert decisions.json()

    origin = client.get(f"/v1/reconstructions/{run_id}/provenance", params={"x": 219, "y": 481})
    assert origin.status_code == 200, origin.text
    assert origin.json()["provenance_class"] in {"ORIGINAL", "BORROWED", "GENERATED_BLEND"}
    assert origin.json()["run_id"] == run_id

    oob = client.get(f"/v1/reconstructions/{run_id}/provenance", params={"x": 5000, "y": 0})
    assert oob.status_code == 422

    provenance = services.repository.get_provenance(run_id)
    npz = services.settings.data_dir / provenance.class_map_uri
    npz.write_bytes(b"not-the-provenance-map")
    mismatch = client.get(f"/v1/reconstructions/{run_id}/provenance", params={"x": 219, "y": 481})
    assert mismatch.status_code == 409
    assert mismatch.json()["error"]["code"] == "ARTIFACT_HASH_CONFLICT"

    npz.unlink()
    missing = client.get(f"/v1/reconstructions/{run_id}/provenance", params={"x": 219, "y": 481})
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "NOT_FOUND"


def test_review_and_report_gates(client: TestClient, services: AppServices, handlers) -> None:
    video_id, _ = _stored_video(client, services)
    seeded = client.post(
        f"/v1/videos/{video_id}/tracks", json=_seed(video_id), headers=key("trk-rev")
    )
    anyio.run(run_job, services, handlers, seeded.json()["job_id"])
    recon = client.post(
        "/v1/reconstructions",
        json={
            "track_id": seeded.json()["track_id"],
            "target_frame_id": f"{video_id}:f417",
            "target_bbox_px": list(PLATE_BBOX),
        },
        headers=key("recon-rev"),
    )
    anyio.run(run_job, services, handlers, recon.json()["job_id"])
    run_id = recon.json()["run_id"]
    run = client.get(f"/v1/reconstructions/{run_id}").json()

    bad_hash = client.post(
        f"/v1/reconstructions/{run_id}/reviews",
        json={
            "reviewer_alias": "analyst-1",
            "decision": "APPROVE",
            "reviewed_result_sha256": "0" * 64,
            "reviewed_provenance_sha256": run["provenance_sha256"],
        },
        headers=key("rev-bad"),
    )
    assert bad_hash.status_code == 409
    assert bad_hash.json()["error"]["code"] == "ARTIFACT_HASH_CONFLICT"

    veto_incomplete = client.post(
        f"/v1/reconstructions/{run_id}/reviews",
        json={
            "reviewer_alias": "analyst-1",
            "decision": "VETO",
            "reviewed_result_sha256": run["result_png_sha256"],
            "reviewed_provenance_sha256": run["provenance_sha256"],
        },
        headers=key("rev-veto-bad"),
    )
    assert veto_incomplete.status_code == 422

    report_before = client.post(
        "/v1/reports",
        json={"run_id": run_id, "review_id": seeded.json()["track_id"]},
        headers=key("rep-early"),
    )
    assert report_before.status_code == 409
    assert report_before.json()["error"]["code"] == "REVIEW_REQUIRED"

    approve = client.post(
        f"/v1/reconstructions/{run_id}/reviews",
        json={
            "reviewer_alias": "analyst-1",
            "decision": "APPROVE",
            "comment": "Inspected donors and provenance overlay.",
            "reviewed_result_sha256": run["result_png_sha256"],
            "reviewed_provenance_sha256": run["provenance_sha256"],
        },
        headers=key("rev-ok"),
    )
    assert approve.status_code == 201, approve.text
    review_id = approve.json()["review_id"]
    assert approve.json()["decision"] == "APPROVE"

    accepted = client.post(
        "/v1/reports",
        json={"run_id": run_id, "review_id": review_id},
        headers=key("rep-ok"),
    )
    assert accepted.status_code == 202, accepted.text
    anyio.run(run_job, services, handlers, accepted.json()["job_id"])
    report = client.get(f"/v1/reports/{accepted.json()['report_id']}")
    assert report.status_code == 200
    assert report.json()["source_verified"] is True


def test_handlers_do_not_mutate_fixtures(handlers) -> None:
    path = CONTRACTS / "track_confirmed.json"
    before = path.read_bytes()
    assert seed_confirms_plate("01234567-89ab-7cde-8f01-23456789abcd:f417", PLATE_BBOX)
    after = path.read_bytes()
    assert before == after
    for name in (
        "track_confirmed.json",
        "track_not_confirmed.json",
        "reconstruction_run_succeeded.json",
        "pixel_provenance.json",
        "evidence_report.json",
    ):
        digest = hashlib.sha256((CONTRACTS / name).read_bytes()).hexdigest()
        assert len(digest) == 64


def test_reidentify_valid_hashes(services: AppServices, handlers) -> None:
    from probity.api.mocks import TrackMockHandler
    from probity.domain.enums import InferenceMode, JobKind, JobState, SubjectType
    from probity.domain.ids import new_uuid7
    from probity.domain.models import JobView
    from probity.ports import ClaimedJob, TrackPayload, TrackRequest
    from tests.api.conftest import TestJobContext

    track_handler = next(h for h in handlers.values() if isinstance(h, TrackMockHandler))
    video_id = new_uuid7()
    case_id = new_uuid7()
    track_id = new_uuid7()
    req = TrackRequest(
        track_id=track_id,
        case_id=case_id,
        video_id=video_id,
        source_sha256="9806895754d3af16f50ba66a99514f4b7c1b6659b278fdf7f8260cfd12bf05e5",
        subject_type=SubjectType.LICENSE_PLATE,
        seed_frame_id=f"{video_id}:f417",
        seed_bbox_px=PLATE_BBOX,
    )
    job = JobView.create(
        job_id=new_uuid7(),
        kind=JobKind.TRACK,
        case_id=case_id,
        subject_id=track_id,
        state=JobState.RUNNING,
        stage="TRACK",
        mode=InferenceMode.FIXTURE,
        correlation_id=new_uuid7(),
        updated_at=services.clock(),
        started_at=services.clock(),
    )
    claimed = ClaimedJob(
        job=job,
        payload=TrackPayload(request=req),
        lease_owner="t",
        lease_expires_at=services.clock(),
    )
    result = anyio.run(track_handler.run, claimed, TestJobContext())
    stored = services.repository.get_track(track_id)
    assert stored.content_sha256 == stored.compute_content_sha256()
    assert stored.track_id == track_id
    assert stored.video_id == video_id
    assert result.state.value in {"SUCCEEDED", "REFUSED"}
