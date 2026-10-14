"""Shared UI state: the API client singleton, workflow steps, and step prerequisites.

The client lives in this importable module (not in ``Home.py``) so that the Streamlit
script, AppTest runs, and tests all share one instance.
"""

from __future__ import annotations

import json
import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from probity.api.live_client import LiveApiClient
from probity.api.mock_client import MockApiClient

STEPS: tuple[str, ...] = ("Source", "Ingest", "Search", "Subject", "Probity", "Review", "Report")

# Seconds between job polls. The design calls for 1 s; the demo uses a shorter cadence
# and tests set PROBITY_UI_POLL_S=0 so that state transitions run instantly.
DEFAULT_POLL_S = float(os.environ.get("PROBITY_UI_POLL_S", "0.5"))

# Survives Streamlit process restarts so Search keeps hitting the live custom clip.
_LIVE_SESSION_PATH = Path(
    os.environ.get("PROBITY_LAST_LIVE_SESSION", "data/last_live_upload.json")
)

_client: MockApiClient | None = None


def persist_live_session(*, video_id: str, job_id: str, case_id: str) -> None:
    """Write the active live upload ids so a UI restart can re-bind Search."""
    path = _LIVE_SESSION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"video_id": video_id, "job_id": job_id, "case_id": case_id},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_live_session() -> dict[str, str] | None:
    path = _LIVE_SESSION_PATH
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    video_id = raw.get("video_id")
    job_id = raw.get("job_id")
    if not video_id or not job_id:
        return None
    return {
        "video_id": str(video_id),
        "job_id": str(job_id),
        "case_id": str(raw.get("case_id") or ""),
    }


def _rehydrate_live(client: MockApiClient, ss: MutableMapping[str, Any] | None) -> None:
    """Restore live upload binding from session state or the last-upload file."""
    if client.uses_live_upload:
        return
    video_id = ss.get("live_video_id") if ss is not None else None
    job_id = ss.get("live_job_id") if ss is not None else None
    if (not video_id or not job_id) and ss is not None:
        saved = load_live_session()
        if saved is not None:
            video_id = video_id or saved["video_id"]
            job_id = job_id or saved["job_id"]
            ss.setdefault("live_video_id", video_id)
            ss.setdefault("live_job_id", job_id)
            if saved.get("case_id"):
                ss.setdefault("live_case_id", saved["case_id"])
            # Custom clip was already searchable before the UI restart.
            if ss.get("ingest_state") is None:
                ss["ingest_state"] = "SUCCEEDED"
    if not video_id or not job_id:
        saved = load_live_session()
        if saved is None:
            return
        video_id, job_id = saved["video_id"], saved["job_id"]
    live = LiveApiClient()
    if not live.health_ok():
        return
    try:
        video = live.get_video(str(video_id))
        case = live.get_case(video.case_id)
        client.adopt_live_upload(
            live_api=live,
            case=case,
            video=video,
            job_id=str(job_id),
            local_path=None,
        )
    except Exception:  # noqa: BLE001 - keep demo client if rehydrate fails
        return


def get_client(ss: MutableMapping[str, Any] | None = None) -> MockApiClient:
    global _client
    if _client is None:
        _client = MockApiClient()
    _rehydrate_live(_client, ss)
    return _client


def reset_client() -> MockApiClient:
    """Restore demo state on the shared client (used by tests and the Reset control)."""
    client = get_client()
    client.reset()
    return client


def poll_interval(ss: MutableMapping[str, Any]) -> float:
    return float(ss.get("poll_interval_s", DEFAULT_POLL_S))


def step_prerequisite(step: str, ss: MutableMapping[str, Any]) -> str | None:
    """Plain-language reason a step is not yet usable, or None when it is."""
    ingest_state = ss.get("ingest_state")
    if step == "Search" and ingest_state not in ("SUCCEEDED", "PARTIAL"):
        return "Requires a searchable video: finish ingestion first."
    if step == "Subject" and not ss.get("selected_segment"):
        return "Requires a selected search result."
    if step == "Probity" and ss.get("target_frame") is None:
        return "Requires a confirmed subject and chosen target frame."
    if step == "Review" and ss.get("run_outcome") != "SUCCEEDED":
        if ss.get("run_outcome") in ("REFUSED", "FAILED"):
            return "The last reconstruction produced no reviewable artifact."
        return "Requires a completed Probity reconstruction."
    if step == "Report" and ss.get("run_outcome") != "SUCCEEDED":
        return "Requires an approved reconstruction."
    return None


def completed_progress(segment_id: str | None = None) -> dict[str, Any]:
    """Session values representing a finished happy path up to Probity (for tests)."""
    client = get_client()
    return {
        "case_confirmed": True,
        "ingest_state": "SUCCEEDED",
        "selected_segment": segment_id or client.search_evidence.results[0].segment_id,
        "target_frame": client.target_frame_number,
        "run_outcome": "SUCCEEDED",
        "recon_done": True,
    }


def go(ss: MutableMapping[str, Any], step: str) -> None:
    ss["step"] = step
