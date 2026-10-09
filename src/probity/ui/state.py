"""Shared UI state: the API client singleton, workflow steps, and step prerequisites.

The client lives in this importable module (not in ``Home.py``) so that the Streamlit
script, AppTest runs, and tests all share one instance.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from typing import Any

from probity.api.mock_client import MockApiClient

STEPS: tuple[str, ...] = ("Source", "Ingest", "Search", "Subject", "Probity", "Review", "Report")

# Seconds between job polls. The design calls for 1 s; the demo uses a shorter cadence
# and tests set PROBITY_UI_POLL_S=0 so that state transitions run instantly.
DEFAULT_POLL_S = float(os.environ.get("PROBITY_UI_POLL_S", "0.5"))

_client: MockApiClient | None = None


def get_client() -> MockApiClient:
    global _client
    if _client is None:
        _client = MockApiClient()
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
