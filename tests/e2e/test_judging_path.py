"""Five-minute judging path driven through the UI with clicks only (AppTest, no browser).

Mirrors the design's judging script: case -> ingest -> search -> subject -> Probity ->
review -> export, ending with a re-verified evidence bundle. The browser version lives
in ``test_playwright_judging.py``.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from probity.reports.export import verify_bundle
from probity.ui.state import reset_client
from tests.ui.conftest import HOME_PATH, click, texts


def test_five_minute_judging_path() -> None:
    client = reset_client()
    at = AppTest.from_file(str(HOME_PATH), default_timeout=60)
    at.session_state["poll_interval_s"] = 0
    at.run()
    assert not at.exception

    # 0:00-0:30 case and source hash
    page = texts(at)
    assert "Fictional Case 0420 (demo)" in page
    assert "VERIFIED CACHE" in page
    assert "Re-hash matches manifest" in page
    click(at, "Confirm Case & Proceed to Ingest")
    assert at.session_state["step"] == "Ingest"

    # 0:30-1:00 ingestion with live job-state transitions
    click(at, "Start Ingestion")
    assert at.session_state["ingest_state"] == "SUCCEEDED"
    assert "NVIDIA Cosmos" in texts(at)
    click(at, "Proceed to Search")

    # 1:00-1:40 prepared query, W&B plan, cited results
    click(at, "🔎 Search")
    page = texts(at)
    assert "Rank #1" in page and "Weights & Biases" in "\n".join(e.label for e in at.expander)
    click(at, "Select Result #1")
    assert at.session_state["step"] == "Subject"

    # 1:40-2:20 confirmed track and target frame
    page = texts(at)
    assert "f417" in page and "CONFIRMED" in page
    click(at, "Run Probity Reconstruction")

    # 2:20-4:10 Probity, provenance, baseline
    assert at.session_state["run_outcome"] == "SUCCEEDED"
    assert "BASELINE IS NON-EVIDENTIARY" in texts(at)
    at.number_input(key="inspect_x").set_value(230).run()
    assert "f409" in texts(at) and "13.633s" in texts(at)
    click(at, "Proceed to Integrity & Review")

    # 4:10-4:40 approve exact hashes
    assert "Integrity" in texts(at)
    next(c for c in at.checkbox if "I have inspected" in c.label).check().run()
    click(at, "Approve")
    assert at.session_state["step"] == "Report"
    assert any("Active Approval Confirmed" in s.value for s in at.success)
    assert "SOURCE HASH RE-VERIFIED NOW" in texts(at)

    # 4:40-5:00 export and verify
    click(at, "Export Evidence Bundle")
    assert not at.exception
    assert "Bundle re-verified" in texts(at)
    export_dir = Path(at.session_state["export_dir"])
    assert verify_bundle(export_dir) == []
    names = [s["span_name"] for s in client.trace.spans]
    assert {"search.plan", "wandb.explain", "review.record"} <= set(names)
