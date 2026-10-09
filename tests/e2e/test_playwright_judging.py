"""Playwright E2E: the five-minute judging path in a real headless Chromium.

Launches ``streamlit run src/probity/ui/Home.py`` against the mock client, then walks the
judging script, including hovering a borrowed pixel inside the provenance canvas and using
**Open source** to seek the original player to the donor frame. Skipped when Playwright or
its Chromium build is not installed.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT / "src" / "probity" / "ui" / "Home.py"
STEP_TIMEOUT_MS = 30_000


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def app_url() -> Iterator[str]:
    port = _free_port()
    env = {**os.environ, "PROBITY_UI_POLL_S": "0.2"}
    proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(HOME),
            "--server.headless",
            "true",
            "--server.port",
            str(port),
            "--server.address",
            "127.0.0.1",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{url}/_stcore/health", timeout=1) as r:  # noqa: S310
                    if r.status == 200:
                        break
            except OSError:
                time.sleep(0.5)
        else:
            pytest.fail("Streamlit did not become healthy within 60 s")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="module")
def browser() -> Iterator[object]:
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - browser binary missing
            pytest.skip(f"Chromium unavailable: {exc}")
        yield b
        b.close()


def test_judging_path_in_browser(app_url: str, browser) -> None:
    started = time.monotonic()
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.set_default_timeout(STEP_TIMEOUT_MS)
    expect = sync_api.expect

    def button(name: str):
        return page.get_by_role("button", name=name).first

    # 0:00-0:30 case setup
    page.goto(app_url)
    expect(page.get_by_text("Research/demo prototype - not for legal conclusions")).to_be_visible()
    expect(page.get_by_text("VERIFIED CACHE", exact=False).first).to_be_visible()
    button("Confirm Case & Proceed to Ingest ➔").click()

    # 0:30-1:00 ingestion with polled stage transitions
    button("▶ Start Ingestion").click()
    expect(button("Proceed to Search ➔")).to_be_visible()
    expect(page.get_by_text("VIDEO: SEARCHABLE")).to_be_visible()
    button("Proceed to Search ➔").click()

    # 1:00-1:40 search
    button("🔎 Search").click()
    expect(page.get_by_text("Rank #1", exact=False).first).to_be_visible()
    button("Select Result #1 for Tracking & Reconstruction ➔").click()

    # 1:40-2:20 subject
    expect(page.get_by_text("Tracking Integrity Gate")).to_be_visible()
    button("Run Probity Reconstruction ➔").click()

    # 2:20-4:10 Probity canvas: hover a borrowed pixel, open its source frame
    frame = page.frame_locator("iframe").first
    canvas = frame.locator("#cR")
    expect(canvas).to_be_visible()
    from probity.api.mock_client import MockApiClient
    from tests.ui.conftest import borrowed_pixel

    px, py, donor_pts = borrowed_pixel(MockApiClient(), 48)
    pos = frame.locator("body").evaluate(
        f"""() => {{ const v = viewport(); const r = cR.getBoundingClientRect();
                   return {{x: ({px} + 0.5 - v.x0) * r.width / v.w, y: ({py} + 0.5 - v.y0) * r.height / v.h}}; }}"""
    )
    canvas.click(position=pos)  # unpin the default pixel
    canvas.hover(position={"x": pos["x"] + 0.2, "y": pos["y"]})
    expect(frame.locator("#iClass")).to_have_text("BORROWED")
    expect(frame.locator("#iFrame")).to_contain_text("f48")
    expect(frame.locator("#iPts")).to_contain_text(f"{donor_pts / 1e6:.3f}s")
    canvas.click(position={"x": pos["x"] + 0.2, "y": pos["y"]})  # pin it
    frame.locator("#openSrc").click()
    expect(frame.locator("#leftTitle")).to_contain_text("SOURCE DONOR f48")
    # Both players moved to the donor frame (frame-exact index, PTS from the manifest).
    assert frame.locator("body").evaluate("() => D.framePts[cur]") == donor_pts
    expect(frame.locator("#ptsA")).to_contain_text("f48")
    expect(frame.get_by_text("RECOMPRESSED PREVIEW - NOT THE CANONICAL RESULT")).to_be_visible()

    page.get_by_role("tab", name="⚖️ Conventional Upscaler (Baseline)").click()
    expect(page.get_by_text("BASELINE IS NON-EVIDENTIARY", exact=False)).to_be_visible()
    button("Proceed to Integrity & Review ➔").click()

    # 4:10-4:40 approve exact hashes
    expect(page.get_by_text("Deterministic Integrity Score")).to_be_visible()
    page.get_by_text("I have inspected the provenance overlay", exact=False).click()
    approve = button("✅ Approve Enhancement")
    expect(approve).to_be_enabled()
    approve.click()

    # 4:40-5:00 export
    expect(page.get_by_text("Active Approval Confirmed", exact=False)).to_be_visible()
    expect(page.get_by_text("SOURCE HASH RE-VERIFIED NOW")).to_be_visible()
    button("📦 Export Evidence Bundle").click()
    expect(page.get_by_text("Bundle re-verified: every manifest hash matches")).to_be_visible()
    expect(page.get_by_role("button", name="📥 Download Bundle (.zip)")).to_be_visible()

    assert time.monotonic() - started < 300, "judging path must finish within five minutes"
    page.close()
