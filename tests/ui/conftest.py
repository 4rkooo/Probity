"""Shared fixtures for Streamlit AppTest UI tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from probity.api.mock_client import MockApiClient
from probity.ui.state import completed_progress, reset_client

HOME_PATH = Path(__file__).resolve().parents[2] / "src" / "probity" / "ui" / "Home.py"


@pytest.fixture
def client() -> MockApiClient:
    return reset_client()


@pytest.fixture
def make_app(client: MockApiClient) -> Callable[..., AppTest]:
    """Build an AppTest at ``step``; ``progress=True`` seeds a finished path up to Probity."""

    def _make(step: str, *, progress: bool = True, **state: Any) -> AppTest:
        at = AppTest.from_file(str(HOME_PATH), default_timeout=60)
        at.session_state["poll_interval_s"] = 0
        at.session_state["step"] = step
        if progress:
            for key, value in completed_progress().items():
                at.session_state[key] = value
        for key, value in state.items():
            at.session_state[key] = value
        return at

    return _make


def click(at: AppTest, label_fragment: str) -> AppTest:
    button = next(b for b in at.button if label_fragment in b.label)
    return button.click().run()


def texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown]
    for kind in (at.warning, at.error, at.success, at.info, at.caption):
        parts.extend(str(e.value) for e in kind)
    return "\n".join(parts)


def borrowed_pixel(client: MockApiClient, frame_number: int) -> tuple[int, int, int]:
    """First BORROWED output pixel sourced from donor ``frame_number``: (x, y, pts_us)."""
    lut = client.run_succeeded.provenance.source_lut
    entry = next(e for e in lut if e.frame_number == frame_number)
    row = next(r for r in client.provenance_exceptions() if r[2] == 1 and r[3] == entry.index)
    return int(row[0]), int(row[1]), entry.pts_us
