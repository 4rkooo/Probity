"""Exact integer/Fraction PTS math and a source-level fps*time guard."""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path

import pytest

from probity.domain.enums import ErrorCode
from probity.domain.errors import UnsupportedMedia
from probity.media.ffprobe import build_frame_entries, ticks_to_us

MEDIA_ROOT = Path(__file__).resolve().parents[2] / "src" / "probity" / "media"


def test_ticks_to_us_matches_demo_formula() -> None:
    tb = Fraction(1, 15360)
    for n in range(2700):
        pts = n * 512
        assert ticks_to_us(pts, tb) == n * 100_000 // 3
    assert ticks_to_us(1381888 + 512, tb) == 90_000_000


def test_vfr_pts_via_fraction_only() -> None:
    tb = Fraction(1, 1000)
    raw = (
        {"pts": 0, "key_frame": 1, "width": 160, "height": 120, "pkt_duration": 1000},
        {"pts": 2500, "key_frame": 0, "width": 160, "height": 120, "pkt_duration": 400},
        {"pts": 4100, "key_frame": 0, "width": 160, "height": 120, "pkt_duration": 900},
    )
    entries, end_ticks = build_frame_entries(raw, tb)
    assert [e.pts_us for e in entries] == [0, 2_500_000, 4_100_000]
    assert [e.pts for e in entries] == [0, 2500, 4100]
    assert end_ticks == 5000
    assert ticks_to_us(end_ticks, tb) == 5_000_000


def test_last_pts_delta_when_duration_missing() -> None:
    tb = Fraction(1, 90_000)
    raw = (
        {"pts": 0, "key_frame": 1, "width": 64, "height": 64},
        {"pts": 3000, "key_frame": 0, "width": 64, "height": 64},
        {"pts": 7500, "key_frame": 0, "width": 64, "height": 64},
    )
    entries, end_ticks = build_frame_entries(raw, tb)
    assert end_ticks == 7500 + (7500 - 3000)
    assert entries[-1].pts_us == ticks_to_us(7500, tb)


def test_non_monotonic_pts_rejected() -> None:
    tb = Fraction(1, 1000)
    raw = (
        {"pts": 100, "key_frame": 1, "width": 16, "height": 16, "duration": 100},
        {"pts": 50, "key_frame": 0, "width": 16, "height": 16, "duration": 100},
    )
    with pytest.raises(UnsupportedMedia) as exc:
        build_frame_entries(raw, tb)
    assert exc.value.code is ErrorCode.NON_MONOTONIC_TIMESTAMPS


def test_equal_microsecond_timestamps_rejected() -> None:
    tb = Fraction(1, 3_000_000)
    raw = (
        {"pts": 0, "key_frame": 1, "width": 16, "height": 16, "duration": 1},
        {"pts": 1, "key_frame": 0, "width": 16, "height": 16, "duration": 1},
        {"pts": 2, "key_frame": 0, "width": 16, "height": 16, "duration": 1},
    )
    with pytest.raises(UnsupportedMedia) as exc:
        build_frame_entries(raw, tb)
    assert exc.value.code is ErrorCode.NON_MONOTONIC_TIMESTAMPS


def test_media_package_has_no_float_fps_time_derivation() -> None:
    offenders: list[str] = []
    for path in sorted(MEDIA_ROOT.glob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            code = line.split("#", 1)[0]
            if "fps" not in code and "frame_rate" not in code:
                continue
            if re.search(r"\b(fps|frame_rate)\s*[\*/]|[\*/]\s*(fps|frame_rate)\b", code):
                offenders.append(f"{path.name}:{lineno}:{line.strip()}")
            if re.search(r"float\s*\(.*?(pts|duration|fps|time_base)", code):
                offenders.append(f"{path.name}:{lineno}:{line.strip()}")
    assert offenders == []
