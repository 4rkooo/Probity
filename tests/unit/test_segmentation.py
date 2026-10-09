"""Deterministic 8 s / 6 s segment windows, including short and VFR edges."""

from __future__ import annotations

import json
from pathlib import Path

from probity.domain.ids import new_uuid7
from probity.domain.models import FrameManifest, FrameManifestEntry
from probity.domain.policy import default_policy
from probity.media.segmentation import DeterministicSegmenter

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_FIX = ROOT / "fixtures/contracts/frame_manifest.json"
SEGMENTS_FIX = ROOT / "fixtures/contracts/video_segments.json"


def _manifest(pts_us: list[int]) -> FrameManifest:
    frames = tuple(
        FrameManifestEntry(
            frame_number=i,
            pts_us=t,
            pts=t,
            is_keyframe=i == 0,
            width_px=160,
            height_px=120,
        )
        for i, t in enumerate(pts_us)
    )
    return FrameManifest.create(
        video_id=new_uuid7(),
        source_sha256="ab" * 32,
        time_base="1/1000000",
        extractor="test",
        frames=frames,
    )


def _plan(duration_us: int, pts_us: list[int] | None = None):
    if pts_us is None:
        pts_us = []
        n = 0
        while True:
            t = n * 100_000 // 3
            if t >= duration_us:
                break
            pts_us.append(t)
            n += 1
    return DeterministicSegmenter(default_policy()).plan(_manifest(pts_us), duration_us)


def test_demo_manifest_has_fifteen_windows() -> None:
    manifest = FrameManifest.model_validate_json(MANIFEST_FIX.read_text())
    expected = json.loads(SEGMENTS_FIX.read_text())
    windows = DeterministicSegmenter(default_policy()).plan(manifest, 90_000_000)
    assert len(windows) == 15
    assert len(expected) == 15
    for window, raw in zip(windows, expected, strict=True):
        assert window.ordinal == raw["ordinal"]
        assert window.start_pts_us == raw["start_pts_us"]
        assert window.end_pts_us == raw["end_pts_us"]
        assert window.start_frame == raw["start_frame"]
        assert window.end_frame == raw["end_frame"]


def test_shorter_than_window() -> None:
    windows = _plan(3_000_000)
    assert len(windows) == 1
    assert windows[0].start_pts_us == 0
    assert windows[0].end_pts_us == 3_000_000
    assert windows[0].start_frame == 0
    assert windows[0].end_frame == 89


def test_exactly_eight_seconds() -> None:
    windows = _plan(8_000_000)
    assert len(windows) == 1
    assert windows[0].start_pts_us == 0
    assert windows[0].end_pts_us == 8_000_000
    assert windows[0].start_frame == 0
    assert windows[0].end_frame == 239


def test_fourteen_seconds() -> None:
    windows = _plan(14_000_000)
    assert [(w.start_pts_us, w.end_pts_us, w.start_frame, w.end_frame) for w in windows] == [
        (0, 8_000_000, 0, 239),
        (6_000_000, 14_000_000, 180, 419),
    ]


def test_eighty_six_seconds() -> None:
    windows = _plan(86_000_000)
    assert windows[-1].start_pts_us == 78_000_000
    assert windows[-1].end_pts_us == 86_000_000
    assert windows[-1].start_frame == 2340
    assert windows[-1].end_frame == 2579
    assert all(w.end_pts_us - w.start_pts_us == 8_000_000 for w in windows)


def test_eighty_six_point_five_seconds() -> None:
    windows = _plan(86_500_000)
    assert windows[-2].start_pts_us == 78_000_000
    assert windows[-2].end_pts_us == 86_000_000
    assert windows[-1].start_pts_us == 84_000_000
    assert windows[-1].end_pts_us == 86_500_000
    assert windows[-1].start_frame == 2520
    assert windows[-1].end_frame == 2594
    assert windows[-1].end_pts_us - windows[-1].start_pts_us == 2_500_000


def test_ninety_seconds_matches_step_grid() -> None:
    windows = _plan(90_000_000)
    assert len(windows) == 15
    assert [w.start_pts_us for w in windows] == [i * 6_000_000 for i in range(15)]
    assert windows[-1].end_pts_us == 90_000_000
    assert windows[-1].start_frame == 2520
    assert windows[-1].end_frame == 2699


def test_vfr_uses_pts_not_frame_count() -> None:
    pts = [0, 100_000, 500_000, 1_000_000, 4_000_000, 9_500_000, 11_000_000]
    windows = _plan(12_000_000, pts_us=pts)
    assert windows[0].start_pts_us == 0
    assert windows[0].end_pts_us == 8_000_000
    assert windows[0].start_frame == 0
    assert windows[0].end_frame == 4  # 4_000_000 < 8s, 9_500_000 is not
    assert windows[-1].end_frame == 6
    assert windows[-1].start_frame == 5  # first pts >= 6s is 9_500_000
