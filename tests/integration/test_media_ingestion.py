"""Upload → hash → probe → manifest → segments → thumbnail/PNG, source tree untouched."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from PIL import Image

from probity.domain.errors import SourcePathViolation
from probity.domain.models import FrameManifest
from probity.domain.policy import default_policy
from probity.media.ffprobe import FfprobeMediaProber
from probity.media.hashing import sha256_file
from probity.media.segmentation import DeterministicSegmenter
from probity.media.source_store import LocalSourceStore
from probity.media.thumbnails import FfmpegThumbnailExtractor

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "fixtures/demo/source/demo-plate-90s.mp4"
MANIFEST_FIX = ROOT / "fixtures/contracts/frame_manifest.json"
SEGMENTS_FIX = ROOT / "fixtures/contracts/video_segments.json"
VIDEO_ID = "01a12164-8fe8-7cab-9c96-17404faf2b24"
SOURCE_SHA = "9806895754d3af16f50ba66a99514f4b7c1b6659b278fdf7f8260cfd12bf05e5"


def _which(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    brew = Path("/opt/homebrew/bin") / name
    return str(brew) if brew.is_file() else None


FFMPEG = _which("ffmpeg")
FFPROBE = _which("ffprobe")
pytestmark = [
    pytest.mark.ffmpeg,
    pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="ffmpeg/ffprobe not found"),
]

if Path("/opt/homebrew/bin").is_dir():
    os.environ["PATH"] = "/opt/homebrew/bin" + os.pathsep + os.environ.get("PATH", "")


def test_demo_clip_ingestion_pipeline(tmp_path: Path) -> None:
    policy = default_policy()
    store = LocalSourceStore(tmp_path / "data")
    with DEMO.open("rb") as handle:
        staged = store.stage_upload(handle, max_bytes=policy.video.max_bytes)
    stored = store.commit(staged)
    assert stored.sha256 == SOURCE_SHA
    assert stored.deduplicated is False
    source_path = store.resolve_read_path(stored.storage_uri)

    prober = FfprobeMediaProber(policy, ffprobe_bin=FFPROBE or "ffprobe")
    probe = prober.probe(source_path)
    assert probe.container == "mp4"
    assert probe.video_codec == "h264"
    assert probe.width_px == 1280
    assert probe.height_px == 720
    assert probe.time_base == "1/15360"
    assert probe.duration_us == 90_000_000
    assert probe.frame_count == 2700
    assert probe.has_audio is False
    assert probe.nominal_fps == "30/1"

    manifest = prober.frame_manifest(source_path, VIDEO_ID, stored.sha256)
    golden = FrameManifest.model_validate_json(MANIFEST_FIX.read_text())
    assert len(manifest.frames) == 2700
    assert manifest.time_base == golden.time_base == "1/15360"
    assert manifest.source_sha256 == golden.source_sha256 == SOURCE_SHA
    for index, (got, expected) in enumerate(zip(manifest.frames, golden.frames, strict=True)):
        assert got.frame_number == expected.frame_number == index
        assert got.pts == expected.pts == index * 512
        assert got.pts_us == expected.pts_us == index * 100_000 // 3
        assert got.width_px == expected.width_px
        assert got.height_px == expected.height_px
        assert got.is_keyframe == expected.is_keyframe
    assert manifest.frames[417].pts_us == 13_900_000

    windows = DeterministicSegmenter(policy).plan(manifest, probe.duration_us)
    expected_windows = json.loads(SEGMENTS_FIX.read_text())
    assert len(windows) == 15
    for window, raw in zip(windows, expected_windows, strict=True):
        assert window.ordinal == raw["ordinal"]
        assert window.start_pts_us == raw["start_pts_us"]
        assert window.end_pts_us == raw["end_pts_us"]
        assert window.start_frame == raw["start_frame"]
        assert window.end_frame == raw["end_frame"]

    extractor = FfmpegThumbnailExtractor(policy, ffmpeg_bin=FFMPEG or "ffmpeg")
    thumb_path = store.derived_path(VIDEO_ID, "thumbs", "s0000", "thumb.jpg")
    png_path = store.derived_path(VIDEO_ID, "frames", "f417", "frame.png")
    chosen = extractor.extract_thumbnail(source_path, windows[0], manifest, thumb_path)
    assert windows[0].start_frame <= chosen <= windows[0].end_frame
    jpeg = Image.open(thumb_path)
    assert jpeg.format == "JPEG"
    assert jpeg.size[0] == policy.segment.thumbnail_width_px

    pixel_sha = extractor.extract_frame_png(source_path, manifest, 417, png_path)
    assert len(pixel_sha) == 64
    png_file_sha, _ = sha256_file(png_path)
    assert pixel_sha != png_file_sha
    still = Image.open(png_path)
    assert still.size == (1280, 720)
    assert still.mode in {"RGB", "RGBA"}

    source_files = [p for p in store.source_root.rglob("*") if p.is_file()]
    assert [p.name for p in source_files] == ["original.mp4"]
    assert source_files[0].stat().st_mode & 0o777 == 0o444

    banned = Path(source_path).parent / "thumb.jpg"
    with pytest.raises(SourcePathViolation):
        extractor.extract_thumbnail(source_path, windows[0], manifest, str(banned))
    with pytest.raises(SourcePathViolation):
        extractor.extract_frame_png(source_path, manifest, 417, str(Path(source_path)))
