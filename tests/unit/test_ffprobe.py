"""ffprobe container/codec gates and exact-PTS probing."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from probity.domain.enums import ErrorCode
from probity.domain.errors import UnsupportedMedia
from probity.domain.ids import new_uuid7
from probity.domain.policy import default_policy
from probity.media.ffprobe import FfprobeMediaProber, ticks_to_us
from probity.media.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "fixtures/demo/source/demo-plate-90s.mp4"


def _which(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    brew = Path("/opt/homebrew/bin") / name
    return str(brew) if brew.is_file() else None


FFMPEG = _which("ffmpeg")
FFPROBE = _which("ffprobe")
skip_ffmpeg = pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="ffmpeg/ffprobe not found")

if FFMPEG and Path("/opt/homebrew/bin").is_dir():
    os.environ["PATH"] = "/opt/homebrew/bin" + os.pathsep + os.environ.get("PATH", "")


def _policy(probe_timeout_s: float | None = None):
    policy = default_policy()
    if probe_timeout_s is None:
        return policy
    return policy.model_copy(
        update={"video": policy.video.model_copy(update={"probe_timeout_s": probe_timeout_s})}
    )


def _prober() -> FfprobeMediaProber:
    return FfprobeMediaProber(_policy(), ffprobe_bin=FFPROBE or "ffprobe")


def encode(
    path: Path,
    *,
    duration: str = "0.4",
    size: str = "160x120",
    fps: str = "10",
    codec: str = "libx264",
    extra: list[str] | None = None,
    lavfi: str | None = None,
) -> Path:
    assert FFMPEG is not None
    cmd = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        lavfi or f"color=c=blue:s={size}:r={fps}:d={duration}",
        "-an",
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
    ]
    if extra:
        cmd.extend(extra)
    cmd.append(str(path))
    subprocess.run(cmd, check=True)
    return path


@skip_ffmpeg
@pytest.mark.ffmpeg
def test_rejects_non_mp4(tmp_path: Path) -> None:
    path = tmp_path / "clip.mkv"
    encode(path, extra=["-f", "matroska"])
    with pytest.raises(UnsupportedMedia, match="ISO|MP4|container"):
        _prober().probe(str(path))


@skip_ffmpeg
@pytest.mark.ffmpeg
def test_rejects_mpeg4_in_mp4(tmp_path: Path) -> None:
    path = tmp_path / "mpeg4.mp4"
    encode(path, codec="mpeg4")
    with pytest.raises(UnsupportedMedia, match="codec"):
        _prober().probe(str(path))


@skip_ffmpeg
@pytest.mark.ffmpeg
def test_rejects_oversize_resolution(tmp_path: Path) -> None:
    path = tmp_path / "big.mp4"
    encode(path, size="1922x1080", duration="0.2", fps="5")
    with pytest.raises(UnsupportedMedia, match="resolution"):
        _prober().probe(str(path))


@skip_ffmpeg
@pytest.mark.ffmpeg
def test_rejects_garbage_bytes(tmp_path: Path) -> None:
    path = tmp_path / "garbage.mp4"
    path.write_bytes(b"this is not an mp4 file" * 32)
    with pytest.raises(UnsupportedMedia):
        _prober().probe(str(path))


@skip_ffmpeg
@pytest.mark.ffmpeg
def test_rejects_audio_only(tmp_path: Path) -> None:
    assert FFMPEG is not None
    path = tmp_path / "audio.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.5",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )
    with pytest.raises(UnsupportedMedia, match="video stream"):
        _prober().probe(str(path))


@skip_ffmpeg
@pytest.mark.ffmpeg
def test_vfr_clip_uses_fraction_pts(tmp_path: Path) -> None:
    assert FFMPEG is not None
    path = tmp_path / "vfr.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x120:r=10:d=0.4",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=160x120:r=25:d=0.4",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0",
            "-fps_mode",
            "passthrough",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    prober = _prober()
    probe = prober.probe(str(path))
    manifest = prober.frame_manifest(str(path), new_uuid7(), sha256_file(path)[0])
    from fractions import Fraction

    tb = Fraction(probe.time_base)
    assert probe.frame_count == len(manifest.frames)
    prev = -1
    for entry in manifest.frames:
        assert entry.pts_us == ticks_to_us(entry.pts, tb)
        assert entry.pts_us > prev
        prev = entry.pts_us
    assert probe.duration_us > manifest.frames[-1].pts_us


@skip_ffmpeg
@pytest.mark.ffmpeg
def test_tiny_h264_mp4_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "ok.mp4"
    encode(path)
    probe = _prober().probe(str(path))
    assert probe.container == "mp4"
    assert probe.video_codec == "h264"
    assert probe.width_px == 160
    assert probe.height_px == 120
    assert probe.duration_us > 0
    assert probe.has_audio is False


def test_probe_timeout_uses_policy_budget(tmp_path: Path) -> None:
    fake = tmp_path / "ffprobe"
    fake.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n")
    fake.chmod(0o755)
    prober = FfprobeMediaProber(_policy(probe_timeout_s=0.2), ffprobe_bin=str(fake))
    with pytest.raises(UnsupportedMedia) as exc:
        prober.probe(str(DEMO))
    assert exc.value.code is ErrorCode.MEDIA_PROBE_TIMEOUT
