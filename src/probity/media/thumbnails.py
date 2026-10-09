"""Lossless PNG frame extraction and non-canonical JPEG thumbnails via FFmpeg.

Canonical stills are RGB24 PNG. ``pixel_sha256`` hashes the raw RGB24 buffer, never the PNG
bytes. Thumbnails sample about eight frames in the window, pick the sharpest by Laplacian
variance of luma, and write a 480 px-wide JPEG. Outputs are refused under ``source/``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from probity.domain.enums import ErrorCode
from probity.domain.errors import ProbityError, SourcePathViolation, UnsupportedMedia
from probity.domain.models import FrameManifest
from probity.domain.policy import PolicyConfig
from probity.media.source_store import assert_not_source
from probity.ports import SegmentWindow

_SAMPLE_FRAMES = 8


def sample_frame_numbers(
    start_frame: int, end_frame: int, count: int = _SAMPLE_FRAMES
) -> tuple[int, ...]:
    """Evenly spaced decoder indices in ``[start_frame, end_frame]``, inclusive."""
    if end_frame < start_frame:
        raise ValueError("end_frame must be >= start_frame")
    if count <= 0:
        raise ValueError("count must be positive")
    span = end_frame - start_frame
    n = span + 1
    k = count if count < n else n
    if k == 1:
        return (start_frame,)
    return tuple(start_frame + (i * span) // (k - 1) for i in range(k))


def luma_plane(rgb: np.ndarray) -> np.ndarray:
    """8-bit Rec.601 luma as float64 from an RGB24 array."""
    r = rgb[..., 0].astype(np.int32, copy=False)
    g = rgb[..., 1].astype(np.int32, copy=False)
    b = rgb[..., 2].astype(np.int32, copy=False)
    return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.float64)


def laplacian_variance(luma: np.ndarray) -> float:
    """Variance of a 4-neighbour Laplacian; higher means a sharper still."""
    padded = np.pad(luma, 1, mode="edge")
    lap = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * luma
    return float(np.var(lap))


def rgb24_sha256(rgb: np.ndarray) -> str:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected HxWx3 uint8 RGB24")
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


class FfmpegThumbnailExtractor:
    """``ThumbnailExtractor`` over the ffmpeg CLI (list args, never a shell)."""

    def __init__(
        self,
        policy: PolicyConfig,
        ffmpeg_bin: str = "ffmpeg",
        *,
        extract_timeout_s: float = 60.0,
    ) -> None:
        self._policy = policy
        self._bin = ffmpeg_bin
        self._extract_timeout_s = extract_timeout_s
        self._version: str | None = None

    @property
    def extractor(self) -> str:
        if self._version is None:
            try:
                proc = subprocess.run(
                    [self._bin, "-version"],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=self._policy.video.probe_timeout_s,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ProbityError("ffmpeg could not be executed") from exc
            first = proc.stdout.decode("utf-8", "replace").splitlines()[:1]
            line = first[0].strip() if first else ""
            if not line:
                raise ProbityError("ffmpeg -version returned no output")
            self._version = line[:200]
        return self._version

    def extract_frame_png(
        self, source_path: str, manifest: FrameManifest, frame_number: int, out_path: str
    ) -> str:
        if frame_number < 0 or frame_number >= len(manifest.frames):
            raise UnsupportedMedia(
                "frame_number out of range",
                details={"frame_number": frame_number, "frame_count": len(manifest.frames)},
            )
        dest = _guarded_output(out_path, protected=(source_path,))
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        _guarded_output(tmp, protected=(source_path,))
        tmp.unlink(missing_ok=True)
        try:
            self._run(
                [
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    f"file:{source_path}",
                    "-map",
                    "0:v:0",
                    "-an",
                    "-vf",
                    f"select=eq(n\\,{frame_number}),format=rgb24",
                    "-frames:v",
                    "1",
                    "-fps_mode",
                    "passthrough",
                    "-update",
                    "1",
                    "-f",
                    "image2",
                    f"file:{tmp}",
                ]
            )
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        rgb = np.array(Image.open(dest).convert("RGB"), dtype=np.uint8)
        expected = manifest.frames[frame_number]
        if rgb.shape[1] != expected.width_px or rgb.shape[0] != expected.height_px:
            raise UnsupportedMedia(
                "decoded frame dimensions do not match the manifest",
                details={
                    "frame_number": frame_number,
                    "width_px": int(rgb.shape[1]),
                    "height_px": int(rgb.shape[0]),
                },
            )
        return rgb24_sha256(rgb)

    def extract_thumbnail(
        self, source_path: str, window: SegmentWindow, manifest: FrameManifest, out_path: str
    ) -> int:
        dest = _guarded_output(out_path, protected=(source_path,))
        samples = sample_frame_numbers(window.start_frame, window.end_frame)
        best_frame: int | None = None
        best_score = -1.0
        best_rgb: np.ndarray | None = None
        with tempfile.TemporaryDirectory(prefix="probity-thumbs-") as tmpdir:
            tmp_root = assert_not_source(tmpdir)
            for frame_number in samples:
                png = tmp_root / f"f{frame_number}.png"
                self.extract_frame_png(source_path, manifest, frame_number, str(png))
                rgb = np.array(Image.open(png).convert("RGB"), dtype=np.uint8)
                score = laplacian_variance(luma_plane(rgb))
                if score > best_score:
                    best_score = score
                    best_frame = frame_number
                    best_rgb = rgb
        if best_frame is None or best_rgb is None:
            raise UnsupportedMedia("thumbnail window contained no frames")
        width = self._policy.segment.thumbnail_width_px
        src_h, src_w = int(best_rgb.shape[0]), int(best_rgb.shape[1])
        height = max(1, (src_h * width) // src_w)
        image = Image.fromarray(best_rgb, mode="RGB").resize(
            (width, height), Image.Resampling.LANCZOS
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        _guarded_output(tmp, protected=(source_path,))
        tmp.unlink(missing_ok=True)
        try:
            image.save(tmp, format="JPEG", quality=85, optimize=True)
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        return best_frame

    def _run(self, args: list[str]) -> None:
        try:
            proc = subprocess.run(
                [self._bin, *args],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=self._extract_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise UnsupportedMedia(
                "frame extraction timed out",
                code=ErrorCode.MEDIA_PROBE_TIMEOUT,
                details={"timeout_s": self._extract_timeout_s},
            ) from exc
        except OSError as exc:
            raise ProbityError("ffmpeg could not be executed") from exc
        if proc.returncode != 0:
            raise UnsupportedMedia(
                "ffmpeg could not extract the requested frame",
                details={"exit_code": proc.returncode},
            )


def _guarded_output(path: str | Path, *, protected: tuple[str | Path, ...] = ()) -> Path:
    resolved = assert_not_source(path, protected=protected)
    if resolved.exists() and resolved.is_symlink():
        raise SourcePathViolation("refusing to write through a symlink")
    return resolved
