"""ffprobe-backed media validation and exact-PTS frame manifests.

All timestamp math is exact integer/``Fraction`` arithmetic in stream ticks:
``pts_us = (pts * tb_num * 1_000_000) // tb_den``. Frame identity is the decoder-output index;
it is never derived from a nominal frame rate.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

from probity.domain.enums import ErrorCode
from probity.domain.errors import ProbityError, UnsupportedMedia
from probity.domain.ids import canonical_sha256
from probity.domain.models import FrameManifest, FrameManifestEntry
from probity.domain.policy import PolicyConfig
from probity.ports import ProbeResult

# Brands that identify an ISO base media (MP4) file. QuickTime ("qt  ") is a different container
# even though FFmpeg's mov demuxer reads both.
_MP4_BRANDS = frozenset(
    {b"isom", b"iso2", b"iso3", b"iso4", b"iso5", b"iso6", b"iso7", b"iso8", b"iso9"}
    | {b"mp41", b"mp42", b"avc1", b"M4V ", b"M4VH", b"M4VP", b"mmp4", b"dash", b"msnv"}
)
_REJECTED_MAJOR_BRANDS = frozenset({b"qt  "})
_FRAME_ENTRIES = "frame=pts,best_effort_timestamp,key_frame,width,height,duration,pkt_duration"


def parse_rational(value: object, *, field: str) -> Fraction:
    """Parse an ffprobe ``num/den`` string into a positive ``Fraction``."""
    if not isinstance(value, str) or value.count("/") != 1:
        raise UnsupportedMedia(f"ffprobe reported an invalid {field}", details={field: value})
    num_s, den_s = value.split("/")
    if not (num_s.isdigit() and den_s.isdigit()) or int(num_s) == 0 or int(den_s) == 0:
        raise UnsupportedMedia(f"ffprobe reported an invalid {field}", details={field: value})
    return Fraction(int(num_s), int(den_s))


def format_rational(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def ticks_to_us(ticks: int, time_base: Fraction) -> int:
    return (ticks * time_base.numerator * 1_000_000) // time_base.denominator


def _int_field(raw: Mapping[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return None


def build_frame_entries(
    raw_frames: Sequence[Mapping[str, Any]], time_base: Fraction
) -> tuple[tuple[FrameManifestEntry, ...], int]:
    """Convert ffprobe frames (decoder output order) to manifest entries plus ``end_ticks``.

    ``end_ticks`` is the last frame's PTS plus its duration (frame/packet duration if reported,
    else the last PTS delta).
    """
    if not raw_frames:
        raise UnsupportedMedia("no decodable video frames")
    entries: list[FrameManifestEntry] = []
    prev_pts: int | None = None
    prev_us: int | None = None
    for index, raw in enumerate(raw_frames):
        pts = _int_field(raw, "pts")
        if pts is None:
            pts = _int_field(raw, "best_effort_timestamp")
        if pts is None:
            raise UnsupportedMedia(
                "frame is missing a presentation timestamp",
                code=ErrorCode.NON_MONOTONIC_TIMESTAMPS,
                details={"frame_number": index},
            )
        if pts < 0:
            raise UnsupportedMedia(
                "negative presentation timestamps are not supported",
                code=ErrorCode.NON_MONOTONIC_TIMESTAMPS,
                details={"frame_number": index, "pts": pts},
            )
        pts_us = ticks_to_us(pts, time_base)
        if prev_pts is not None and prev_us is not None and (pts <= prev_pts or pts_us <= prev_us):
            raise UnsupportedMedia(
                "presentation timestamps are not strictly increasing at microsecond precision",
                code=ErrorCode.NON_MONOTONIC_TIMESTAMPS,
                details={"frame_number": index, "pts": pts, "previous_pts": prev_pts},
            )
        width = _int_field(raw, "width")
        height = _int_field(raw, "height")
        if not width or not height or width <= 0 or height <= 0:
            raise UnsupportedMedia("decoded frame has no dimensions", details={"frame": index})
        entries.append(
            FrameManifestEntry(
                frame_number=index,
                pts_us=pts_us,
                pts=pts,
                is_keyframe=_int_field(raw, "key_frame") == 1,
                width_px=width,
                height_px=height,
            )
        )
        prev_pts, prev_us = pts, pts_us

    last = raw_frames[-1]
    duration = _int_field(last, "pkt_duration")
    if duration is None or duration <= 0:
        duration = _int_field(last, "duration")
    if duration is None or duration <= 0:
        if len(entries) < 2:
            raise UnsupportedMedia("cannot determine the duration of a single-frame video")
        duration = entries[-1].pts - entries[-2].pts
    return tuple(entries), entries[-1].pts + duration


def check_iso_bmff(path: Path) -> None:
    """Require a leading ``ftyp`` box whose brands identify an MP4 (not QuickTime) file."""
    with path.open("rb") as handle:
        header = handle.read(4096)
    if len(header) < 16 or header[4:8] != b"ftyp":
        raise UnsupportedMedia("not an ISO base media (MP4) file", details={"container": "other"})
    size = int.from_bytes(header[0:4], "big")
    if size < 16 or size > len(header):
        raise UnsupportedMedia("malformed MP4 ftyp box")
    major = header[8:12]
    compatible = {header[i : i + 4] for i in range(16, size - 3, 4)}
    if major in _REJECTED_MAJOR_BRANDS or not ({major} | compatible) & _MP4_BRANDS:
        raise UnsupportedMedia(
            "file brand is not MP4",
            details={"major_brand": major.decode("latin-1")},
        )


@dataclass(frozen=True)
class _Analysis:
    probe: ProbeResult
    time_base: str
    frames: tuple[FrameManifestEntry, ...]


class FfprobeMediaProber:
    """``MediaProber`` over the ffprobe CLI (list args, never a shell).

    ``policy.video.probe_timeout_s`` bounds the container/stream probe. Enumerating every decoded
    frame scales with video length, so it has its own bound ``frames_timeout_s``.
    """

    def __init__(
        self,
        policy: PolicyConfig,
        ffprobe_bin: str = "ffprobe",
        *,
        frames_timeout_s: float = 180.0,
    ) -> None:
        self._policy = policy
        self._bin = ffprobe_bin
        self._frames_timeout_s = frames_timeout_s
        self._extractor: str | None = None
        self._cache: tuple[tuple[object, ...], _Analysis] | None = None

    @property
    def extractor(self) -> str:
        if self._extractor is None:
            out = self._run(["-version"], self._policy.video.probe_timeout_s)
            first = out.decode("utf-8", "replace").splitlines()[0].strip() if out else ""
            if not first:
                raise ProbityError("ffprobe -version returned no output")
            self._extractor = first[:200]
        return self._extractor

    def probe(self, path: str) -> ProbeResult:
        return self._analyze(path).probe

    def frame_manifest(self, path: str, video_id: str, source_sha256: str) -> FrameManifest:
        analysis = self._analyze(path)
        return FrameManifest.create(
            video_id=video_id,
            source_sha256=source_sha256,
            time_base=analysis.time_base,
            extractor=analysis.probe.extractor,
            frames=analysis.frames,
        )

    # ------------------------------------------------------------------------------ internals

    def _analyze(self, path: str) -> _Analysis:
        file_path = Path(path).resolve()
        if not file_path.is_file():
            raise UnsupportedMedia("media file not found")
        stat = file_path.stat()
        key = (str(file_path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        if stat.st_size == 0:
            raise UnsupportedMedia("empty media file")
        check_iso_bmff(file_path)
        analysis = self._analyze_uncached(file_path)
        self._cache = (key, analysis)
        return analysis

    def _analyze_uncached(self, path: Path) -> _Analysis:
        video_policy = self._policy.video
        meta = self._json(
            ["-show_format", "-show_streams", "-i", f"file:{path}"],
            video_policy.probe_timeout_s,
        )
        fmt = meta.get("format")
        streams = meta.get("streams")
        if not isinstance(fmt, dict) or not isinstance(streams, list):
            raise UnsupportedMedia("ffprobe returned no container information")
        format_names = str(fmt.get("format_name", "")).split(",")
        if "mp4" not in format_names:
            raise UnsupportedMedia(
                "container is not MP4", details={"format_name": fmt.get("format_name")}
            )
        videos = [s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"]
        if len(videos) != 1:
            raise UnsupportedMedia(
                "exactly one video stream is required", details={"video_streams": len(videos)}
            )
        video = videos[0]
        codec_name = video.get("codec_name")
        if codec_name not in video_policy.allowed_codecs:
            raise UnsupportedMedia(
                "unsupported video codec",
                details={"video_codec": codec_name, "allowed": list(video_policy.allowed_codecs)},
            )
        codec = cast(Literal["h264", "hevc"], codec_name)
        width, height = _int_field(video, "width"), _int_field(video, "height")
        if not width or not height or width <= 0 or height <= 0:
            raise UnsupportedMedia("video stream has no dimensions")
        if width > video_policy.max_width or height > video_policy.max_height:
            raise UnsupportedMedia(
                "video resolution exceeds the maximum",
                details={
                    "width_px": width,
                    "height_px": height,
                    "max_width": video_policy.max_width,
                    "max_height": video_policy.max_height,
                },
            )
        max_duration_us = video_policy.max_duration_s * 1_000_000
        declared = fmt.get("duration")
        if isinstance(declared, str):
            try:
                declared_us = Fraction(declared) * 1_000_000
            except (ValueError, ZeroDivisionError):
                declared_us = None
            if declared_us is not None and declared_us > max_duration_us:
                raise UnsupportedMedia(
                    "video duration exceeds the maximum",
                    details={"max_duration_s": video_policy.max_duration_s},
                )
        time_base = parse_rational(video.get("time_base"), field="time_base")

        frames_json = self._json(
            [
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_entries",
                _FRAME_ENTRIES,
                "-i",
                f"file:{path}",
            ],
            self._frames_timeout_s,
            reject_stderr=True,
        )
        raw_frames = frames_json.get("frames")
        if not isinstance(raw_frames, list):
            raise UnsupportedMedia("no decodable video frames")
        frames, end_ticks = build_frame_entries(raw_frames, time_base)
        duration_us = ticks_to_us(end_ticks, time_base)
        if not 0 < duration_us <= max_duration_us:
            raise UnsupportedMedia(
                "video duration is out of range",
                details={"duration_us": duration_us, "max_duration_s": video_policy.max_duration_s},
            )

        normalized = {
            "format": {k: v for k, v in fmt.items() if k != "filename"},
            "streams": streams,
        }
        probe = ProbeResult(
            container="mp4",
            video_codec=codec,
            width_px=width,
            height_px=height,
            duration_us=duration_us,
            time_base=format_rational(time_base),
            nominal_fps=self._nominal_fps(video, len(frames), duration_us),
            frame_count=len(frames),
            has_audio=any(isinstance(s, dict) and s.get("codec_type") == "audio" for s in streams),
            probe_sha256=canonical_sha256(normalized),
            extractor=self.extractor,
        )
        return _Analysis(probe=probe, time_base=probe.time_base, frames=frames)

    @staticmethod
    def _nominal_fps(video: Mapping[str, Any], frame_count: int, duration_us: int) -> str:
        """Display-only rate; falls back to frames/duration when ffprobe reports 0/0."""
        for key in ("avg_frame_rate", "r_frame_rate"):
            try:
                return format_rational(parse_rational(video.get(key), field=key))
            except UnsupportedMedia:
                continue
        return format_rational(Fraction(frame_count * 1_000_000, duration_us))

    def _json(
        self, args: list[str], timeout_s: float, *, reject_stderr: bool = False
    ) -> dict[str, Any]:
        out = self._run(
            ["-v", "error", "-print_format", "json", *args], timeout_s, reject_stderr=reject_stderr
        )
        try:
            data = json.loads(out)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise UnsupportedMedia("ffprobe returned malformed output") from exc
        if not isinstance(data, dict):
            raise UnsupportedMedia("ffprobe returned malformed output")
        return data

    def _run(self, args: list[str], timeout_s: float, *, reject_stderr: bool = False) -> bytes:
        try:
            proc = subprocess.run(
                [self._bin, *args],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise UnsupportedMedia(
                "media probe timed out",
                code=ErrorCode.MEDIA_PROBE_TIMEOUT,
                details={"timeout_s": timeout_s},
            ) from exc
        except OSError as exc:
            raise ProbityError("ffprobe could not be executed") from exc
        if proc.returncode != 0:
            raise UnsupportedMedia(
                "ffprobe could not read the media", details={"exit_code": proc.returncode}
            )
        if reject_stderr and proc.stderr.strip():
            raise UnsupportedMedia("video frames are not cleanly decodable")
        return proc.stdout
