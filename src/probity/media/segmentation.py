"""Deterministic overlapping segment windows in exact integer microseconds.

Windows are ``duration_s`` wide with a step of ``duration_s - overlap_s`` (8 s / 6 s in the
demo policy). Frame indices come from the exact-PTS manifest: the first frame whose
``pts_us >= start`` and the last frame whose ``pts_us < end``. The final window always
includes the last decoded frame. Variable frame rate is allowed; frame identity is never
rounded from a nominal rate.
"""

from __future__ import annotations

from collections.abc import Sequence

from probity.domain.errors import UnsupportedMedia
from probity.domain.models import FrameManifest, FrameManifestEntry
from probity.domain.policy import PolicyConfig
from probity.ports import SegmentWindow


def _first_frame_at_or_after(frames: Sequence[FrameManifestEntry], pts_us: int) -> int | None:
    for entry in frames:
        if entry.pts_us >= pts_us:
            return entry.frame_number
    return None


def _last_frame_before(frames: Sequence[FrameManifestEntry], pts_us: int) -> int | None:
    last: int | None = None
    for entry in frames:
        if entry.pts_us < pts_us:
            last = entry.frame_number
        else:
            break
    return last


class DeterministicSegmenter:
    """``Segmenter`` over policy segment duration/overlap, using only integer microsecond math."""

    def __init__(self, policy: PolicyConfig) -> None:
        self._policy = policy

    def plan(self, manifest: FrameManifest, duration_us: int) -> tuple[SegmentWindow, ...]:
        if duration_us <= 0:
            raise UnsupportedMedia("duration_us must be positive")
        frames = manifest.frames
        if not frames:
            raise UnsupportedMedia("frame manifest is empty")
        window_us = self._policy.segment.duration_us
        step_us = self._policy.segment.step_us
        planned: list[SegmentWindow] = []
        start = 0
        ordinal = 0
        while True:
            end = min(start + window_us, duration_us)
            is_final = end >= duration_us
            start_frame = _first_frame_at_or_after(frames, start)
            end_frame = _last_frame_before(frames, end)
            if is_final:
                end_frame = frames[-1].frame_number
            if start_frame is None or end_frame is None or end_frame < start_frame:
                raise UnsupportedMedia(
                    "segment window contains no frames",
                    details={"start_pts_us": start, "end_pts_us": end, "ordinal": ordinal},
                )
            planned.append(
                SegmentWindow(
                    ordinal=ordinal,
                    start_pts_us=start,
                    end_pts_us=end,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )
            if is_final:
                break
            start += step_us
            ordinal += 1
        return tuple(planned)
