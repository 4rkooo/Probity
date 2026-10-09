"""Identifier, wall-clock, and canonical-hash primitives shared by every module."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import UTC, datetime
from typing import Any

UUID7_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RFC3339_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")
FRAME_ID_RE = re.compile(r"^(?P<video_id>[0-9a-f-]{36}):f(?P<n>0|[1-9]\d*)$")
SEGMENT_ID_RE = re.compile(r"^(?P<video_id>[0-9a-f-]{36}):s(?P<n>\d{4,})$")


def new_uuid7(unix_ms: int | None = None) -> str:
    """Return a lowercase RFC 9562 UUIDv7 string."""
    ms = time.time_ns() // 1_000_000 if unix_ms is None else unix_ms
    raw = bytearray(ms.to_bytes(6, "big") + os.urandom(10))
    raw[6] = 0x70 | (raw[6] & 0x0F)
    raw[8] = 0x80 | (raw[8] & 0x3F)
    return str(uuid.UUID(bytes=bytes(raw)))


def is_uuid7(value: str) -> bool:
    return bool(UUID7_RE.match(value))


def frame_id(video_id: str, frame_number: int) -> str:
    """Deterministic frame ID: ``{video_id}:f{frame_number}`` (decode-output index, unpadded)."""
    if frame_number < 0:
        raise ValueError("frame_number must be >= 0")
    return f"{video_id}:f{frame_number}"


def segment_id(video_id: str, ordinal: int) -> str:
    """Deterministic segment ID: ``{video_id}:s{ordinal:04d}`` (zero-based)."""
    if ordinal < 0:
        raise ValueError("ordinal must be >= 0")
    return f"{video_id}:s{ordinal:04d}"


def parse_frame_id(value: str) -> tuple[str, int]:
    match = FRAME_ID_RE.match(value)
    if not match or not is_uuid7(match["video_id"]):
        raise ValueError(f"invalid frame_id: {value!r}")
    return match["video_id"], int(match["n"])


def parse_segment_id(value: str) -> tuple[str, int]:
    match = SEGMENT_ID_RE.match(value)
    if not match or not is_uuid7(match["video_id"]):
        raise ValueError(f"invalid segment_id: {value!r}")
    return match["video_id"], int(match["n"])


def utc_now() -> str:
    """RFC 3339 UTC wall-clock string with microsecond precision and ``Z`` suffix."""
    return format_utc(datetime.now(UTC))


def format_utc(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("naive datetime is not allowed")
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_utc(value: str) -> datetime:
    if not RFC3339_UTC_RE.match(value):
        raise ValueError(f"not an RFC 3339 UTC timestamp: {value!r}")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def canonical_json(data: Any) -> bytes:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8, NaN/Infinity rejected."""
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(data: Any) -> str:
    return sha256_hex(canonical_json(data))
