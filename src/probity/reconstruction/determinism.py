"""Seeded identifiers and clocks so golden fixtures are bit-stable across runs and machines."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from probity.domain.ids import format_utc, parse_utc


def seeded_uuid7(label: str, unix_ms: int) -> str:
    """UUIDv7 whose 74 random bits come from SHA-256(label); same label and time, same ID."""
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    raw = bytearray(unix_ms.to_bytes(6, "big") + digest[:10])
    raw[6] = 0x70 | (raw[6] & 0x0F)
    raw[8] = 0x80 | (raw[8] & 0x3F)
    return str(uuid.UUID(bytes=bytes(raw)))


def unit_float(label: str) -> float:
    """Deterministic float in [0, 1) derived from SHA-256(label); independent of NumPy streams."""
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@dataclass(frozen=True)
class FixedClock:
    """Monotonic fake clock: ``at(i)`` is ``start + i * step_us`` as an RFC 3339 UTC string."""

    start: str
    step_us: int = 1_000

    def at(self, tick: int) -> str:
        return format_utc(parse_utc(self.start) + timedelta(microseconds=tick * self.step_us))

    @property
    def unix_ms(self) -> int:
        return int(parse_utc(self.start).timestamp() * 1000)


def utc_from_unix_ms(unix_ms: int) -> str:
    return format_utc(datetime.fromtimestamp(unix_ms / 1000, tz=UTC))
