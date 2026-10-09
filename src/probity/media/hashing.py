"""Streaming SHA-256 over binary streams and files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO

CHUNK_SIZE = 1024 * 1024


def sha256_stream(stream: BinaryIO, chunk_size: int = CHUNK_SIZE) -> tuple[str, int]:
    """Hash ``stream`` to EOF in bounded chunks; return ``(hex_digest, byte_length)``."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
    return digest.hexdigest(), total


def sha256_file(path: str | Path, chunk_size: int = CHUNK_SIZE) -> tuple[str, int]:
    with Path(path).open("rb") as handle:
        return sha256_stream(handle, chunk_size)
