"""Golden SHA-256 vectors for streaming and file hashing."""

from __future__ import annotations

import io
from pathlib import Path

from probity.media.hashing import CHUNK_SIZE, sha256_file, sha256_stream

EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
ONE_MIB_PLUS_17 = "ee00e469f2bdf447766b815707f434911e270ae9f5e77f3a8688a03b3026216f"
ONE_MIB_PLUS = b"\x5a" * (CHUNK_SIZE + 17)


def test_sha256_empty_stream() -> None:
    digest, n = sha256_stream(io.BytesIO(b""))
    assert digest == EMPTY
    assert n == 0


def test_sha256_abc_fips_vector() -> None:
    digest, n = sha256_stream(io.BytesIO(b"abc"))
    assert digest == ABC
    assert n == 3


def test_sha256_abc_chunk_size_one() -> None:
    digest, n = sha256_stream(io.BytesIO(b"abc"), chunk_size=1)
    assert digest == ABC
    assert n == 3


def test_sha256_one_mib_plus() -> None:
    digest, n = sha256_stream(io.BytesIO(ONE_MIB_PLUS), chunk_size=CHUNK_SIZE)
    assert n == CHUNK_SIZE + 17
    assert digest == ONE_MIB_PLUS_17


def test_sha256_file_matches_stream(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(ONE_MIB_PLUS)
    digest, n = sha256_file(path)
    assert (digest, n) == sha256_stream(io.BytesIO(ONE_MIB_PLUS))
    assert digest == ONE_MIB_PLUS_17
