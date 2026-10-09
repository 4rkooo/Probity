"""Write-once source storage, payload limits, and path isolation."""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from probity.domain.enums import ReasonCode
from probity.domain.errors import (
    NotFound,
    PayloadTooLarge,
    SourceHashMismatch,
    SourcePathViolation,
    UnsupportedMedia,
)
from probity.domain.ids import new_uuid7
from probity.media.hashing import CHUNK_SIZE, sha256_file
from probity.media.source_store import (
    LocalSourceStore,
    assert_not_source,
    source_storage_uri,
)
from probity.ports import StagedUpload

PAYLOAD = b"probity-source-bytes-not-a-video"
MAX_SHA = "ab" * 32


class GuardedStream:
    """Binary stream that fails the test if more than ``max_read`` bytes are requested."""

    def __init__(self, blob: bytes, max_read: int) -> None:
        self._blob = blob
        self._pos = 0
        self.max_read = max_read
        self.requested = 0

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            raise AssertionError("unbounded read is not allowed")
        if self.requested + n > self.max_read:
            raise AssertionError(
                f"read({n}) would exceed max_read={self.max_read} (already {self.requested})"
            )
        self.requested += n
        chunk = self._blob[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk


def _store(tmp_path: Path) -> LocalSourceStore:
    return LocalSourceStore(tmp_path / "data")


def test_empty_upload_is_unsupported(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(UnsupportedMedia, match="empty"):
        store.stage_upload(io.BytesIO(b""), max_bytes=1024)
    assert list((store.data_dir / "staging").glob("upload-*.part")) == []


def test_payload_too_large_does_not_read_past_limit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    max_bytes = 64
    blob = b"x" * (max_bytes + 50)
    stream = GuardedStream(blob, max_read=max_bytes + 1)
    with pytest.raises(PayloadTooLarge):
        store.stage_upload(stream, max_bytes=max_bytes)
    assert stream.requested <= max_bytes + 1
    assert list((store.data_dir / "staging").glob("upload-*.part")) == []


def test_payload_too_large_chunked_stream(tmp_path: Path) -> None:
    store = _store(tmp_path)
    max_bytes = CHUNK_SIZE
    blob = b"y" * (max_bytes + 3)
    stream = GuardedStream(blob, max_read=max_bytes + 1)
    with pytest.raises(PayloadTooLarge):
        store.stage_upload(stream, max_bytes=max_bytes)
    assert list((store.data_dir / "staging").glob("upload-*.part")) == []


def test_commit_is_write_once_readonly_and_content_addressed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    staged = store.stage_upload(io.BytesIO(PAYLOAD), max_bytes=1024)
    first = store.commit(staged)
    dest = Path(store.resolve_read_path(first.storage_uri))
    assert first.deduplicated is False
    assert first.storage_uri == source_storage_uri(first.sha256)
    assert dest.is_file()
    assert dest.stat().st_mode & 0o777 == 0o444
    with pytest.raises(PermissionError):
        os.open(dest, os.O_WRONLY)
    digest, length = sha256_file(dest)
    assert digest == first.sha256
    assert length == first.byte_length == len(PAYLOAD)

    staged_again = store.stage_upload(io.BytesIO(PAYLOAD), max_bytes=1024)
    second = store.commit(staged_again)
    assert second.deduplicated is True
    assert second.sha256 == first.sha256
    assert second.storage_uri == first.storage_uri
    assert not Path(staged_again.staging_path).exists()
    originals = list((store.source_root).rglob("original.mp4"))
    assert len(originals) == 1


def test_post_write_rehash_mismatch_removes_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    staged = store.stage_upload(io.BytesIO(PAYLOAD), max_bytes=1024)
    import probity.media.source_store as module

    def fake_sha256_file(path: str | Path, chunk_size: int = CHUNK_SIZE) -> tuple[str, int]:
        return "00" * 32, 1

    monkeypatch.setattr(module, "sha256_file", fake_sha256_file)
    with pytest.raises(SourceHashMismatch):
        store.commit(staged)
    assert list(store.source_root.rglob("original.mp4")) == []


def test_altered_source_verify_mismatch(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stored = store.commit(store.stage_upload(io.BytesIO(PAYLOAD), max_bytes=1024))
    dest = Path(store.resolve_read_path(stored.storage_uri))
    os.chmod(dest, 0o644)
    dest.write_bytes(PAYLOAD + b"-tampered")
    os.chmod(dest, 0o444)
    result = store.verify(stored.storage_uri, stored.sha256)
    assert result.verified is False
    assert result.observed_sha256 != stored.sha256
    assert result.reason_code is ReasonCode.SOURCE_HASH_MISMATCH


def test_verify_missing_file_observed_none(tmp_path: Path) -> None:
    store = _store(tmp_path)
    uri = source_storage_uri(MAX_SHA)
    (store.source_root / "sha256" / MAX_SHA[:2] / MAX_SHA).mkdir(parents=True)
    result = store.verify(uri, MAX_SHA)
    assert result.observed_sha256 is None
    assert result.verified is False
    assert result.reason_code is ReasonCode.SOURCE_HASH_MISMATCH


def test_verify_ok(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stored = store.commit(store.stage_upload(io.BytesIO(PAYLOAD), max_bytes=1024))
    result = store.verify(stored.storage_uri, stored.sha256)
    assert result.verified is True
    assert result.observed_sha256 == stored.sha256
    assert result.reason_code is ReasonCode.SOURCE_HASH_VERIFIED


def test_discard_removes_staging(tmp_path: Path) -> None:
    store = _store(tmp_path)
    staged = store.stage_upload(io.BytesIO(PAYLOAD), max_bytes=1024)
    assert Path(staged.staging_path).is_file()
    store.discard(staged)
    assert not Path(staged.staging_path).exists()
    store.discard(staged)


def test_resolve_read_path_rejects_traversal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(SourcePathViolation):
        store.resolve_read_path("source/sha256/../etc/passwd")
    with pytest.raises(SourcePathViolation):
        store.resolve_read_path(f"source/sha256/00/{MAX_SHA}/original.mp4")
    with pytest.raises(SourcePathViolation):
        store.resolve_read_path("/etc/passwd")
    with pytest.raises(NotFound):
        store.resolve_read_path(source_storage_uri(MAX_SHA))


def test_resolve_read_path_rejects_symlink_escape(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stored = store.commit(store.stage_upload(io.BytesIO(PAYLOAD), max_bytes=1024))
    dest = Path(store.resolve_read_path(stored.storage_uri))
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"escaped")
    dest.unlink()
    dest.symlink_to(outside)
    with pytest.raises(SourcePathViolation):
        store.resolve_read_path(stored.storage_uri)


def test_derived_path_isolation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    video_id = new_uuid7()
    derived = Path(store.derived_path(video_id, "thumbs", "seg-0", "thumb.jpg"))
    assert derived.is_relative_to(store.derived_root)
    assert not derived.is_relative_to(store.source_root)
    with pytest.raises(SourcePathViolation):
        store.derived_path(video_id, "thumbs", "..", "thumb.jpg")
    with pytest.raises(SourcePathViolation):
        store.derived_path("not-a-uuid", "thumbs", "x", "t.jpg")
    with pytest.raises(SourcePathViolation):
        store.derived_path(video_id, "THUMBS", "x", "t.jpg")
    with pytest.raises(SourcePathViolation):
        store.derived_path(video_id, "thumbs", "x", "../t.jpg")
    with pytest.raises(SourcePathViolation):
        assert_not_source(store.source_root / "anything.png", source_root=store.source_root)


def test_derived_path_rejects_source_symlink(tmp_path: Path) -> None:
    store = _store(tmp_path)
    video_id = new_uuid7()
    first = Path(store.derived_path(video_id, "frames", "f417", "frame.png"))
    entity = first.parent
    for child in entity.iterdir():
        child.unlink()
    entity.rmdir()
    entity.symlink_to(store.source_root)
    with pytest.raises(SourcePathViolation):
        store.derived_path(video_id, "frames", "f417", "other.png")


def test_assert_not_source_rejects_hardlink(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stored = store.commit(store.stage_upload(io.BytesIO(PAYLOAD), max_bytes=1024))
    dest = Path(store.resolve_read_path(stored.storage_uri))
    alias = store.derived_root / "alias.mp4"
    os.link(dest, alias)
    with pytest.raises(SourcePathViolation):
        assert_not_source(alias, protected=(dest,))


def test_staged_path_must_belong_to_store(tmp_path: Path) -> None:
    store = _store(tmp_path)
    outsider = tmp_path / "upload-x.part"
    outsider.write_bytes(PAYLOAD)
    staged = StagedUpload(staging_path=str(outsider), sha256="11" * 32, byte_length=len(PAYLOAD))
    with pytest.raises(SourcePathViolation):
        store.commit(staged)
