"""Write-once, content-addressed local storage for source videos plus the derived-path guard.

Layout under ``data_dir``::

    staging/upload-*.part                              private (0600) in-flight uploads
    source/sha256/{first2}/{sha256}/original.mp4       immutable originals (0444)
    derived/{video_id}/{kind}/{entity_id}/{filename}   every derivative

Immutability is application-level: the store never opens a source object with write flags, never
replaces one, and re-hashes after placement. It is not formal evidence custody.
"""

from __future__ import annotations

import errno
import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import BinaryIO

from probity.domain.enums import ReasonCode
from probity.domain.errors import (
    NotFound,
    PayloadTooLarge,
    SourceHashMismatch,
    SourcePathViolation,
    UnsupportedMedia,
    ValidationFailed,
)
from probity.domain.ids import SHA256_RE, is_uuid7, utc_now
from probity.media.hashing import CHUNK_SIZE, sha256_file
from probity.ports import SourceVerification, StagedUpload, StoredSource

SOURCE_URI_RE = re.compile(
    r"^source/sha256/(?P<prefix>[0-9a-f]{2})/(?P<sha256>[0-9a-f]{64})/original\.mp4$"
)
_SOURCE_TREE_RE = re.compile(r"(^|/)source/sha256/[0-9a-f]{2}/[0-9a-f]{64}(/|$)")
_KIND_RE = re.compile(r"^[a-z_]+$")
_ENTITY_RE = re.compile(r"^[A-Za-z0-9:._-]+$")
_FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_STAGING_PREFIX = "upload-"
_STAGING_SUFFIX = ".part"


def source_storage_uri(sha256: str) -> str:
    if not SHA256_RE.match(sha256):
        raise ValidationFailed("sha256 must be 64 lowercase hex characters")
    return f"source/sha256/{sha256[:2]}/{sha256}/original.mp4"


def derived_storage_uri(video_id: str, kind: str, entity_id: str, filename: str) -> str:
    """Relative ``derived/...`` URI after validating every component."""
    _validate_derived_components(video_id, kind, entity_id, filename)
    return f"derived/{video_id}/{kind}/{entity_id}/{filename}"


def assert_not_source(
    path: str | Path,
    *,
    source_root: str | Path | None = None,
    protected: tuple[str | Path, ...] = (),
) -> Path:
    """Resolve ``path`` (following symlinks) and refuse it if it lands in a source tree.

    Rejects any content-addressed ``source/sha256/..`` location, anything beneath ``source_root``,
    and any path that is (or hard-links to) one of the ``protected`` files. Returns the resolved
    path that writers must use.
    """
    resolved = Path(path).resolve()
    if _SOURCE_TREE_RE.search(resolved.as_posix()):
        raise SourcePathViolation("refusing to write inside the immutable source tree")
    if source_root is not None and resolved.is_relative_to(Path(source_root).resolve()):
        raise SourcePathViolation("refusing to write beneath source/")
    for item in protected:
        guarded = Path(item).resolve()
        if resolved == guarded:
            raise SourcePathViolation("refusing to write over a source file")
        if resolved.exists() and guarded.exists() and os.path.samefile(resolved, guarded):
            raise SourcePathViolation("refusing to write to a hard link of a source file")
    return resolved


def _validate_derived_components(video_id: str, kind: str, entity_id: str, filename: str) -> None:
    if not is_uuid7(video_id):
        raise SourcePathViolation("derived video_id must be a lowercase UUIDv7")
    if not _KIND_RE.match(kind):
        raise SourcePathViolation("derived kind must match [a-z_]+")
    for label, value, pattern in (
        ("entity_id", entity_id, _ENTITY_RE),
        ("filename", filename, _FILENAME_RE),
    ):
        if value in {".", ".."} or not pattern.match(value):
            raise SourcePathViolation(f"invalid derived {label}: {value!r}")


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file_readonly(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class LocalSourceStore:
    """Implements both ``SourceStore`` and ``DerivedStore`` over a local data directory."""

    def __init__(self, data_dir: str | Path) -> None:
        root = Path(data_dir)
        root.mkdir(parents=True, exist_ok=True)
        self._root = root.resolve()
        self._staging = self._root / "staging"
        self._source = self._root / "source"
        self._derived = self._root / "derived"
        for directory in (self._staging, self._source, self._derived):
            if directory.is_symlink():
                raise SourcePathViolation(f"{directory.name}/ must not be a symlink")
            directory.mkdir(exist_ok=True)
        os.chmod(self._staging, 0o700)

    @property
    def data_dir(self) -> Path:
        return self._root

    @property
    def source_root(self) -> Path:
        return self._source

    @property
    def derived_root(self) -> Path:
        return self._derived

    # ---------------------------------------------------------------------------- SourceStore

    def stage_upload(self, stream: BinaryIO, max_bytes: int) -> StagedUpload:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        fd, name = tempfile.mkstemp(
            prefix=_STAGING_PREFIX, suffix=_STAGING_SUFFIX, dir=self._staging
        )
        path = Path(name)
        digest = hashlib.sha256()
        total = 0
        limit = max_bytes + 1
        try:
            with os.fdopen(fd, "wb") as out:
                while total < limit:
                    chunk = stream.read(min(CHUNK_SIZE, limit - total))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise PayloadTooLarge(
                            "upload exceeds the maximum accepted size",
                            details={"max_bytes": max_bytes},
                        )
                    digest.update(chunk)
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            if total == 0:
                raise UnsupportedMedia("empty upload")
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return StagedUpload(staging_path=str(path), sha256=digest.hexdigest(), byte_length=total)

    def commit(self, staged: StagedUpload) -> StoredSource:
        staged_path = self._staged_path(staged)
        uri = source_storage_uri(staged.sha256)
        dest = self._root / uri
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._reject_symlinks(dest.parent)
        if os.path.lexists(dest):
            return self._deduplicate(staged, dest, uri)
        os.chmod(staged_path, 0o444)
        linked = False
        try:
            # link() never replaces an existing name, unlike rename().
            os.link(staged_path, dest, follow_symlinks=False)
            linked = True
        except FileExistsError:
            return self._deduplicate(staged, dest, uri)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            if os.path.lexists(dest):
                return self._deduplicate(staged, dest, uri)
            os.rename(staged_path, dest)
        if linked:
            staged_path.unlink()
        os.chmod(dest, 0o444)
        _fsync_file_readonly(dest)
        for directory in (
            dest.parent,
            dest.parent.parent,
            self._source / "sha256",
            self._source,
            self._staging,
        ):
            _fsync_dir(directory)
        observed, length = sha256_file(dest)
        if observed != staged.sha256 or length != staged.byte_length:
            # The object was created by this call and never matched; remove it rather than
            # leave bytes under the wrong content address.
            dest.unlink()
            raise SourceHashMismatch(
                "source bytes changed between staging and placement",
                details={"expected_sha256": staged.sha256, "observed_sha256": observed},
            )
        return StoredSource(
            sha256=observed,
            byte_length=length,
            storage_uri=uri,
            deduplicated=False,
            verified_at=utc_now(),
        )

    def discard(self, staged: StagedUpload) -> None:
        try:
            path = self._staged_path(staged)
        except NotFound:
            return
        path.unlink(missing_ok=True)

    def verify(self, storage_uri: str, expected_sha256: str) -> SourceVerification:
        if not SHA256_RE.match(expected_sha256):
            raise ValidationFailed("expected_sha256 must be 64 lowercase hex characters")
        path = self._source_path(storage_uri)
        try:
            observed: str | None = sha256_file(path)[0]
        except FileNotFoundError:
            observed = None
        verified = observed == expected_sha256
        return SourceVerification(
            storage_uri=storage_uri,
            expected_sha256=expected_sha256,
            observed_sha256=observed,
            verified=verified,
            checked_at=utc_now(),
            reason_code=(
                ReasonCode.SOURCE_HASH_VERIFIED if verified else ReasonCode.SOURCE_HASH_MISMATCH
            ),
        )

    def resolve_read_path(self, storage_uri: str) -> str:
        path = self._source_path(storage_uri)
        if not path.is_file():
            raise NotFound("source object not found", details={"storage_uri": storage_uri})
        return str(path)

    # --------------------------------------------------------------------------- DerivedStore

    def derived_path(self, video_id: str, kind: str, entity_id: str, filename: str) -> str:
        _validate_derived_components(video_id, kind, entity_id, filename)
        target = self._derived / video_id / kind / entity_id / filename
        self._reject_symlinks(target.parent, base=self._derived)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._reject_symlinks(target.parent, base=self._derived)
        resolved = self.assert_not_source(target)
        if not resolved.is_relative_to(self._derived):
            raise SourcePathViolation("derived path escapes derived/")
        return str(target)

    def assert_not_source(self, path: str | Path) -> Path:
        resolved = assert_not_source(path, source_root=self._source)
        if not resolved.is_relative_to(self._root):
            raise SourcePathViolation("path escapes the data directory")
        return resolved

    # -------------------------------------------------------------------------------- helpers

    def _staged_path(self, staged: StagedUpload) -> Path:
        path = Path(staged.staging_path)
        if (
            not path.is_absolute()
            or path.parent != self._staging
            or not path.name.startswith(_STAGING_PREFIX)
            or not path.name.endswith(_STAGING_SUFFIX)
        ):
            raise SourcePathViolation("staged upload is not inside this store's staging area")
        if path.is_symlink():
            raise SourcePathViolation("staged upload must not be a symlink")
        if not path.is_file():
            raise NotFound("staged upload not found")
        return path

    def _source_path(self, storage_uri: str) -> Path:
        match = SOURCE_URI_RE.match(storage_uri) if isinstance(storage_uri, str) else None
        if match is None or match["prefix"] != match["sha256"][:2]:
            raise SourcePathViolation("not a content-addressed source URI")
        path = self._root / storage_uri
        self._reject_symlinks(path, base=self._source)
        if not path.resolve().is_relative_to(self._source):
            raise SourcePathViolation("source URI escapes source/")
        return path

    def _reject_symlinks(self, path: Path, base: Path | None = None) -> None:
        base = self._root if base is None else base
        current = base
        for part in path.relative_to(base).parts:
            current = current / part
            if current.is_symlink():
                raise SourcePathViolation("symlinks are not allowed in storage paths")

    def _deduplicate(self, staged: StagedUpload, dest: Path, uri: str) -> StoredSource:
        if dest.is_symlink() or not dest.is_file():
            raise SourcePathViolation("existing source object is not a regular file")
        observed, length = sha256_file(dest)
        self.discard(staged)
        if observed != staged.sha256:
            raise SourceHashMismatch(
                "existing source object does not match its content address",
                details={"expected_sha256": staged.sha256, "observed_sha256": observed},
            )
        return StoredSource(
            sha256=observed,
            byte_length=length,
            storage_uri=uri,
            deduplicated=True,
            verified_at=utc_now(),
        )
