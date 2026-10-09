"""Upload staging: Content-Length gate, XOR fixture/body, stream into SourceStore."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import anyio
from fastapi import Request

from probity.api.deps import AppServices
from probity.domain.errors import (
    FixtureNotFound,
    PayloadTooLarge,
    UnsupportedMedia,
    ValidationFailed,
)
from probity.domain.fixtures import FixtureManifest
from probity.ports import SourceStore, StagedUpload

ALLOWED_UPLOAD_TYPES = frozenset({"video/mp4", "application/octet-stream"})


def _media_type(request: Request) -> str:
    raw = request.headers.get("content-type", "")
    return raw.split(";", 1)[0].strip().lower()


def _content_length(request: Request) -> int | None:
    raw = request.headers.get("content-length")
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValidationFailed("Content-Length is not an integer.") from exc


def fixture_manifest(services: AppServices, fixture_id: str) -> FixtureManifest:
    for item in services.fixture_catalog.fixtures:
        if item.fixture_id == fixture_id:
            return item
    raise FixtureNotFound(f"Unknown fixture_id {fixture_id!r}.")


def _staging_dir(store: SourceStore) -> Path:
    raw = getattr(store, "staging_dir", None)
    if isinstance(raw, Path):
        raw.mkdir(parents=True, exist_ok=True)
        return raw
    return Path(tempfile.gettempdir())


async def stage_from_request(request: Request, store: SourceStore, max_bytes: int) -> StagedUpload:
    """Copy the ASGI body to a temp file, stopping at ``max_bytes``, then stage it.

    Reading stays on the event loop so ``BaseHTTPMiddleware`` can feed the body.
    ``SourceStore.stage_upload`` still enforces the byte cap and computes the hash.
    """

    fd, name = tempfile.mkstemp(prefix="body-", dir=_staging_dir(store))
    path = Path(name)
    total = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            async for chunk in request.stream():
                if not chunk:
                    continue
                if total + len(chunk) > max_bytes:
                    raise PayloadTooLarge("Upload exceeds the configured maximum size.")
                handle.write(chunk)
                total += len(chunk)
        if total <= 0:
            raise ValidationFailed("Upload body is required unless fixture_id is set.")

        def _stage() -> StagedUpload:
            with path.open("rb") as handle:
                return store.stage_upload(handle, max_bytes)

        return await anyio.to_thread.run_sync(_stage)
    finally:
        path.unlink(missing_ok=True)


async def stage_from_path(path: Path, store: SourceStore, max_bytes: int) -> StagedUpload:
    def _stage() -> StagedUpload:
        with path.open("rb") as handle:
            return store.stage_upload(handle, max_bytes)

    return await anyio.to_thread.run_sync(_stage)


async def accept_upload(
    request: Request,
    services: AppServices,
    *,
    fixture_id: str | None,
) -> tuple[StagedUpload, FixtureManifest | None]:
    max_bytes = services.policy.video.max_bytes
    length = _content_length(request)
    if length is not None and length > max_bytes:
        raise PayloadTooLarge("Upload exceeds the configured maximum size.")

    if fixture_id is not None:
        if length is not None and length > 0:
            raise ValidationFailed("Pass either a raw MP4 body or ?fixture_id=, not both.")
        manifest = fixture_manifest(services, fixture_id)
        source_path = services.fixture_root / manifest.source.path
        if not source_path.is_file():
            raise FixtureNotFound(f"Unknown fixture_id {fixture_id!r}.")
        staged = await stage_from_path(source_path, services.source_store, max_bytes)
        return staged, manifest

    media = _media_type(request)
    if media not in ALLOWED_UPLOAD_TYPES:
        raise UnsupportedMedia("Uploads must be video/mp4 or application/octet-stream.")
    if length == 0:
        raise ValidationFailed("Upload body is required unless fixture_id is set.")
    staged = await stage_from_request(request, services.source_store, max_bytes)
    return staged, None
