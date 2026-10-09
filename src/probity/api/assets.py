"""Authorized local asset streaming with byte-range support."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import Request
from fastapi.responses import Response

from probity.api.deps import AppServices
from probity.api.errors import json_error
from probity.domain.enums import AssetKind, ErrorCode
from probity.domain.errors import NotFound, SourcePathViolation, ValidationFailed
from probity.domain.models import AssetRef

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


def repo_root(services: AppServices) -> Path:
    """``fixture_root`` is ``fixtures/demo``; contracts and demo live under the repo root."""

    return services.fixture_root.resolve().parent.parent


def resolve_asset_file(services: AppServices, asset: AssetRef) -> Path:
    uri = asset.storage_uri
    if ".." in uri.split("/"):
        raise SourcePathViolation("path traversal is not allowed")
    if asset.kind is AssetKind.SOURCE_VIDEO or uri.startswith("source/"):
        return Path(services.source_store.resolve_read_path(uri))
    if uri.startswith("derived/"):
        root = services.settings.data_dir.resolve()
        path = (root / uri).resolve()
        if not path.is_relative_to(root):
            raise SourcePathViolation("derived path escaped the data root")
        return path
    if uri.startswith("fixtures/"):
        root = repo_root(services)
        path = (root / uri).resolve()
        if not path.is_relative_to(root):
            raise SourcePathViolation("fixture path escaped the repository root")
        return path
    raise SourcePathViolation("asset storage_uri is not a permitted prefix")


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    match = _RANGE_RE.fullmatch(header.strip())
    if match is None:
        raise ValidationFailed("Only a single bytes=start-end Range is supported.")
    start_s, end_s = match.group(1), match.group(2)
    if start_s == "" and end_s == "":
        raise ValidationFailed("Range must specify a start or suffix length.")
    if start_s == "":
        suffix = int(end_s)
        if suffix <= 0:
            return None
        start = max(size - suffix, 0)
        end = size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
        if end >= size:
            end = size - 1
    if start >= size or start > end:
        return None
    return start, end


def build_asset_response(
    request: Request,
    services: AppServices,
    asset: AssetRef,
    range_header: str | None,
) -> Response:
    path = resolve_asset_file(services, asset)
    if not path.is_file():
        raise NotFound("Asset bytes are not available.")
    size = path.stat().st_size
    if size != asset.byte_length:
        raise SourcePathViolation("asset byte_length does not match stored bytes")

    headers = {
        "Cache-Control": "no-store",
        "Accept-Ranges": "bytes",
        "Content-Type": asset.media_type,
    }
    if not range_header:
        data = path.read_bytes()
        headers["Content-Length"] = str(size)
        return Response(content=data, media_type=asset.media_type, headers=headers)

    bounds = _parse_range(range_header, size)
    if bounds is None:
        return json_error(
            request,
            status_code=416,
            code=ErrorCode.VALIDATION_FAILED,
            message="Requested range is not satisfiable.",
            details={"size": size},
        )
    start, end = bounds
    length = end - start + 1
    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read(length)
    headers["Content-Length"] = str(len(data))
    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return Response(
        content=data,
        status_code=206,
        media_type=asset.media_type,
        headers=headers,
    )
