"""Lane E: NON-EVIDENTIARY conventional upscaler baseline (section 12). SIGNATURES FROZEN.

The baseline (Real-ESRGAN x4plus in live mode) is for side-by-side evaluation only. Its output is
written only under ``derived/{video_id}/baseline/{run_id}/`` as an ``AssetRef`` of kind
``BASELINE`` with ``non_evidentiary=True``; ``io.assert_evidentiary_input`` rejects it, so it can
never become a donor, tracker input, or ``FrameReference``. Nothing in the reconstruction path
imports this module. Tests use a fake ``Upscaler``; no model download, no network.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from probity.domain.enums import AssetKind
from probity.domain.ids import SHA256_RE, is_uuid7, parse_utc
from probity.domain.models import AssetRef
from probity.ports import DerivedStore
from probity.reconstruction.determinism import seeded_uuid7
from probity.reconstruction.io import (
    atomic_write_bytes,
    decode_png,
    encode_png,
    file_sha256,
    pixel_sha256,
    sha256_bytes,
    write_json,
)
from probity.reconstruction.types import Obs

LANE = "lane E"
BASELINE_KIND = "baseline"
BASELINE_FILENAME = "baseline.png"
BASELINE_MODEL_ID = "real-esrgan-x4plus"
BASELINE_METADATA_FILENAME = "baseline.json"
BASELINE_LABEL = "non-evidentiary baseline"
BASELINE_MEDIA_TYPE = "image/png"
METADATA_SCHEMA_VERSION = "1.0"

_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class Upscaler(Protocol):
    model_id: str
    model_sha256: str | None
    license_note: str

    def upscale(self, bgr: np.ndarray) -> np.ndarray: ...


def _check_segment(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SEGMENT_RE.match(value) or value.startswith("."):
        raise ValueError(f"{name} {value!r} is not a safe single path segment")


def baseline_uri(video_id: str, run_id: str, filename: str = BASELINE_FILENAME) -> str:
    """``derived/{video_id}/baseline/{run_id}/{filename}``."""
    if not isinstance(video_id, str) or not is_uuid7(video_id):
        raise ValueError(f"video_id {video_id!r} is not a UUIDv7")
    _check_segment("run_id", run_id)
    _check_segment("filename", filename)
    return f"derived/{video_id}/{BASELINE_KIND}/{run_id}/{filename}"


def _check_image(image: object, what: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{what} must be a numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{what} must be BGR uint8 (H, W, 3), got {image.dtype} {image.shape}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"{what} is empty")
    return image


def run_baseline(upscaler: Upscaler, target: Obs) -> np.ndarray:
    """Upscale ``target.crop`` only; returns BGR uint8. Never reads donors or results."""
    crop = _check_image(target.crop, "target.crop")
    h, w = crop.shape[:2]
    out = _check_image(upscaler.upscale(np.ascontiguousarray(crop).copy()), "upscaler output")
    oh, ow = out.shape[:2]
    if oh % h or ow % w or oh // h != ow // w:
        raise ValueError(
            f"upscaler output {ow}x{oh} is not one integer scale of the {w}x{h} target crop"
        )
    return np.ascontiguousarray(out).copy()


def _check_upscaler(upscaler: Upscaler) -> None:
    if not isinstance(upscaler.model_id, str) or not upscaler.model_id.strip():
        raise ValueError("upscaler.model_id must be a non-empty string")
    sha = upscaler.model_sha256
    if sha is not None and (not isinstance(sha, str) or not SHA256_RE.match(sha)):
        raise ValueError("upscaler.model_sha256 must be None or 64 lowercase hex characters")
    if not isinstance(upscaler.license_note, str) or not upscaler.license_note.strip():
        raise ValueError("upscaler.license_note must be recorded")


def _store_path(store: DerivedStore, video_id: str, run_id: str, filename: str) -> Path:
    path = Path(store.derived_path(video_id, BASELINE_KIND, run_id, filename))
    expected = ("derived", video_id, BASELINE_KIND, run_id, filename)
    if not path.is_absolute() or path.parts[-5:] != expected:
        raise ValueError(f"derived store returned {path}, not .../{'/'.join(expected)}")
    return path


def baseline_metadata(asset: AssetRef, *, video_id: str, run_id: str, pixel_sha: str,
                      width_px: int, height_px: int, upscaler: Upscaler) -> dict[str, Any]:
    """Sidecar record: the model hash and license are recorded with every baseline output."""
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "label": BASELINE_LABEL,
        "non_evidentiary": True,
        "asset_id": asset.asset_id,
        "case_id": asset.case_id,
        "video_id": video_id,
        "run_id": run_id,
        "kind": str(asset.kind),
        "storage_uri": asset.storage_uri,
        "metadata_uri": baseline_uri(video_id, run_id, BASELINE_METADATA_FILENAME),
        "media_type": asset.media_type,
        "byte_length": asset.byte_length,
        "sha256": asset.sha256,
        "pixel_sha256": pixel_sha,
        "width_px": width_px,
        "height_px": height_px,
        "model_id": upscaler.model_id,
        "model_sha256": upscaler.model_sha256,
        "license_note": upscaler.license_note,
        "created_at": asset.created_at,
    }


def write_baseline(image: np.ndarray, store: DerivedStore, *, case_id: str, video_id: str,
                   run_id: str, upscaler: Upscaler, created_at: str) -> AssetRef:
    """Atomically write a lossless PNG via ``store.derived_path(video_id, "baseline", run_id, ...)``
    and return its ``AssetRef`` (kind BASELINE, non_evidentiary=True)."""
    _check_image(image, "baseline image")
    _check_upscaler(upscaler)
    uri = baseline_uri(video_id, run_id, BASELINE_FILENAME)
    unix_ms = int(parse_utc(created_at).timestamp() * 1000)

    data = encode_png(np.ascontiguousarray(image))
    pixel_sha = pixel_sha256(image)
    if pixel_sha256(decode_png(data)) != pixel_sha:
        raise ValueError("baseline PNG encoding is not lossless")
    asset = AssetRef(
        asset_id=seeded_uuid7(f"{uri}|{pixel_sha}", unix_ms),
        case_id=case_id,
        kind=AssetKind.BASELINE,
        storage_uri=uri,
        media_type=BASELINE_MEDIA_TYPE,
        byte_length=len(data),
        sha256=sha256_bytes(data),
        non_evidentiary=True,
        created_at=created_at,
    )

    png_path = _store_path(store, video_id, run_id, BASELINE_FILENAME)
    meta_path = _store_path(store, video_id, run_id, BASELINE_METADATA_FILENAME)
    atomic_write_bytes(png_path, data)
    if file_sha256(png_path) != asset.sha256:
        raise OSError(f"{png_path}: written bytes do not match the encoded PNG")
    h, w = image.shape[:2]
    write_json(meta_path, baseline_metadata(asset, video_id=video_id, run_id=run_id,
                                            pixel_sha=pixel_sha, width_px=w, height_px=h,
                                            upscaler=upscaler))
    return asset
