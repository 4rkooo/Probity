"""Atomic artifact I/O, pixel hashing, and the evidentiary frame loader.

``pixel_sha256`` hashes decoded pixels, not encoded bytes: PNG/zlib output can differ between
library builds while the pixels are identical. Golden comparisons use pixel and array digests;
file SHA-256 values are recorded for the committed bytes only.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel

from probity.domain.enums import AssetKind, ReasonCode
from probity.domain.errors import ProbityError
from probity.domain.models import AssetRef, FrameReference

PIXEL_HASH_VERSION = "probity-pixel-v1"
ARRAY_HASH_VERSION = "probity-array-v1"
PNG_COMPRESSION = 6
NPZ_COMPRESSLEVEL = 6
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

# Path segments that only ever hold derived, generated, or non-evidentiary pixels.
NON_EVIDENTIARY_SEGMENTS = frozenset({"baseline", "reconstruction", "reconstructions"})
NON_EVIDENTIARY_ASSET_KINDS = frozenset(
    {
        AssetKind.BASELINE,
        AssetKind.RESULT_PNG,
        AssetKind.INSPECTION_CLIP,
        AssetKind.THUMBNAIL,
        AssetKind.PROVENANCE_PREVIEW,
    }
)


class NonEvidentiaryInput(ProbityError):
    """A generated, upscaled, reconstructed, or otherwise non-source asset was offered as input."""

    reason_code = ReasonCode.GENERATED_INPUT_REJECTED


class FrameIntegrityError(ProbityError):
    """Decoded pixels do not match the recorded ``pixel_sha256`` or frame geometry."""

    reason_code = ReasonCode.DECODE_NOT_DETERMINISTIC


# ------------------------------------------------------------------------------------------------
# Hashing
# ------------------------------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(bgr: np.ndarray) -> str:
    """SHA-256 of ``probity-pixel-v1|uint8|H,W,3|RGB\\n`` followed by C-order RGB bytes."""
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("pixel_sha256 expects an HxWx3 uint8 image")
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w, _ = rgb.shape
    header = f"{PIXEL_HASH_VERSION}|uint8|{h},{w},3|RGB\n".encode("ascii")
    return sha256_bytes(header + rgb.tobytes())


def array_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Order-independent digest of named arrays (name, dtype, shape, C-order bytes)."""
    digest = hashlib.sha256(f"{ARRAY_HASH_VERSION}\n".encode("ascii"))
    for name in sorted(arrays):
        arr = np.ascontiguousarray(arrays[name])
        shape = ",".join(str(d) for d in arr.shape)
        digest.update(f"{name}|{arr.dtype.str}|{shape}\n".encode("ascii"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


# ------------------------------------------------------------------------------------------------
# Atomic writes
# ------------------------------------------------------------------------------------------------


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via a sibling temp file and ``os.replace``; the handle is closed before the rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def json_bytes(data: Any) -> bytes:
    """Pretty JSON with ``\\n`` line endings regardless of platform."""
    if isinstance(data, BaseModel):
        data = data.model_dump(mode="json")
    elif isinstance(data, (list, tuple)):
        data = [d.model_dump(mode="json") if isinstance(d, BaseModel) else d for d in data]
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def write_json(path: Path, data: Any) -> None:
    atomic_write_bytes(path, json_bytes(data))


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes().decode("utf-8"))


# ------------------------------------------------------------------------------------------------
# PNG
# ------------------------------------------------------------------------------------------------


def encode_png(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def decode_png(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FrameIntegrityError("PNG decoding failed")
    return img


def write_png(path: Path, bgr: np.ndarray) -> str:
    """Write a lossless PNG atomically and return its ``pixel_sha256``."""
    atomic_write_bytes(path, encode_png(bgr))
    return pixel_sha256(bgr)


def read_png(path: Path) -> np.ndarray:
    return decode_png(path.read_bytes())


# ------------------------------------------------------------------------------------------------
# Deterministic NPZ (np.savez embeds wall-clock zip timestamps)
# ------------------------------------------------------------------------------------------------


def encode_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in sorted(arrays):
            item = io.BytesIO()
            np.lib.format.write_array(item, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH)
            info.create_system = 3
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, item.getvalue(), compresslevel=NPZ_COMPRESSLEVEL)
    return buf.getvalue()


def decode_npz(data: bytes) -> dict[str, np.ndarray]:
    """Load every array eagerly so no file handle outlives the call (Windows locks)."""
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        return {name: np.array(npz[name]) for name in npz.files}


def read_npz(path: Path) -> dict[str, np.ndarray]:
    return decode_npz(path.read_bytes())


# ------------------------------------------------------------------------------------------------
# Evidentiary frame loader
# ------------------------------------------------------------------------------------------------

FrameResolver = Callable[[FrameReference], Path]


def assert_evidentiary_uri(uri: str, video_id: str) -> None:
    parts = uri.split("/")
    if any(part in NON_EVIDENTIARY_SEGMENTS for part in parts):
        raise NonEvidentiaryInput(f"non-evidentiary path refused as frame input: {uri}")
    if len(parts) != 5 or parts[0] != "derived" or parts[1] != video_id or parts[2] != "frames":
        raise NonEvidentiaryInput(f"frame input must be derived/{video_id}/frames/...: {uri}")
    if ".." in parts or not parts[-1].endswith(".png"):
        raise NonEvidentiaryInput(f"frame input must be a lossless PNG without traversal: {uri}")


def assert_evidentiary_input(item: object) -> FrameReference:
    """Only a source-decoded ``FrameReference`` may enter tracking or reconstruction."""
    if isinstance(item, AssetRef):
        if item.non_evidentiary or item.kind in NON_EVIDENTIARY_ASSET_KINDS:
            raise NonEvidentiaryInput(
                f"{item.kind} asset {item.asset_id} is non-evidentiary and cannot be a frame input"
            )
        raise NonEvidentiaryInput("an AssetRef is not a FrameReference; resolve the source frame")
    if not isinstance(item, FrameReference):
        raise NonEvidentiaryInput(f"{type(item).__name__} is not a FrameReference")
    if item.lossless_png_uri is None or item.pixel_sha256 is None:
        raise NonEvidentiaryInput(f"{item.frame_id} has no extracted lossless PNG")
    assert_evidentiary_uri(item.lossless_png_uri, item.video_id)
    return item


def load_frame(item: object, resolve: FrameResolver) -> np.ndarray:
    """Decode a source frame PNG and verify its geometry and ``pixel_sha256``."""
    ref = assert_evidentiary_input(item)
    img = read_png(resolve(ref))
    if img.shape != (ref.height_px, ref.width_px, 3):
        raise FrameIntegrityError(
            f"{ref.frame_id}: decoded shape {img.shape} != {ref.height_px}x{ref.width_px}"
        )
    observed = pixel_sha256(img)
    if observed != ref.pixel_sha256:
        raise FrameIntegrityError(f"{ref.frame_id}: pixel_sha256 mismatch")
    img.setflags(write=False)
    return img


def frame_png_uri(video_id: str, frame_number: int) -> str:
    """``derived/{video_id}/frames/f{n}/frame.png``; the entity folder avoids ':' for Windows."""
    return f"derived/{video_id}/frames/f{frame_number}/frame.png"


class FixtureFrameResolver:
    """Maps ``derived/{video_id}/frames/f{n}/frame.png`` to ``{root}/frames/f{n:04d}.png``."""

    def __init__(self, root: Path, video_id: str) -> None:
        self._frames = root / "frames"
        self._video_id = video_id

    def __call__(self, ref: FrameReference) -> Path:
        assert ref.lossless_png_uri is not None
        assert_evidentiary_uri(ref.lossless_png_uri, self._video_id)
        if ref.lossless_png_uri != frame_png_uri(self._video_id, ref.frame_number):
            raise NonEvidentiaryInput(f"unexpected fixture frame uri {ref.lossless_png_uri}")
        return self._frames / f"f{ref.frame_number:04d}.png"
