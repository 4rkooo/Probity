from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from probity.domain.enums import AssetKind
from probity.domain.ids import frame_id
from probity.domain.models import AssetRef, FrameReference
from probity.reconstruction.determinism import seeded_uuid7
from probity.reconstruction.io import (
    FixtureFrameResolver,
    FrameIntegrityError,
    NonEvidentiaryInput,
    array_sha256,
    assert_evidentiary_input,
    assert_evidentiary_uri,
    decode_npz,
    encode_npz,
    encode_png,
    frame_png_uri,
    json_bytes,
    load_frame,
    pixel_sha256,
    write_png,
)

VID = seeded_uuid7("test:video", 1_760_000_000_000)
CASE = seeded_uuid7("test:case", 1_760_000_000_000)


def _image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)


def _frame_ref(n: int, digest: str, uri: str | None = None) -> FrameReference:
    return FrameReference.create(
        frame_id=frame_id(VID, n), video_id=VID, frame_number=n, pts_us=n * 66_666,
        is_keyframe=True, width_px=16, height_px=12,
        lossless_png_uri=uri or frame_png_uri(VID, n), pixel_sha256=digest,
    )


def _asset(kind: AssetKind, non_evidentiary: bool) -> AssetRef:
    return AssetRef(
        asset_id=seeded_uuid7(f"asset:{kind}", 1_760_000_000_000), case_id=CASE, kind=kind,
        storage_uri=f"derived/{VID}/baseline/x/out.png", media_type="image/png", byte_length=1,
        sha256="0" * 64, non_evidentiary=non_evidentiary, created_at="2026-10-09T17:00:00Z",
    )


def test_pixel_sha256_is_channel_order_sensitive_and_encoding_independent() -> None:
    img = _image()
    assert pixel_sha256(img) == pixel_sha256(img.copy())
    assert pixel_sha256(img) != pixel_sha256(np.ascontiguousarray(img[:, :, ::-1]))
    from probity.reconstruction.io import decode_png

    assert pixel_sha256(decode_png(encode_png(img))) == pixel_sha256(img)


def test_npz_encoding_is_byte_deterministic_and_round_trips() -> None:
    arrays = {"b": np.arange(6, dtype=np.uint16).reshape(2, 3), "a": np.ones((2, 3), np.float32)}
    first, second = encode_npz(arrays), encode_npz(dict(reversed(list(arrays.items()))))
    assert first == second
    back = decode_npz(first)
    assert array_sha256(back) == array_sha256(arrays)


def test_json_bytes_use_lf_only() -> None:
    data = json_bytes({"k": ["a", "b"]})
    assert b"\r\n" not in data and data.endswith(b"\n")


def test_loader_reads_and_verifies_source_frame(tmp_path: Path) -> None:
    img = _image()
    digest = write_png(tmp_path / "frames" / "f0003.png", img)
    ref = _frame_ref(3, digest)
    out = load_frame(ref, FixtureFrameResolver(tmp_path, VID))
    assert np.array_equal(out, img)
    assert not out.flags.writeable


def test_loader_detects_tampered_pixels(tmp_path: Path) -> None:
    digest = write_png(tmp_path / "frames" / "f0003.png", _image(0))
    write_png(tmp_path / "frames" / "f0003.png", _image(1))
    with pytest.raises(FrameIntegrityError):
        load_frame(_frame_ref(3, digest), FixtureFrameResolver(tmp_path, VID))


@pytest.mark.parametrize("kind", [AssetKind.BASELINE, AssetKind.RESULT_PNG,
                                  AssetKind.INSPECTION_CLIP, AssetKind.THUMBNAIL])
def test_loader_rejects_generated_or_derived_assets(kind: AssetKind) -> None:
    with pytest.raises(NonEvidentiaryInput):
        assert_evidentiary_input(_asset(kind, non_evidentiary=True))


def test_loader_rejects_non_frame_objects() -> None:
    with pytest.raises(NonEvidentiaryInput):
        assert_evidentiary_input(f"derived/{VID}/frames/f1/frame.png")
    with pytest.raises(NonEvidentiaryInput):
        assert_evidentiary_input(_image())


@pytest.mark.parametrize("uri", [
    f"derived/{VID}/baseline/f1/frame.png",
    f"derived/{VID}/reconstruction/run/result.png",
    f"derived/{VID}/frames/f1/../../baseline/x.png",
    f"derived/{VID}/frames/f1/frame.jpg",
    "fixtures/synthetic/plate_translate_v1/ground_truth/clean_target.png",
])
def test_evidentiary_uri_rejects_non_source_paths(uri: str) -> None:
    with pytest.raises(NonEvidentiaryInput):
        assert_evidentiary_uri(uri, VID)


def test_frame_reference_cannot_point_at_baseline_or_ground_truth() -> None:
    for uri in (f"derived/{VID}/baseline/f1/frame.png",
                "fixtures/synthetic/x/ground_truth/clean_target.png"):
        with pytest.raises(ValidationError):
            _frame_ref(1, "0" * 64, uri=uri)


def test_fixture_resolver_refuses_unexpected_uri(tmp_path: Path) -> None:
    ref = _frame_ref(3, "0" * 64, uri=f"derived/{VID}/frames/other/frame.png")
    with pytest.raises(NonEvidentiaryInput):
        FixtureFrameResolver(tmp_path, VID)(ref)
