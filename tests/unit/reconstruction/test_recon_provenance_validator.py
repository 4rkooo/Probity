from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from probity.domain.enums import AlignmentMethod, Interpolation, ProvenanceClass, SourceRole
from probity.domain.ids import frame_id
from probity.domain.models import SourceLutEntry
from probity.reconstruction.determinism import seeded_uuid7
from probity.reconstruction.provenance import (
    GeneratedPixelError,
    ProvenanceArrays,
    ProvenanceIncomplete,
    assert_supported_changes,
    unsupported_changed_pixel_rate,
    validate_arrays,
)

VID = seeded_uuid7("prov:video", 1_760_000_000_000)


def _lut(n_donors: int) -> list[SourceLutEntry]:
    rows = [SourceLutEntry(index=0, frame_id=frame_id(VID, 10), frame_number=10, pts_us=1,
                           role=SourceRole.TARGET, alignment_method=AlignmentMethod.IDENTITY,
                           interpolation=Interpolation.IDENTITY)]
    for i in range(1, n_donors + 1):
        rows.append(SourceLutEntry(
            index=i, frame_id=frame_id(VID, 10 + i), frame_number=10 + i, pts_us=1 + i,
            role=SourceRole.DONOR, alignment_method=AlignmentMethod.AKAZE_HOMOGRAPHY,
            matrix=(1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            interpolation=Interpolation.LANCZOS4, color_gain=(1.0, 1.0, 1.0),
            color_bias=(0.0, 0.0, 0.0)))
    return rows


@st.composite
def valid_case(draw: st.DrawFn) -> tuple[ProvenanceArrays, int, int, list[SourceLutEntry]]:
    w = draw(st.integers(4, 14))
    h = draw(st.integers(4, 10))
    n_donors = draw(st.integers(1, 3))
    arrays = ProvenanceArrays.identity(w, h)
    for _ in range(draw(st.integers(0, 3))):
        x1 = draw(st.integers(0, w - 1))
        y1 = draw(st.integers(0, h - 1))
        x2 = draw(st.integers(x1 + 1, w))
        y2 = draw(st.integers(y1 + 1, h))
        row = draw(st.integers(1, n_donors))
        dx = draw(st.floats(-0.75, 0.75, allow_nan=False))
        arrays.cls[y1:y2, x1:x2] = ProvenanceClass.BORROWED
        arrays.source_index[y1:y2, x1:x2] = row
        xs = np.clip(arrays.source_x[y1:y2, x1:x2] * 0 + np.arange(x1, x2) + dx, 0, w - 1)
        arrays.source_x[y1:y2, x1:x2] = xs.astype(np.float32)
    return arrays, w, h, _lut(n_donors)


def _sizes(lut: list[SourceLutEntry], w: int, h: int) -> dict[int, tuple[int, int]]:
    return {e.index: (w, h) for e in lut}


@settings(max_examples=60, deadline=None)
@given(valid_case())
def test_valid_provenance_always_passes(case: tuple) -> None:
    arrays, w, h, lut = case
    counts = validate_arrays(arrays, w, h, lut, _sizes(lut, w, h))
    assert counts.ORIGINAL + counts.BORROWED == w * h
    assert counts.GENERATED_BLEND == 0


CORRUPTIONS = ("hole", "class2", "bad_index", "nan", "inf", "out_of_bounds",
               "original_not_identity", "borrowed_from_target", "shape", "dtype")


@settings(max_examples=150, deadline=None)
@given(valid_case(), st.sampled_from(CORRUPTIONS), st.data())
def test_any_corruption_is_rejected(case: tuple, kind: str, data: st.DataObject) -> None:
    arrays, w, h, lut = case
    y = data.draw(st.integers(0, h - 1))
    x = data.draw(st.integers(0, w - 1))
    borrowed = np.argwhere(arrays.cls == ProvenanceClass.BORROWED)
    if kind in {"out_of_bounds", "borrowed_from_target"} and len(borrowed):
        y, x = (int(v) for v in borrowed[data.draw(st.integers(0, len(borrowed) - 1))])
    elif kind in {"out_of_bounds", "borrowed_from_target"}:
        arrays.cls[y, x] = ProvenanceClass.BORROWED
        arrays.source_index[y, x] = 1
    if kind == "hole":
        arrays.cls[y, x] = data.draw(st.sampled_from([3, 7, 255]))
    elif kind == "class2":
        arrays.cls[y, x] = ProvenanceClass.GENERATED_BLEND
    elif kind == "bad_index":
        arrays.source_index[y, x] = len(lut) + data.draw(st.integers(0, 100))
    elif kind == "nan":
        arrays.source_x[y, x] = np.float32(math.nan)
    elif kind == "inf":
        arrays.source_y[y, x] = np.float32(math.inf)
    elif kind == "out_of_bounds":
        arrays.source_x[y, x] = np.float32(data.draw(st.sampled_from([-0.5, w - 0.5, w + 3.0])))
    elif kind == "original_not_identity":
        arrays.cls[y, x] = ProvenanceClass.ORIGINAL
        arrays.source_index[y, x] = 0
        arrays.source_x[y, x] = np.float32(x + 0.25)
    elif kind == "borrowed_from_target":
        arrays.source_index[y, x] = 0
    elif kind == "shape":
        arrays = ProvenanceArrays(arrays.cls[:-1], arrays.source_index, arrays.source_x,
                                  arrays.source_y)
    elif kind == "dtype":
        arrays = ProvenanceArrays(arrays.cls, arrays.source_index.astype(np.int32),
                                  arrays.source_x, arrays.source_y)
    expected = GeneratedPixelError if kind == "class2" else ProvenanceIncomplete
    with pytest.raises(expected):
        validate_arrays(arrays, w, h, lut, _sizes(lut, w, h))


def test_missing_lut_row_for_source_size_is_rejected() -> None:
    arrays = ProvenanceArrays.identity(6, 4)
    arrays.cls[0, 0] = ProvenanceClass.BORROWED
    arrays.source_index[0, 0] = 1
    lut = _lut(1)
    with pytest.raises(ProvenanceIncomplete):
        validate_arrays(arrays, 6, 4, lut, {0: (6, 4)})


def test_lut_without_target_first_is_rejected() -> None:
    arrays = ProvenanceArrays.identity(6, 4)
    with pytest.raises(ProvenanceIncomplete):
        validate_arrays(arrays, 6, 4, [], {0: (6, 4)})


def test_unsupported_changed_pixels_are_detected() -> None:
    target = np.zeros((4, 6, 3), np.uint8)
    result = target.copy()
    arrays = ProvenanceArrays.identity(6, 4)
    assert unsupported_changed_pixel_rate(result, target, arrays) == 0.0
    result[1, 1] = 5
    assert unsupported_changed_pixel_rate(result, target, arrays) == 1.0
    with pytest.raises(ProvenanceIncomplete):
        assert_supported_changes(result, target, arrays)
    arrays.cls[1, 1] = ProvenanceClass.BORROWED
    assert_supported_changes(result, target, arrays)
