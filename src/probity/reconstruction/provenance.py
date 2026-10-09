"""Per-pixel provenance arrays (npz-pixel-v1), the artifact validator, and coverage summaries.

A provenance artifact is either complete or refused. The validator never repairs; it reports every
problem and raises ``ProvenanceIncomplete`` (or ``GeneratedPixelError`` for class 2).

Lane D owns this module. The validator below (step 1) and the writer stubs at the end have FROZEN
public signatures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from probity.domain.enums import ProvenanceClass, ReasonCode, SourceRole
from probity.domain.errors import ProbityError
from probity.domain.models import CoverageCounts, CoveragePct, PixelProvenance, SourceLutEntry
from probity.reconstruction.io import decode_npz, encode_npz
from probity.reconstruction.types import AlignedDonor, FusionResult, Obs, ProvenanceArrays

BBox = tuple[int, int, int, int]
LANE = "lane D"

ENCODING_VERSION = "npz-pixel-v1"
ARRAY_DTYPES: Mapping[str, np.dtype] = {
    "class": np.dtype(np.uint8),
    "source_index": np.dtype(np.uint16),
    "source_x": np.dtype(np.float32),
    "source_y": np.dtype(np.float32),
}
VALID_CLASSES = (int(ProvenanceClass.ORIGINAL), int(ProvenanceClass.BORROWED))


class ProvenanceIncomplete(ProbityError):
    reason_code = ReasonCode.PROVENANCE_INCOMPLETE

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems), details={"problems": list(problems)})
        self.problems = tuple(problems)


class GeneratedPixelError(ProvenanceIncomplete):
    reason_code = ReasonCode.GENERATED_SEMANTIC_PIXEL


def arrays_from_dict(arrays: Mapping[str, np.ndarray]) -> ProvenanceArrays:
    missing = set(ARRAY_DTYPES) - set(arrays)
    extra = set(arrays) - set(ARRAY_DTYPES)
    if missing or extra:
        raise ProvenanceIncomplete(
            [f"npz arrays must be exactly {sorted(ARRAY_DTYPES)}; "
             f"missing={sorted(missing)} extra={sorted(extra)}"]
        )
    return ProvenanceArrays(
        cls=arrays["class"],
        source_index=arrays["source_index"],
        source_x=arrays["source_x"],
        source_y=arrays["source_y"],
    )


def encode_provenance(arrays: ProvenanceArrays) -> bytes:
    return encode_npz(arrays.as_dict())


def decode_provenance(data: bytes) -> ProvenanceArrays:
    return arrays_from_dict(decode_npz(data))


def validate_arrays(
    arrays: ProvenanceArrays,
    width: int,
    height: int,
    lut: Sequence[SourceLutEntry],
    source_sizes: Mapping[int, tuple[int, int]],
) -> CoverageCounts:
    """Check dimensions, dtypes, enum range, LUT resolution, coordinates, and exact pixel count.

    ``source_sizes`` maps LUT index to that source frame's ``(width, height)``.
    """
    problems: list[str] = []
    named = arrays.as_dict()
    for name, dtype in ARRAY_DTYPES.items():
        arr = named[name]
        if arr.shape != (height, width):
            problems.append(f"{name} shape {arr.shape} != {(height, width)}")
        if arr.dtype != dtype:
            problems.append(f"{name} dtype {arr.dtype} != {dtype}")
    if problems:
        raise ProvenanceIncomplete(problems)

    cls_map = arrays.cls
    generated = int((cls_map == ProvenanceClass.GENERATED_BLEND).sum())
    if generated:
        raise GeneratedPixelError([f"{generated} GENERATED_BLEND pixels present"])

    known = np.isin(cls_map, VALID_CLASSES)
    if not known.all():
        problems.append(f"{int((~known).sum())} pixels have no valid provenance class")

    if not lut or lut[0].index != 0 or lut[0].role is not SourceRole.TARGET:
        problems.append("source_lut[0] must be the TARGET frame")
    if [e.index for e in lut] != list(range(len(lut))):
        problems.append("source_lut indices must be contiguous from 0")
    missing_sizes = [e.index for e in lut if e.index not in source_sizes]
    if missing_sizes:
        problems.append(f"no source frame size for LUT rows {missing_sizes}")

    idx = arrays.source_index
    unresolved = idx >= len(lut)
    if unresolved.any():
        problems.append(f"{int(unresolved.sum())} pixels reference a source_index with no LUT row")

    original = cls_map == ProvenanceClass.ORIGINAL
    borrowed = cls_map == ProvenanceClass.BORROWED
    if (idx[original] != 0).any():
        problems.append("ORIGINAL pixels must reference source_index 0 (the target)")
    donor_rows = np.array([e.index for e in lut if e.role is SourceRole.DONOR], dtype=np.int64)
    if borrowed.any() and not np.isin(idx[borrowed].astype(np.int64), donor_rows).all():
        problems.append("BORROWED pixels must reference a DONOR LUT row")

    sx, sy = arrays.source_x, arrays.source_y
    covered = original | borrowed
    finite = np.isfinite(sx) & np.isfinite(sy)
    if not finite[covered].all():
        bad = int((covered & ~finite).sum())
        problems.append(f"{bad} covered pixels have non-finite coordinates")

    ys, xs = np.indices((height, width), dtype=np.float32)
    if not ((sx[original] == xs[original]) & (sy[original] == ys[original])).all():
        problems.append("ORIGINAL pixels must carry identity target coordinates")

    if borrowed.any() and not unresolved.any() and not missing_sizes:
        for row in np.unique(idx[borrowed]):
            sel = borrowed & (idx == row)
            w, h = source_sizes[int(row)]
            bx, by = sx[sel], sy[sel]
            inside = (bx >= 0) & (bx <= w - 1) & (by >= 0) & (by <= h - 1)
            if not inside.all():
                problems.append(
                    f"{int((~inside).sum())} pixels from LUT row {int(row)} fall outside its frame"
                )

    counts = CoverageCounts(
        ORIGINAL=int(original.sum()),
        BORROWED=int(borrowed.sum()),
        GENERATED_BLEND=generated,
    )
    if counts.ORIGINAL + counts.BORROWED + counts.GENERATED_BLEND != width * height:
        problems.append("coverage counts do not equal width*height")
    if problems:
        raise ProvenanceIncomplete(problems)
    return counts


def changed_pixel_mask(result: np.ndarray, target: np.ndarray) -> np.ndarray:
    if result.shape != target.shape:
        raise ProvenanceIncomplete(["result and target shapes differ"])
    return np.asarray((result != target).any(axis=2))


def unsupported_changed_pixel_rate(
    result: np.ndarray, target: np.ndarray, arrays: ProvenanceArrays
) -> float:
    """Changed pixels not labelled BORROWED / all changed pixels (0.0 when nothing changed)."""
    changed = changed_pixel_mask(result, target)
    total = int(changed.sum())
    if total == 0:
        return 0.0
    unsupported = changed & (arrays.cls != ProvenanceClass.BORROWED)
    return int(unsupported.sum()) / total


def assert_supported_changes(
    result: np.ndarray, target: np.ndarray, arrays: ProvenanceArrays
) -> None:
    rate = unsupported_changed_pixel_rate(result, target, arrays)
    if rate != 0.0:
        raise ProvenanceIncomplete([f"unsupported changed-pixel rate {rate:.6f} != 0"])


def coverage_pct(counts: CoverageCounts, total: int) -> CoveragePct:
    return CoveragePct(
        ORIGINAL=round(100.0 * counts.ORIGINAL / total, 4),
        BORROWED=round(100.0 * counts.BORROWED / total, 4),
        GENERATED_BLEND=round(100.0 * counts.GENERATED_BLEND / total, 4),
    )


def subject_coverage_pct(arrays: ProvenanceArrays, bbox: BBox) -> CoveragePct:
    x1, y1, x2, y2 = bbox
    sub = arrays.cls[y1:y2, x1:x2]
    counts = CoverageCounts(
        ORIGINAL=int((sub == ProvenanceClass.ORIGINAL).sum()),
        BORROWED=int((sub == ProvenanceClass.BORROWED).sum()),
        GENERATED_BLEND=int((sub == ProvenanceClass.GENERATED_BLEND).sum()),
    )
    return coverage_pct(counts, sub.size)


# ------------------------------------------------------------------------------------------------
# Writer (lane D)
# ------------------------------------------------------------------------------------------------


def build_source_lut(target: Obs, donors: Sequence[AlignedDonor],
                     decision_ids: Mapping[int, Sequence[str]]) -> tuple[SourceLutEntry, ...]:
    """Row 0 is the TARGET (identity); row k is ``donors[k - 1]`` (already in LUT order).

    Each donor row records frame, PTS, method, matrix (9 values for a homography; the top two
    rows, 6 values, for an ECC affine), LANCZOS4, color gain/bias, and ``decision_ids`` keyed by
    donor frame number.
    """
    raise NotImplementedError(LANE)


def build_provenance_arrays(target: Obs, fusion: FusionResult, donors: Sequence[AlignedDonor],
                            lut: Sequence[SourceLutEntry]) -> ProvenanceArrays:
    """Full-frame arrays sized ``target.frame_size``: identity ORIGINAL everywhere, then each
    borrowed tile gets BORROWED, its LUT row, and the donor's ``source_x/source_y`` (target-crop
    frame mapped to full frame). Never writes GENERATED_BLEND. Returns arrays with ``lut`` set.
    """
    raise NotImplementedError(LANE)


def provenance_record(arrays: ProvenanceArrays, *, run_id: str, subject_box: BBox, npz_uri: str,
                      npz_sha256: str, created_at: str) -> PixelProvenance:
    """Validate with ``validate_arrays`` and build the ``PixelProvenance`` record; never partial."""
    raise NotImplementedError(LANE)
