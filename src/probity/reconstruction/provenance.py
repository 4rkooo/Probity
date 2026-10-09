"""Per-pixel provenance arrays (npz-pixel-v1), the artifact validator, and coverage summaries.

A provenance artifact is either complete or refused. The validator never repairs; it reports every
problem and raises ``ProvenanceIncomplete`` (or ``GeneratedPixelError`` for class 2).

Lane D owns this module. The validator below (step 1) and the writer stubs at the end have FROZEN
public signatures.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import ValidationError

from probity.domain.enums import (
    AlignmentMethod,
    Interpolation,
    ProvenanceClass,
    ReasonCode,
    SourceRole,
)
from probity.domain.errors import ProbityError
from probity.domain.ids import parse_frame_id
from probity.domain.models import CoverageCounts, CoveragePct, PixelProvenance, SourceLutEntry
from probity.reconstruction.io import decode_npz, encode_npz, sha256_bytes
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


class DonorLineageError(ProbityError):
    """A donor from another video or track, the target itself, or a bridged-only observation was
    offered to the provenance writer. Preflight should have rejected it; this is a hard stop."""

    reason_code = ReasonCode.DONOR_TRACK_MISMATCH

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems), details={"problems": list(problems)})
        self.problems = tuple(problems)


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


MATRIX_VALUES = {AlignmentMethod.AKAZE_HOMOGRAPHY: 9, AlignmentMethod.ECC_AFFINE: 6}
TRANSFORM_PREFIX = {AlignmentMethod.AKAZE_HOMOGRAPHY: "h", AlignmentMethod.ECC_AFFINE: "a"}
MAX_LUT_ROWS = int(np.iinfo(np.uint16).max) + 1


def check_donor_lineage(target: Obs, donor: Obs) -> None:
    """A donor must be another detector-backed frame of the target's video and track."""
    problems: list[str] = []
    if donor.video_id != target.video_id:
        problems.append(f"donor {donor.frame_id} is from video {donor.video_id}, "
                        f"target from {target.video_id}")
    if donor.track_id != target.track_id:
        problems.append(f"donor {donor.frame_id} is on track {donor.track_id}, "
                        f"target on {target.track_id}")
    if donor.frame_number == target.frame_number:
        problems.append(f"the target frame {target.frame_id} cannot be its own donor")
    if not donor.detector_backed or donor.bridged:
        problems.append(f"donor {donor.frame_id} is not a detector-backed observation")
    if donor.frame_size != target.frame_size:
        problems.append(f"donor {donor.frame_id} frame size {donor.frame_size} != target "
                        f"{target.frame_size}")
    if problems:
        raise DonorLineageError(problems)


def _lut_matrix(donor: AlignedDonor) -> tuple[float, ...]:
    alignment = donor.alignment
    where = donor.obs.frame_id
    if not alignment.accepted:
        raise ProvenanceIncomplete([f"{where}: alignment was not accepted"])
    if alignment.method not in MATRIX_VALUES:
        raise ProvenanceIncomplete([f"{where}: {alignment.method} is not a donor alignment"])
    if alignment.matrix is None:
        raise ProvenanceIncomplete([f"{where}: accepted alignment has no matrix"])
    m = np.asarray(alignment.matrix, dtype=np.float64)
    if m.shape != (3, 3) or not np.isfinite(m).all():
        raise ProvenanceIncomplete([f"{where}: matrix must be a finite 3x3 array"])
    if alignment.method is AlignmentMethod.ECC_AFFINE and tuple(m[2]) != (0.0, 0.0, 1.0):
        raise ProvenanceIncomplete([f"{where}: ECC affine last row must be (0, 0, 1)"])
    det = float(np.linalg.det(m))
    if not math.isfinite(det) or det == 0.0:
        raise ProvenanceIncomplete([f"{where}: matrix is singular"])
    values = m if alignment.method is AlignmentMethod.AKAZE_HOMOGRAPHY else m[:2]
    return tuple(float(v) for v in values.reshape(-1))


def _donor_entry(index: int, donor: AlignedDonor, ids: Sequence[str]) -> SourceLutEntry:
    color = donor.color
    where = donor.obs.frame_id
    if not color.accepted:
        raise ProvenanceIncomplete([f"{where}: color fit was not accepted"])
    coeffs = (*color.gain, *color.bias)
    if len(color.gain) != 3 or len(color.bias) != 3 or not all(map(math.isfinite, coeffs)):
        raise ProvenanceIncomplete([f"{where}: color gain/bias must be three finite values each"])
    if not ids:
        raise ProvenanceIncomplete([f"{where}: donor has no policy decision ids"])
    if len(set(ids)) != len(ids):
        raise ProvenanceIncomplete([f"{where}: duplicate decision ids"])
    method = donor.alignment.method
    try:
        return SourceLutEntry(
            index=index, frame_id=where, frame_number=donor.obs.frame_number,
            pts_us=donor.obs.pts_us, role=SourceRole.DONOR,
            transform_id=f"{TRANSFORM_PREFIX[method]}_{donor.obs.frame_number}",
            alignment_method=method, matrix=_lut_matrix(donor),
            interpolation=Interpolation.LANCZOS4,
            color_gain=tuple(float(g) for g in color.gain),
            color_bias=tuple(float(b) for b in color.bias),
            decision_ids=tuple(ids))
    except ValidationError as exc:
        raise ProvenanceIncomplete([f"{where}: invalid LUT row: {exc}"]) from exc


def build_source_lut(target: Obs, donors: Sequence[AlignedDonor],
                     decision_ids: Mapping[int, Sequence[str]]) -> tuple[SourceLutEntry, ...]:
    """Row 0 is the TARGET (identity); row k is ``donors[k - 1]`` (already in LUT order).

    Each donor row records frame, PTS, method, matrix (9 values for a homography; the top two
    rows, 6 values, for an ECC affine), LANCZOS4, color gain/bias, and ``decision_ids`` keyed by
    donor frame number.
    """
    donors = tuple(donors)
    for d in donors:
        check_donor_lineage(target, d.obs)
    numbers = [d.obs.frame_number for d in donors]
    if len(set(numbers)) != len(numbers):
        raise ProvenanceIncomplete([f"duplicate donor frames {numbers}"])
    if list(donors) != sorted(donors, key=lambda d: (-d.R, d.obs.frame_number)):
        raise ProvenanceIncomplete([f"donors {numbers} are not in LUT order (R desc, frame asc)"])
    if len(donors) + 1 > MAX_LUT_ROWS:
        raise ProvenanceIncomplete([f"{len(donors)} donors exceed the uint16 source_index range"])
    missing = [n for n in numbers if n not in decision_ids]
    if missing:
        raise ProvenanceIncomplete([f"no decision ids for donor frames {missing}"])
    try:
        rows = [SourceLutEntry(index=0, frame_id=target.frame_id, frame_number=target.frame_number,
                               pts_us=target.pts_us, role=SourceRole.TARGET,
                               alignment_method=AlignmentMethod.IDENTITY,
                               interpolation=Interpolation.IDENTITY)]
    except ValidationError as exc:
        raise ProvenanceIncomplete([f"{target.frame_id}: invalid TARGET LUT row: {exc}"]) from exc
    for k, d in enumerate(donors, start=1):
        rows.append(_donor_entry(k, d, tuple(decision_ids[d.obs.frame_number])))
    return tuple(rows)


def _check_donor_arrays(donor: AlignedDonor, crop_shape: tuple[int, int]) -> list[str]:
    where = donor.obs.frame_id
    problems: list[str] = []
    expected = {
        "warped_crop": (donor.warped_crop, (*crop_shape, 3), np.dtype(np.uint8)),
        "valid_mask": (donor.valid_mask, crop_shape, np.dtype(np.bool_)),
        "source_x": (donor.source_x, crop_shape, np.dtype(np.float32)),
        "source_y": (donor.source_y, crop_shape, np.dtype(np.float32)),
    }
    for name, (arr, shape, dtype) in expected.items():
        if arr.shape != shape or arr.dtype != dtype:
            problems.append(f"{where}: {name} is {arr.dtype}{arr.shape}, expected {dtype}{shape}")
    return problems


def build_provenance_arrays(target: Obs, fusion: FusionResult, donors: Sequence[AlignedDonor],
                            lut: Sequence[SourceLutEntry]) -> ProvenanceArrays:
    """Full-frame arrays sized ``target.frame_size``: identity ORIGINAL everywhere, then each
    borrowed tile gets BORROWED, its LUT row, and the donor's ``source_x/source_y`` (target-crop
    frame mapped to full frame). Never writes GENERATED_BLEND. Returns arrays with ``lut`` set.
    """
    donors = tuple(donors)
    lut = tuple(lut)
    expected_lut = build_source_lut(
        target, donors, {e.frame_number: e.decision_ids for e in lut if e.role is SourceRole.DONOR})
    if lut != expected_lut:
        raise ProvenanceIncomplete(["source LUT does not match the target and donors"])
    if fusion.donor_lut_order != tuple(d.obs.frame_number for d in donors):
        raise ProvenanceIncomplete([f"fusion LUT order {fusion.donor_lut_order} != donors "
                                    f"{tuple(d.obs.frame_number for d in donors)}"])
    w, h = target.frame_size
    result = fusion.result_frame
    if result.shape != (h, w, 3) or result.dtype != np.uint8:
        raise ProvenanceIncomplete([f"result frame is {result.dtype}{result.shape}, "
                                    f"expected uint8{(h, w, 3)}"])
    ex1, ey1, ex2, ey2 = target.expanded_box
    sx1, sy1, sx2, sy2 = target.bbox_px
    crop_shape = (ey2 - ey1, ex2 - ex1)
    if not (0 <= ex1 <= sx1 < sx2 <= ex2 <= w and 0 <= ey1 <= sy1 < sy2 <= ey2 <= h):
        raise ProvenanceIncomplete([f"subject box {target.bbox_px} must lie inside expanded box "
                                    f"{target.expanded_box} inside the frame"])
    if target.expanded_crop.shape != (*crop_shape, 3):
        raise ProvenanceIncomplete(["target expanded crop does not match its expanded box"])
    problems = [p for d in donors for p in _check_donor_arrays(d, crop_shape)]
    if problems:
        raise ProvenanceIncomplete(problems)

    base = ProvenanceArrays.identity(w, h)
    cls_map = np.array(base.cls, copy=True)
    idx = np.array(base.source_index, copy=True)
    src_x = np.array(base.source_x, copy=True)
    src_y = np.array(base.source_y, copy=True)
    claimed = np.zeros((h, w), dtype=bool)
    for t in sorted(fusion.tile_decisions, key=lambda t: (t.tile_index, t.box)):
        if t.reason_code is not ReasonCode.BORROW_TILE_ACCEPTED:
            if t.source_index != 0:
                problems.append(f"tile {t.tile_index} ({t.reason_code}) keeps the original but "
                                f"names source {t.source_index}")
            continue
        x1, y1, x2, y2 = t.box
        if not 1 <= t.source_index <= len(donors):
            problems.append(f"tile {t.tile_index} borrows from LUT row {t.source_index}, which "
                            f"is not a donor")
            continue
        if not (sx1 <= x1 < x2 <= sx2 and sy1 <= y1 < y2 <= sy2):
            problems.append(f"tile {t.tile_index} box {t.box} is outside the subject box")
            continue
        if claimed[y1:y2, x1:x2].any():
            problems.append(f"tile {t.tile_index} overlaps another borrowed tile")
            continue
        claimed[y1:y2, x1:x2] = True
        d = donors[t.source_index - 1]
        r1, r2, c1, c2 = y1 - ey1, y2 - ey1, x1 - ex1, x2 - ex1
        if not d.valid_mask[r1:r2, c1:c2].all():
            problems.append(f"tile {t.tile_index} uses invalid samples of {d.obs.frame_id}")
        if not np.array_equal(result[y1:y2, x1:x2], d.warped_crop[r1:r2, c1:c2]):
            problems.append(f"tile {t.tile_index} result pixels are not the warped samples of "
                            f"{d.obs.frame_id}")
        cls_map[y1:y2, x1:x2] = ProvenanceClass.BORROWED
        idx[y1:y2, x1:x2] = t.source_index
        src_x[y1:y2, x1:x2] = d.source_x[r1:r2, c1:c2]
        src_y[y1:y2, x1:x2] = d.source_y[r1:r2, c1:c2]

    changed = (result[ey1:ey2, ex1:ex2] != target.expanded_crop).any(axis=2)
    unsupported = int((changed & ~claimed[ey1:ey2, ex1:ex2]).sum())
    if unsupported:
        problems.append(f"{unsupported} changed pixels in the expanded box are not BORROWED")
    if problems:
        raise ProvenanceIncomplete(problems)

    arrays = ProvenanceArrays(cls=cls_map, source_index=idx, source_x=src_x, source_y=src_y,
                              lut=lut)
    sizes = {0: target.frame_size} | {k: d.obs.frame_size for k, d in enumerate(donors, start=1)}
    validate_arrays(arrays, w, h, lut, sizes)
    return arrays


def provenance_record(arrays: ProvenanceArrays, *, run_id: str, subject_box: BBox, npz_uri: str,
                      npz_sha256: str, created_at: str) -> PixelProvenance:
    """Validate with ``validate_arrays`` and build the ``PixelProvenance`` record; never partial."""
    lut = tuple(arrays.lut)
    if arrays.cls.ndim != 2:
        raise ProvenanceIncomplete([f"class map must be 2-D, got shape {arrays.cls.shape}"])
    if not lut:
        raise ProvenanceIncomplete(["arrays carry no source LUT"])
    h, w = arrays.cls.shape
    try:
        videos = {parse_frame_id(e.frame_id)[0] for e in lut}
    except ValueError as exc:
        raise ProvenanceIncomplete([str(exc)]) from exc
    video_id = parse_frame_id(lut[0].frame_id)[0]
    if videos != {video_id}:
        raise DonorLineageError([f"source LUT spans videos {sorted(videos)}"])
    # Every LUT row is a frame of one video, so every source frame has the output's size.
    counts = validate_arrays(arrays, w, h, lut, {e.index: (w, h) for e in lut})

    problems: list[str] = []
    x1, y1, x2, y2 = subject_box
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        problems.append(f"subject box {subject_box} is not a non-empty box inside {w}x{h}")
    else:
        outside = np.ones((h, w), dtype=bool)
        outside[y1:y2, x1:x2] = False
        stray = int((outside & (arrays.cls == ProvenanceClass.BORROWED)).sum())
        if stray:
            problems.append(f"{stray} BORROWED pixels lie outside the subject box")
    prefix = f"derived/{video_id}/reconstruction/{run_id}/"
    if not npz_uri.startswith(prefix) or not npz_uri.endswith(".npz"):
        problems.append(f"npz uri must be {prefix}<name>.npz, got {npz_uri}")
    if sha256_bytes(encode_provenance(arrays)) != npz_sha256:
        problems.append("npz_sha256 is not the SHA-256 of these arrays' npz encoding")
    if problems:
        raise ProvenanceIncomplete(problems)

    total = w * h
    try:
        return PixelProvenance.create(
            created_at=created_at, run_id=run_id, width_px=w, height_px=h,
            subject_bbox_px=subject_box, class_map_uri=npz_uri, source_index_uri=npz_uri,
            source_xy_uri=npz_uri, source_lut=lut, coverage_counts=counts,
            coverage_pct=coverage_pct(counts, total),
            subject_coverage_pct=subject_coverage_pct(arrays, subject_box),
            coverage_complete=counts.ORIGINAL + counts.BORROWED + counts.GENERATED_BLEND == total,
            artifact_sha256=npz_sha256)
    except ValidationError as exc:
        raise ProvenanceIncomplete([f"invalid PixelProvenance: {exc}"]) from exc
