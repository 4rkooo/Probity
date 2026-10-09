"""Lane E: evaluation metrics (section 12). Ground-truth metrics never enter reconstruction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from probity.domain.enums import (
    AlignmentMethod,
    Interpolation,
    PolicyOutcome,
    PolicyStage,
    ProvenanceClass,
    ReasonCode,
    SourceRole,
)
from probity.domain.ids import frame_id
from probity.domain.models import SourceLutEntry
from probity.eval.metrics import (
    alignment_rejection_rate,
    evaluation_row,
    linear_luma,
    ocr_character_accuracy,
    provenance_percentages,
    psnr,
    ssim,
    supported_changed_pixel_rate,
)
from probity.reconstruction.decisions import DecisionLog
from probity.reconstruction.determinism import FixedClock, seeded_uuid7
from probity.reconstruction.types import ProvenanceArrays

VID = seeded_uuid7("lane-e:metrics-video", 1_760_000_000_000)
RUN = seeded_uuid7("lane-e:metrics-run", 1_760_000_000_000)
CLOCK = FixedClock("2026-10-09T17:00:00.000000Z")


def _gray(value: int, size: int = 16) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _lut(*donor_indices: int) -> tuple[SourceLutEntry, ...]:
    rows = [
        SourceLutEntry(
            index=0, frame_id=frame_id(VID, 10), frame_number=10, pts_us=0,
            role=SourceRole.TARGET, alignment_method=AlignmentMethod.IDENTITY, matrix=None,
            interpolation=Interpolation.IDENTITY,
        )
    ]
    for index in donor_indices:
        rows.append(SourceLutEntry(
            index=index, frame_id=frame_id(VID, 10 + index), frame_number=10 + index,
            pts_us=index * 66_666, role=SourceRole.DONOR, transform_id=f"h_{index}",
            alignment_method=AlignmentMethod.AKAZE_HOMOGRAPHY,
            matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            interpolation=Interpolation.LANCZOS4,
            decision_ids=(seeded_uuid7(f"lane-e:lut:{index}", 1_760_000_000_000),),
        ))
    return tuple(rows)


def _arrays(cls: np.ndarray, index: np.ndarray | None = None, *,
            lut: tuple[SourceLutEntry, ...] = ()) -> ProvenanceArrays:
    h, w = cls.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    if index is None:
        index = np.zeros((h, w), dtype=np.uint16)
    return ProvenanceArrays(cls=cls, source_index=index, source_x=xs, source_y=ys, lut=lut)


def _align_log() -> DecisionLog:
    return DecisionLog(RUN, CLOCK)


def _align(log: DecisionLog, subject: str, outcome: PolicyOutcome, code: ReasonCode) -> None:
    log.add(code, PolicyStage.ALIGN, subject, outcome, "test alignment gate",
            observed=1.0 if outcome is PolicyOutcome.ACCEPT else 0.0,
            operator=">=", threshold=0.65, units="fraction",
            policy_key="alignment.min_inlier_ratio")


# ------------------------------------------------------------------------------------------------
# Linear luma / PSNR / SSIM
# ------------------------------------------------------------------------------------------------


def test_linear_luma_is_rec709_after_srgb_decode() -> None:
    black = linear_luma(_gray(0))
    white = linear_luma(_gray(255))
    mid = linear_luma(_gray(128))
    assert black.shape == (16, 16) and black.dtype == np.float64
    assert np.allclose(black, 0.0) and np.allclose(white, 1.0)
    assert 0.0 < float(mid.mean()) < 0.5


def test_psnr_identical_is_infinite_and_black_vs_white_is_zero() -> None:
    a = _gray(32)
    assert psnr(a, a) == math.inf
    assert psnr(_gray(0), _gray(255)) == 0.0


def test_psnr_mask_inside_and_outside_the_registered_region() -> None:
    truth = _gray(40)
    result = truth.copy()
    result[0, 0] = (200, 200, 200)
    inside = np.zeros(truth.shape[:2], dtype=bool)
    inside[1:, 1:] = True
    outside = np.zeros_like(inside)
    outside[0, 0] = True
    assert psnr(result, truth, mask=inside) == math.inf
    assert psnr(result, truth, mask=outside) < math.inf
    with pytest.raises(ValueError, match="no registered"):
        psnr(result, truth, mask=np.zeros(truth.shape[:2], dtype=bool))


def test_ssim_identical_is_one_and_drops_when_pixels_change() -> None:
    a = _gray(80, size=24)
    b = a.copy()
    b[8:16, 8:16] = 200
    assert ssim(a, a) == pytest.approx(1.0, abs=1e-12)
    assert 0.0 < ssim(a, b) < 1.0


def test_ssim_rejects_images_smaller_than_the_window() -> None:
    tiny = _gray(10, size=10)
    with pytest.raises(ValueError, match="11x11"):
        ssim(tiny, tiny)


# ------------------------------------------------------------------------------------------------
# Provenance percentages and supported changed-pixel rate
# ------------------------------------------------------------------------------------------------


def test_provenance_percentages_are_exact_quarters() -> None:
    cls = np.zeros((2, 2), dtype=np.uint8)
    cls[0, 1] = ProvenanceClass.BORROWED
    cls[1, 0] = ProvenanceClass.GENERATED_BLEND
    pct = provenance_percentages(_arrays(cls))
    assert pct == {"ORIGINAL": 50.0, "BORROWED": 25.0, "GENERATED_BLEND": 25.0}


def test_provenance_percentages_subject_box_inside_vs_outside() -> None:
    cls = np.zeros((4, 4), dtype=np.uint8)
    cls[1, 1] = ProvenanceClass.BORROWED
    arrays = _arrays(cls)
    inside = provenance_percentages(arrays, (1, 1, 2, 2))
    assert inside == {"ORIGINAL": 0.0, "BORROWED": 100.0, "GENERATED_BLEND": 0.0}
    with pytest.raises(ValueError, match="outside"):
        provenance_percentages(arrays, (0, 0, 5, 1))


def test_supported_changed_pixel_rate_is_one_when_nothing_changed() -> None:
    img = _gray(7)
    cls = np.zeros((16, 16), dtype=np.uint8)
    assert supported_changed_pixel_rate(img, img, _arrays(cls, lut=_lut(1))) == 1.0


def test_supported_changed_pixel_just_inside_borrowed_donor_lut() -> None:
    target = _gray(7)
    result = target.copy()
    result[0, 0] = (9, 9, 9)
    cls = np.zeros((16, 16), dtype=np.uint8)
    cls[0, 0] = ProvenanceClass.BORROWED
    index = np.zeros((16, 16), dtype=np.uint16)
    index[0, 0] = 1
    assert supported_changed_pixel_rate(result, target, _arrays(cls, index, lut=_lut(1))) == 1.0


def test_supported_changed_pixel_just_outside_missing_donor_lut() -> None:
    target = _gray(7)
    result = target.copy()
    result[0, 0] = (9, 9, 9)
    cls = np.zeros((16, 16), dtype=np.uint8)
    cls[0, 0] = ProvenanceClass.BORROWED
    index = np.zeros((16, 16), dtype=np.uint16)
    index[0, 0] = 1
    assert supported_changed_pixel_rate(result, target, _arrays(cls, index, lut=_lut())) == 0.0


def test_changed_original_pixel_is_unsupported() -> None:
    target = _gray(7)
    result = target.copy()
    result[0, 0] = (9, 9, 9)
    cls = np.zeros((16, 16), dtype=np.uint8)
    assert supported_changed_pixel_rate(result, target, _arrays(cls, lut=_lut(1))) == 0.0


def test_golden_completed_supported_rate_is_one(golden_completed, translate_images) -> None:
    g = golden_completed
    assert g.result is not None and g.arrays is not None
    target_n = int(g.run.target_frame_id.rsplit("f", 1)[-1])
    assert supported_changed_pixel_rate(g.result, translate_images[target_n], g.arrays) == 1.0
    assert provenance_percentages(g.arrays) == {
        "ORIGINAL": g.provenance.coverage_pct.ORIGINAL,
        "BORROWED": g.provenance.coverage_pct.BORROWED,
        "GENERATED_BLEND": g.provenance.coverage_pct.GENERATED_BLEND,
    }


# ------------------------------------------------------------------------------------------------
# OCR (evaluation strings only)
# ------------------------------------------------------------------------------------------------


def test_ocr_character_accuracy_exact_match_is_one() -> None:
    assert ocr_character_accuracy("PRB 4K7", "PRB 4K7") == 1.0


def test_ocr_one_position_wrong_is_exact_fraction() -> None:
    # 7-character plate; one substitution is 6/7, just outside a perfect match.
    assert ocr_character_accuracy("PRB 4K8", "PRB 4K7") == pytest.approx(6 / 7)
    assert ocr_character_accuracy("PRB 4K7", "PRB 4K7") == 1.0


def test_ocr_extra_characters_do_not_change_the_denominator() -> None:
    assert ocr_character_accuracy("PRB 4K7X", "PRB 4K7") == 1.0
    assert ocr_character_accuracy("PRB", "PRB 4K7") == pytest.approx(3 / 7)


def test_ocr_rejects_empty_truth() -> None:
    with pytest.raises(ValueError, match="empty truth"):
        ocr_character_accuracy("A", "")


# ------------------------------------------------------------------------------------------------
# Alignment rejection rate
# ------------------------------------------------------------------------------------------------


def test_alignment_rejection_rate_zero_when_every_considered_donor_is_accepted(
        golden_completed) -> None:
    rates = alignment_rejection_rate(golden_completed.decisions)
    assert rates["total"] == 0.0
    assert ReasonCode.ALIGNMENT_FAILED.value not in rates


def test_alignment_rejection_just_inside_all_accepted() -> None:
    log = _align_log()
    _align(log, "donor-a", PolicyOutcome.ACCEPT, ReasonCode.ALIGNMENT_AKAZE_ACCEPTED)
    _align(log, "donor-b", PolicyOutcome.ACCEPT, ReasonCode.ALIGNMENT_ECC_FALLBACK_ACCEPTED)
    assert alignment_rejection_rate(log.rows) == {"total": 0.0}


def test_alignment_rejection_just_outside_one_failed_of_two() -> None:
    log = _align_log()
    _align(log, "donor-a", PolicyOutcome.ACCEPT, ReasonCode.ALIGNMENT_AKAZE_ACCEPTED)
    _align(log, "donor-b", PolicyOutcome.REJECT, ReasonCode.ALIGNMENT_FAILED)
    rates = alignment_rejection_rate(log.rows)
    assert rates == {"total": 0.5, "ALIGNMENT_FAILED": 0.5}


def test_alignment_rejection_ignores_preflight_and_splits_by_reason() -> None:
    log = _align_log()
    log.add(ReasonCode.DONOR_OBSTRUCTED, PolicyStage.PREFLIGHT, "donor-x", PolicyOutcome.REJECT,
            "obstructed", observed=0.3, operator="<=", threshold=0.15, units="fraction")
    _align(log, "donor-a", PolicyOutcome.REJECT, ReasonCode.ALIGNMENT_FAILED)
    _align(log, "donor-b", PolicyOutcome.REJECT, ReasonCode.ALIGNMENT_FAILED)
    _align(log, "donor-c", PolicyOutcome.ACCEPT, ReasonCode.ALIGNMENT_AKAZE_ACCEPTED)
    rates = alignment_rejection_rate(log.rows)
    assert rates["total"] == pytest.approx(2 / 3)
    assert rates["ALIGNMENT_FAILED"] == pytest.approx(2 / 3)
    assert "DONOR_OBSTRUCTED" not in rates


def test_alignment_rejection_empty_log_is_zero() -> None:
    assert alignment_rejection_rate(()) == {"total": 0.0}


# ------------------------------------------------------------------------------------------------
# Evaluation row
# ------------------------------------------------------------------------------------------------


def test_evaluation_row_sorts_metrics_and_rejects_non_finite() -> None:
    cid = seeded_uuid7("lane-e:corr", 1_760_000_000_000)
    row = evaluation_row("trueframe", cid, {"psnr_db": 31.5, "ssim": 0.9})
    assert [m.name for m in row.metrics] == ["psnr_db", "ssim"]
    assert row.correlation_id == cid
    with pytest.raises(ValueError, match="not finite"):
        evaluation_row("trueframe", cid, {"psnr_db": math.inf})
    with pytest.raises(ValueError, match="UUIDv7"):
        evaluation_row("trueframe", "not-a-uuid", {"ssim": 0.9})
