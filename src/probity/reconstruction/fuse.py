"""Lane C: donor ranking and 8x8 winner-take-all fusion (section 9, steps 8-9). SIGNATURES FROZEN.

Keep the original tile when its normalized sharpness >= ``fusion.keep_original_sharpness`` or no
donor improves it by ``fusion.min_sharpness_improvement``. Otherwise consider only donors with full
valid coverage for the tile and residual <= ``color.max_mean_abs_residual_8bit``, and pick exactly
one by ``0.50 * improvement + 0.30 * R + 0.20 * (1 - residual / 18)``. Never average donors; never
read a previous result. Ties break by LUT order (rank desc, frame asc).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from probity.domain.enums import PolicyStage, ReasonCode
from probity.domain.errors import ValidationFailed
from probity.domain.policy import PolicyConfig
from probity.reconstruction import quality as q
from probity.reconstruction.types import (
    AlignedDonor,
    BBox,
    FusionResult,
    Gate,
    Obs,
    QualityScores,
    TileDecision,
)

LANE = "lane C"
R_WEIGHTS = {"Q": 0.45, "A": 0.35, "T": 0.20}
SELECT_WEIGHTS = {"improvement": 0.50, "rank": 0.30, "residual": 0.20}
R_TOLERANCE = 1e-9


def _unit(name: str, value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite score in [0, 1], got {value!r}")
    return value


# ------------------------------------------------------------------------------------------------
# Ranking (step 8)
# ------------------------------------------------------------------------------------------------


def temporal_preference(dt_s: float, cfg: PolicyConfig) -> float:
    """T = exp(-|dt_s| / donor.temporal_decay_s)."""
    if not math.isfinite(dt_s):
        raise ValueError(f"dt_s must be finite, got {dt_s!r}")
    return math.exp(-abs(dt_s) / cfg.donor.temporal_decay_s)


def rank_score(quality: QualityScores, A: float, T: float) -> float:
    """R = 0.45 Q + 0.35 A + 0.20 T."""
    w = R_WEIGHTS
    return w["Q"] * _unit("Q", quality.Q) + w["A"] * _unit("A", A) + w["T"] * _unit("T", T)


def _same_subject(donors: Sequence[AlignedDonor], target: Obs | None = None) -> None:
    """Donors never cross videos or tracks, never repeat a frame, and never are bridged-only."""
    ref = target if target is not None else (donors[0].obs if donors else None)
    seen: set[int] = set()
    for d in donors:
        o = d.obs
        if ref is not None and (o.video_id, o.track_id) != (ref.video_id, ref.track_id):
            raise ValidationFailed(
                f"donor {o.frame_id} is from video/track {o.video_id}/{o.track_id}, "
                f"expected {ref.video_id}/{ref.track_id}"
            )
        if target is not None and o.frame_number == target.frame_number:
            raise ValidationFailed(f"donor {o.frame_id} is the target frame")
        if o.frame_number in seen:
            raise ValidationFailed(f"donor frame {o.frame_number} appears twice")
        seen.add(o.frame_number)
        if not o.detector_backed or o.bridged:
            raise ValidationFailed(f"donor {o.frame_id} is not a detector-backed observation")
        if not d.alignment.accepted or not d.color.accepted:
            raise ValidationFailed(f"donor {o.frame_id} did not pass alignment and color gates")
        expected = rank_score(d.quality, d.alignment.A, d.T)
        if not math.isclose(d.R, expected, rel_tol=R_TOLERANCE, abs_tol=R_TOLERANCE):
            raise ValidationFailed(f"donor {o.frame_id} has R={d.R!r}, formula gives {expected!r}")


def _lut_key(d: AlignedDonor) -> tuple[float, int]:
    return (-d.R, d.obs.frame_number)


def rank_donors(donors: Sequence[AlignedDonor], cfg: PolicyConfig) -> tuple[AlignedDonor, ...]:
    """Sort by (R desc, frame_number asc), keep at most ``donor.max_count``; this is LUT order."""
    _same_subject(donors)
    return tuple(sorted(donors, key=_lut_key)[: cfg.donor.max_count])


def donor_count_gate(ranked: Sequence[AlignedDonor], cfg: PolicyConfig) -> Gate:
    """``INSUFFICIENT_COMPATIBLE_DONORS``: at least ``donor.min_count`` ranked donors, even if one
    donor would supply every chosen tile. Log it at ``PolicyStage.RANK`` on subject ``donors``."""
    return Gate(
        ReasonCode.INSUFFICIENT_COMPATIBLE_DONORS,
        PolicyStage.RANK,
        len(ranked),
        ">=",
        cfg.donor.min_count,
        "donors",
        "donor.min_count",
        "Enough compatible donors",
        "Fewer compatible donors than required",
    )


# ------------------------------------------------------------------------------------------------
# Tile selection (step 9)
# ------------------------------------------------------------------------------------------------


def relative_improvement(s_donor: float, s_target: float, cfg: PolicyConfig) -> float:
    """(S_donor - S_target) / max(S_target, fusion.sharpness_ratio_epsilon)."""
    _unit("S_donor", s_donor)
    _unit("S_target", s_target)
    return (s_donor - s_target) / max(s_target, cfg.fusion.sharpness_ratio_epsilon)


def selection_score(improvement: float, R: float, residual_8bit: float, cfg: PolicyConfig) -> float:
    """0.50 * improvement + 0.30 * R + 0.20 * (1 - residual / color.max_mean_abs_residual_8bit)."""
    if not math.isfinite(improvement) or not math.isfinite(residual_8bit) or residual_8bit < 0:
        raise ValueError("improvement and residual must be finite; residual must be >= 0")
    limit = cfg.color.max_mean_abs_residual_8bit
    if limit > 0:
        fit = 1.0 - residual_8bit / limit
    elif residual_8bit == 0:
        fit = 1.0
    else:
        raise ValueError("residual above a zero residual limit cannot be scored")
    w = SELECT_WEIGHTS
    return w["improvement"] * improvement + w["rank"] * _unit("R", R) + w["residual"] * fit


@dataclass(frozen=True, slots=True)
class TileCandidate:
    """One donor measured on one tile. ``lut_index`` is k >= 1 (LUT order); ``coverage`` is the
    fraction of tile pixels whose ``valid_mask`` is True; ``residual_8bit`` is the mean absolute
    BGR difference to the target tile, or None when coverage failed and it was not measured."""

    lut_index: int
    s_donor: float
    coverage: float
    residual_8bit: float | None
    R: float


def decide_tile(
    tile_index: int,
    box: BBox,
    s_target: float,
    candidates: Sequence[TileCandidate],
    cfg: PolicyConfig,
) -> TileDecision:
    """Winner-take-all choice for one tile; ``candidates`` must be in LUT order.

    Gates in order: keep-clear (``s_target >= keep_original_sharpness``), then per donor the
    improvement, coverage and residual gates. A donor that passes all three is scored; the strict
    maximum wins, so ties go to the earlier LUT index.
    """
    f = cfg.fusion
    _unit("S_target", s_target)
    if s_target >= f.keep_original_sharpness:
        return TileDecision(tile_index, box, 0, 0.0, ReasonCode.KEEP_ORIGINAL_CLEAR)
    if [c.lut_index for c in candidates] != sorted({c.lut_index for c in candidates}) or any(
        c.lut_index < 1 for c in candidates
    ):
        raise ValueError("tile candidates must have distinct LUT indices >= 1 in LUT order")
    best: tuple[float, TileCandidate, float, float] | None = None
    best_gain = -math.inf
    any_gain = False
    closest: float | None = None
    for c in candidates:
        imp = relative_improvement(c.s_donor, s_target, cfg)
        best_gain = max(best_gain, imp)
        if imp < f.min_sharpness_improvement:
            continue
        any_gain = True
        if c.coverage < f.required_valid_tile_coverage:
            continue
        if c.residual_8bit is None:
            raise ValueError(f"LUT {c.lut_index}: residual must be measured when coverage passes")
        res = c.residual_8bit
        closest = res if closest is None else min(closest, res)
        if res > cfg.color.max_mean_abs_residual_8bit:
            continue
        score = selection_score(imp, c.R, res, cfg)
        if best is None or score > best[0]:
            best = (score, c, imp, res)
    if best is not None:
        score, c, imp, res = best
        return TileDecision(
            tile_index, box, c.lut_index, score, ReasonCode.BORROW_TILE_ACCEPTED, imp, res
        )
    gain = best_gain if candidates else 0.0
    if any_gain:
        return TileDecision(
            tile_index, box, 0, 0.0, ReasonCode.PHOTOMETRIC_INCOMPATIBLE, gain, closest or 0.0
        )
    return TileDecision(tile_index, box, 0, 0.0, ReasonCode.NO_TILE_IMPROVED, gain)


def _check_target(target_frame: np.ndarray, target: Obs) -> None:
    w, h = target.frame_size
    if target_frame.dtype != np.uint8 or target_frame.shape != (h, w, 3):
        raise ValueError(f"target_frame must be uint8 ({h}, {w}, 3), got {target_frame.shape}")
    x1, y1, x2, y2 = target.bbox_px
    ex1, ey1, ex2, ey2 = target.expanded_box
    if not (0 <= ex1 <= x1 < x2 <= ex2 <= w and 0 <= ey1 <= y1 < y2 <= ey2 <= h):
        raise ValueError("subject box must be non-empty and inside the expanded box and frame")
    if not np.array_equal(q.crop(target_frame, target.bbox_px), target.crop) or not (
        np.array_equal(q.crop(target_frame, target.expanded_box), target.expanded_crop)
    ):
        raise ValidationFailed(f"target_frame does not hold the pixels of {target.frame_id}")


def _check_donors(target: Obs, donors: Sequence[AlignedDonor], cfg: PolicyConfig) -> None:
    if len(donors) > cfg.donor.max_count:
        raise ValidationFailed(f"{len(donors)} donors exceed donor.max_count")
    _same_subject(donors, target)
    keys = [_lut_key(d) for d in donors]
    if keys != sorted(keys):
        raise ValidationFailed("donors are not in LUT order (R desc, frame asc)")
    ex1, ey1, ex2, ey2 = target.expanded_box
    shape = (ey2 - ey1, ex2 - ex1)
    for d in donors:
        if d.warped_crop.dtype != np.uint8 or d.warped_crop.shape != (*shape, 3):
            raise ValueError(f"donor {d.obs.frame_id}: warped_crop must be uint8 {(*shape, 3)}")
        if d.valid_mask.dtype != np.bool_ or d.valid_mask.shape != shape:
            raise ValueError(f"donor {d.obs.frame_id}: valid_mask must be bool {shape}")


def fuse_tiles(
    target_frame: np.ndarray, target: Obs, donors: Sequence[AlignedDonor], cfg: PolicyConfig
) -> FusionResult:
    """``target_frame`` is the full BGR target; ``donors`` are already in LUT order.

    Tiles are ``quality.tiles(target.bbox_px, 8)``; sharpness is
    ``quality.tile_sharpness_scores`` on the subject luma.
    """
    _check_target(target_frame, target)
    _check_donors(target, donors, cfg)
    x1, y1, x2, y2 = target.bbox_px
    ex1, ey1, _, _ = target.expanded_box
    boxes = q.tiles(target.bbox_px, cfg.fusion.tile_px)
    rel = [(a - x1, b - y1, c - x1, e - y1) for a, b, c, e in boxes]
    s_target = q.tile_sharpness_scores(q.crop(q.to_luma(target_frame), target.bbox_px), rel, cfg)
    subject = (slice(y1 - ey1, y2 - ey1), slice(x1 - ex1, x2 - ex1))
    s_donor = [
        q.tile_sharpness_scores(q.to_luma(d.warped_crop)[subject], rel, cfg) for d in donors
    ]
    required = cfg.fusion.required_valid_tile_coverage
    result = target_frame.copy()
    decisions: list[TileDecision] = []
    for i, box in enumerate(boxes):
        bx1, by1, bx2, by2 = box
        in_crop = (slice(by1 - ey1, by2 - ey1), slice(bx1 - ex1, bx2 - ex1))
        original = target_frame[by1:by2, bx1:bx2].astype(np.float64)
        candidates = []
        for k, d in enumerate(donors, start=1):
            coverage = float(d.valid_mask[in_crop].mean())
            residual = None
            if coverage >= required:
                residual = float(np.abs(d.warped_crop[in_crop].astype(np.float64) - original).mean())
            candidates.append(TileCandidate(k, s_donor[k - 1][i], coverage, residual, d.R))
        decision = decide_tile(i, box, s_target[i], candidates, cfg)
        if decision.source_index:
            result[by1:by2, bx1:bx2] = donors[decision.source_index - 1].warped_crop[in_crop]
        decisions.append(decision)
    return FusionResult(result, tuple(decisions), tuple(d.obs.frame_number for d in donors))


def tile_gates(result: FusionResult, cfg: PolicyConfig) -> tuple[tuple[str, Gate], ...]:
    """``(subject_ref, gate)`` rows to log: one ``BORROW_TILE_ACCEPTED`` per borrowed tile and one
    aggregate ``KEEP_ORIGINAL_CLEAR`` row with counts per keep reason."""
    f = cfg.fusion
    rows: list[tuple[str, Gate]] = []
    kept_clear = kept_no_gain = kept_residual = 0
    for td in result.tile_decisions:
        if td.reason_code is ReasonCode.BORROW_TILE_ACCEPTED:
            if td.source_index < 1 or td.source_index > len(result.donor_lut_order):
                raise ValidationFailed(
                    f"tile {td.tile_index} is borrowed but source_index={td.source_index} "
                    f"is outside LUT 1..{len(result.donor_lut_order)}"
                )
            frame_n = result.donor_lut_order[td.source_index - 1]
            x1, y1, _, _ = td.box
            rows.append(
                (
                    f"tile:{x1},{y1}",
                    Gate(
                        ReasonCode.BORROW_TILE_ACCEPTED,
                        PolicyStage.FUSE,
                        td.improvement,
                        ">=",
                        f.min_sharpness_improvement,
                        "ratio",
                        "fusion.min_sharpness_improvement",
                        f"Borrowed from f{frame_n}; residual {td.residual_8bit:.2f}/255",
                        "Sharpness improvement below minimum",
                    ),
                )
            )
            continue
        if td.source_index != 0:
            raise ValidationFailed(
                f"tile {td.tile_index} reason {td.reason_code} has source_index={td.source_index}"
            )
        if td.reason_code is ReasonCode.KEEP_ORIGINAL_CLEAR:
            kept_clear += 1
        elif td.reason_code is ReasonCode.NO_TILE_IMPROVED:
            kept_no_gain += 1
        elif td.reason_code in (
            ReasonCode.PHOTOMETRIC_INCOMPATIBLE,
            ReasonCode.REVERT_TILE_VALIDATION,
        ):
            kept_residual += 1
        else:
            raise ValidationFailed(f"tile {td.tile_index} has unexpected reason {td.reason_code}")
    n_kept = kept_clear + kept_no_gain + kept_residual
    rows.append(
        (
            "tiles",
            Gate(
                ReasonCode.KEEP_ORIGINAL_CLEAR,
                PolicyStage.FUSE,
                n_kept,
                ">=",
                0,
                "tiles",
                "fusion.keep_original_sharpness",
                (
                    f"Kept original: {kept_clear} already clear, {kept_no_gain} not improved, "
                    f"{kept_residual} failed residual/coverage"
                ),
                "No original tiles kept",
            ),
        )
    )
    return tuple(rows)
