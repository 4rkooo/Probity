"""Integrator: revert-only validation of borrowed tiles (section 9, algorithm step 11).

After fusion, recompute photometric residual and a 2-px boundary discontinuity against the
unchanged target. Revert any failing tile to original. A second pass may only remove borrowing,
never add it and never use borrowed pixels as neighbor context. Stop when fewer than
``iteration.convergence_changed_fraction`` of subject pixels change, or after
``iteration.max_passes`` total passes (pass 1 is fuse + validate; pass 2 is revert-only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from probity.domain.enums import ReasonCode
from probity.domain.policy import PolicyConfig
from probity.reconstruction.types import BBox, FusionResult, TileDecision

LANE = "integrator"
BAND_PX = 2


@dataclass(frozen=True, slots=True)
class ValidationPass:
    """Outcome of one revert-only pass. ``changed_fraction`` is over the subject box."""

    fusion: FusionResult
    n_reverted: int
    changed_fraction: float
    reverted_tiles: tuple[TileDecision, ...]


def borrowed_mask(fusion: FusionResult, shape: tuple[int, int]) -> np.ndarray:
    """True on pixels currently borrowed (full-frame bool)."""
    mask = np.zeros(shape, dtype=bool)
    for td in fusion.tile_decisions:
        if td.reason_code is ReasonCode.BORROW_TILE_ACCEPTED and td.source_index >= 1:
            x1, y1, x2, y2 = td.box
            mask[y1:y2, x1:x2] = True
    return mask


def tile_residual_8bit(result: np.ndarray, target: np.ndarray, box: BBox) -> float:
    """Mean absolute BGR difference in 8-bit units over the tile vs the unchanged target."""
    x1, y1, x2, y2 = box
    return float(
        np.abs(result[y1:y2, x1:x2].astype(np.float64) - target[y1:y2, x1:x2].astype(np.float64))
        .mean()
    )


def boundary_discontinuity_8bit(
    result: np.ndarray,
    target: np.ndarray,
    box: BBox,
    original_context: np.ndarray,
    *,
    band_px: int = BAND_PX,
) -> float | None:
    """Mean abs BGR difference between the tile's interior rim and adjacent ORIGINAL neighbors.

    ``original_context`` is True where a pixel may be used as a neighbor (not currently borrowed).
    Returns None when no original neighbor exists; that tile then fails closed on discontinuity
    only if residual also fails, and is not reverted for discontinuity alone.
    """
    x1, y1, x2, y2 = box
    height, width = target.shape[:2]
    diffs: list[np.ndarray] = []

    def _pair(inside: tuple[slice, slice], outside: tuple[slice, slice]) -> None:
        oy, ox = outside
        ctx = original_context[oy, ox]
        if not np.any(ctx):
            return
        inner = result[inside].astype(np.float64)
        outer = target[outside].astype(np.float64)
        if inner.shape != outer.shape:
            return
        sel = np.broadcast_to(ctx[..., None], inner.shape)
        diffs.append(np.abs(inner - outer)[sel].reshape(-1, 3))

    for offset in range(band_px):
        iy, oy = y1 + offset, y1 - 1
        if 0 <= oy < height and iy < y2:
            _pair((slice(iy, iy + 1), slice(x1, x2)), (slice(oy, oy + 1), slice(x1, x2)))
        iy, oy = y2 - 1 - offset, y2
        if 0 <= oy < height and iy >= y1:
            _pair((slice(iy, iy + 1), slice(x1, x2)), (slice(oy, oy + 1), slice(x1, x2)))
        ix, ox = x1 + offset, x1 - 1
        if 0 <= ox < width and ix < x2:
            _pair((slice(y1, y2), slice(ix, ix + 1)), (slice(y1, y2), slice(ox, ox + 1)))
        ix, ox = x2 - 1 - offset, x2
        if 0 <= ox < width and ix >= x1:
            _pair((slice(y1, y2), slice(ix, ix + 1)), (slice(y1, y2), slice(ox, ox + 1)))

    if not diffs:
        return None
    stacked = np.concatenate(diffs, axis=0)
    if stacked.size == 0:
        return None
    return float(stacked.mean())


def _should_revert(
    residual: float,
    discontinuity: float | None,
    cfg: PolicyConfig,
) -> bool:
    if residual > cfg.color.max_mean_abs_residual_8bit:
        return True
    if discontinuity is not None and discontinuity > cfg.validation.max_boundary_discontinuity_8bit:
        return True
    return False


def validate_pass(
    fusion: FusionResult,
    target_frame: np.ndarray,
    subject_box: BBox,
    cfg: PolicyConfig,
) -> ValidationPass:
    """Revert borrowed tiles that fail residual or 2-px boundary discontinuity.

    Neighbor context is the unchanged target on pixels that are not currently borrowed.
    """
    result = np.array(fusion.result_frame, copy=True)
    shape = target_frame.shape[:2]
    original_context = ~borrowed_mask(fusion, shape)
    new_decisions: list[TileDecision] = []
    reverted: list[TileDecision] = []
    for td in fusion.tile_decisions:
        if td.reason_code is not ReasonCode.BORROW_TILE_ACCEPTED or td.source_index < 1:
            new_decisions.append(td)
            continue
        residual = tile_residual_8bit(result, target_frame, td.box)
        disc = boundary_discontinuity_8bit(
            result, target_frame, td.box, original_context
        )
        if not _should_revert(residual, disc, cfg):
            new_decisions.append(td)
            continue
        x1, y1, x2, y2 = td.box
        result[y1:y2, x1:x2] = target_frame[y1:y2, x1:x2]
        reverted_td = TileDecision(
            td.tile_index,
            td.box,
            0,
            0.0,
            ReasonCode.REVERT_TILE_VALIDATION,
            td.improvement,
            residual,
        )
        new_decisions.append(reverted_td)
        reverted.append(reverted_td)
    sx1, sy1, sx2, sy2 = subject_box
    changed = float(
        (
            result[sy1:sy2, sx1:sx2] != fusion.result_frame[sy1:sy2, sx1:sx2]
        ).any(axis=2).mean()
    ) if sx2 > sx1 and sy2 > sy1 else 0.0
    updated = FusionResult(result, tuple(new_decisions), fusion.donor_lut_order)
    return ValidationPass(updated, len(reverted), changed, tuple(reverted))


def validate_and_revert(
    fusion: FusionResult,
    target_frame: np.ndarray,
    subject_box: BBox,
    cfg: PolicyConfig,
) -> tuple[FusionResult, int, tuple[TileDecision, ...]]:
    """Pass 1 already fused; this runs the validation revert and at most one revert-only pass.

    Returns ``(fusion, iteration_count, all_reverted_tiles)``. ``iteration_count`` is 1 after
    fuse+validate, or 2 if a second revert-only pass ran.
    """
    all_reverted: list[TileDecision] = []
    first = validate_pass(fusion, target_frame, subject_box, cfg)
    all_reverted.extend(first.reverted_tiles)
    current = first.fusion
    iteration = 1
    if (
        first.changed_fraction >= cfg.iteration.convergence_changed_fraction
        and iteration < cfg.iteration.max_passes
    ):
        second = validate_pass(current, target_frame, subject_box, cfg)
        all_reverted.extend(second.reverted_tiles)
        current = second.fusion
        iteration = 2
    return current, iteration, tuple(all_reverted)
