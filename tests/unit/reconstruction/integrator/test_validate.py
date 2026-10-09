"""Integrator revert-only validation: second pass never adds borrowing."""

from __future__ import annotations

import numpy as np

from probity.domain.enums import ReasonCode
from probity.reconstruction.types import FusionResult, TileDecision
from probity.reconstruction.validate import validate_and_revert, validate_pass


def _box(x1: int, y1: int, tile: int = 8) -> tuple[int, int, int, int]:
    return (x1, y1, x1 + tile, y1 + tile)


def test_validation_reverts_high_residual_tile(policy) -> None:
    target = np.zeros((32, 32, 3), dtype=np.uint8)
    result = target.copy()
    box = _box(8, 8)
    result[8:16, 8:16] = 255
    fusion = FusionResult(
        result,
        (TileDecision(0, box, 1, 1.0, ReasonCode.BORROW_TILE_ACCEPTED, 0.5, 255.0),),
        (10,),
    )
    out = validate_pass(fusion, target, (0, 0, 32, 32), policy)
    assert out.n_reverted == 1
    assert out.fusion.tile_decisions[0].reason_code is ReasonCode.REVERT_TILE_VALIDATION
    assert out.fusion.tile_decisions[0].source_index == 0
    assert np.array_equal(out.fusion.result_frame[8:16, 8:16], target[8:16, 8:16])


def test_second_pass_only_reverts(policy) -> None:
    target = np.zeros((32, 32, 3), dtype=np.uint8)
    result = target.copy()
    box_a, box_b = _box(8, 8), _box(16, 8)
    result[8:16, 8:16] = 40
    result[8:16, 16:24] = 255
    fusion = FusionResult(
        result,
        (
            TileDecision(0, box_a, 1, 1.0, ReasonCode.BORROW_TILE_ACCEPTED, 0.2, 40.0),
            TileDecision(1, box_b, 2, 1.0, ReasonCode.BORROW_TILE_ACCEPTED, 0.2, 255.0),
        ),
        (10, 11),
    )
    updated, iteration, reverted = validate_and_revert(fusion, target, (0, 0, 32, 32), policy)
    assert iteration <= policy.iteration.max_passes
    reasons = {td.reason_code for td in updated.tile_decisions}
    assert ReasonCode.REVERT_TILE_VALIDATION in reasons
    borrowed = [
        td for td in updated.tile_decisions
        if td.reason_code is ReasonCode.BORROW_TILE_ACCEPTED
    ]
    # A second pass must not introduce a new source_index; it may only drop borrows.
    assert all(td.source_index in (1, 2) for td in borrowed)
    assert len(reverted) >= 1
