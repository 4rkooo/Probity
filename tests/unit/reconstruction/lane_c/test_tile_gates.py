"""Lane C: rule-coded ``tile_gates`` rows (borrow per tile, one aggregate keep)."""

from __future__ import annotations

import pytest

from probity.domain.enums import PolicyStage, ReasonCode
from probity.domain.errors import ValidationFailed
from probity.domain.policy import PolicyConfig
from probity.reconstruction import fuse
from probity.reconstruction.types import FusionResult, TileDecision

from ._builders import checker, crop_shape, make_donor, make_target


TILE_A = (16, 16, 24, 24)
TILE_B = (24, 16, 32, 24)
TILE_C = (32, 16, 40, 24)


def _borrow(index: int, box: tuple[int, int, int, int], source: int, *, imp: float, res: float) -> TileDecision:
    return TileDecision(index, box, source, 0.75, ReasonCode.BORROW_TILE_ACCEPTED, imp, res)


def test_tile_gates_borrow_rows_and_aggregate_keep(policy: PolicyConfig) -> None:
    result = FusionResult(
        result_frame=None,  # type: ignore[arg-type]
        tile_decisions=(
            _borrow(0, TILE_A, 1, imp=0.20, res=4.0),
            TileDecision(1, TILE_B, 0, 0.0, ReasonCode.KEEP_ORIGINAL_CLEAR),
            TileDecision(2, TILE_C, 0, 0.0, ReasonCode.NO_TILE_IMPROVED, 0.05),
            TileDecision(3, (40, 16, 48, 24), 0, 0.0, ReasonCode.PHOTOMETRIC_INCOMPATIBLE, 0.4, 19.0),
        ),
        donor_lut_order=(20, 22),
    )
    rows = fuse.tile_gates(result, policy)
    assert [ref for ref, _ in rows] == ["tile:16,16", "tiles"]
    borrow_ref, borrow = rows[0]
    assert borrow_ref == "tile:16,16"
    assert borrow.rule_code is ReasonCode.BORROW_TILE_ACCEPTED
    assert borrow.stage is PolicyStage.FUSE
    assert (borrow.observed, borrow.operator, borrow.threshold) == (
        0.20,
        ">=",
        policy.fusion.min_sharpness_improvement,
    )
    assert (borrow.units, borrow.policy_key) == ("ratio", "fusion.min_sharpness_improvement")
    assert borrow.ok
    assert borrow.accept_reason == "Borrowed from f20; residual 4.00/255"

    keep_ref, keep = rows[1]
    assert keep_ref == "tiles"
    assert keep.rule_code is ReasonCode.KEEP_ORIGINAL_CLEAR
    assert keep.stage is PolicyStage.FUSE
    assert (keep.observed, keep.operator, keep.threshold) == (3, ">=", 0)
    assert (keep.units, keep.policy_key) == ("tiles", "fusion.keep_original_sharpness")
    assert keep.ok
    assert keep.accept_reason == (
        "Kept original: 1 already clear, 1 not improved, 1 failed residual/coverage"
    )


def test_tile_gates_always_emits_keep_row_when_every_tile_borrowed(policy: PolicyConfig) -> None:
    result = FusionResult(
        result_frame=None,  # type: ignore[arg-type]
        tile_decisions=(_borrow(0, TILE_A, 2, imp=0.50, res=1.25),),
        donor_lut_order=(20, 22),
    )
    rows = fuse.tile_gates(result, policy)
    assert rows[0][0] == "tile:16,16"
    assert "f22" in rows[0][1].accept_reason
    assert rows[1][1].observed == 0
    assert rows[1][1].accept_reason == (
        "Kept original: 0 already clear, 0 not improved, 0 failed residual/coverage"
    )
    assert rows[1][1].ok


def test_tile_gates_counts_revert_as_residual_keep(policy: PolicyConfig) -> None:
    result = FusionResult(
        result_frame=None,  # type: ignore[arg-type]
        tile_decisions=(
            TileDecision(0, TILE_A, 0, 0.0, ReasonCode.REVERT_TILE_VALIDATION, 0.3, 25.0),
        ),
        donor_lut_order=(20,),
    )
    _, keep = fuse.tile_gates(result, policy)[0]
    assert keep.accept_reason == (
        "Kept original: 0 already clear, 0 not improved, 1 failed residual/coverage"
    )


def test_tile_gates_matches_fuse_tiles_decisions(policy: PolicyConfig) -> None:
    img, target = make_target(policy)
    donor = make_donor(target, 20, policy, warped=checker(crop_shape(target), 10))
    fused = fuse.fuse_tiles(img, target, (donor,), policy)
    rows = fuse.tile_gates(fused, policy)
    borrowed = [td for td in fused.tile_decisions if td.reason_code is ReasonCode.BORROW_TILE_ACCEPTED]
    assert len(rows) == len(borrowed) + 1
    for (ref, gate), td in zip(rows[:-1], borrowed, strict=True):
        assert ref == f"tile:{td.box[0]},{td.box[1]}"
        assert gate.observed == td.improvement
        assert gate.ok
    assert rows[-1][0] == "tiles"
    assert rows[-1][1].observed == len(fused.tile_decisions) - len(borrowed)


def test_tile_gates_rejects_borrow_outside_lut(policy: PolicyConfig) -> None:
    result = FusionResult(
        result_frame=None,  # type: ignore[arg-type]
        tile_decisions=(_borrow(0, TILE_A, 3, imp=0.2, res=1.0),),
        donor_lut_order=(20, 22),
    )
    with pytest.raises(ValidationFailed, match="source_index"):
        fuse.tile_gates(result, policy)
