"""Lane C step 8: temporal preference T, rank score R, LUT order, max_count cap, min_count gate."""

from __future__ import annotations

import math

import pytest

from probity.domain.enums import PolicyStage, ReasonCode
from probity.domain.errors import ValidationFailed
from probity.domain.policy import PolicyConfig
from probity.reconstruction import fuse

from ._builders import TRACK, VIDEO, make_donor, make_target, quality


def test_temporal_preference_formula(policy: PolicyConfig) -> None:
    decay = policy.donor.temporal_decay_s
    assert fuse.temporal_preference(0.0, policy) == 1.0
    assert fuse.temporal_preference(decay, policy) == pytest.approx(math.exp(-1.0), abs=1e-15)
    assert fuse.temporal_preference(-1.0, policy) == fuse.temporal_preference(1.0, policy)
    assert fuse.temporal_preference(0.5, policy) > fuse.temporal_preference(1.5, policy)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_temporal_preference_rejects_non_finite(policy: PolicyConfig, bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        fuse.temporal_preference(bad, policy)


def test_rank_score_formula() -> None:
    assert fuse.rank_score(quality(0.6), 0.8, 0.5) == pytest.approx(
        0.45 * 0.6 + 0.35 * 0.8 + 0.20 * 0.5, abs=1e-15
    )
    assert fuse.rank_score(quality(1.0), 1.0, 1.0) == pytest.approx(1.0, abs=1e-15)
    assert fuse.rank_score(quality(0.0), 0.0, 0.0) == 0.0


@pytest.mark.parametrize(("Q", "A", "T"), [(1.1, 0.5, 0.5), (0.5, -0.1, 0.5), (0.5, 0.5, math.nan)])
def test_rank_score_rejects_out_of_range(Q: float, A: float, T: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fuse.rank_score(quality(Q), A, T)


def test_rank_order_is_r_desc_then_frame_asc(policy: PolicyConfig) -> None:
    _, target = make_target(policy)
    # Same Q/A/T -> identical R; frames 40 and 20 tie and must order by frame ascending.
    tie_hi = make_donor(target, 40, policy, Q=0.8, A=0.9, T=0.9)
    tie_lo = make_donor(target, 20, policy, Q=0.8, A=0.9, T=0.9)
    best = make_donor(target, 50, policy, Q=0.95, A=0.95, T=0.9)
    worst = make_donor(target, 10, policy, Q=0.56, A=0.7, T=0.9)
    ranked = fuse.rank_donors([worst, tie_hi, best, tie_lo], policy)
    assert [d.obs.frame_number for d in ranked] == [50, 20, 40, 10]
    assert tie_hi.R == tie_lo.R


def test_rank_order_is_input_order_independent(policy: PolicyConfig) -> None:
    _, target = make_target(policy)
    donors = [
        make_donor(target, n, policy, Q=q_, A=0.9)
        for n, q_ in [(22, 0.7), (24, 0.7), (36, 0.9), (38, 0.6), (26, 0.8)]
    ]
    expected = [d.obs.frame_number for d in fuse.rank_donors(donors, policy)]
    for order in (donors[::-1], donors[2:] + donors[:2], sorted(donors, key=lambda d: d.R)):
        assert [d.obs.frame_number for d in fuse.rank_donors(order, policy)] == expected


def test_max_count_cap_inside_and_outside(policy: PolicyConfig) -> None:
    _, target = make_target(policy)
    cap = policy.donor.max_count
    donors = [make_donor(target, 20 + i, policy, Q=0.6 + 0.01 * i) for i in range(cap + 1)]
    at_cap = fuse.rank_donors(donors[:cap], policy)
    assert len(at_cap) == cap
    over = fuse.rank_donors(donors, policy)
    assert len(over) == cap
    lowest = min(donors, key=lambda d: (d.R, -d.obs.frame_number))
    assert lowest not in over
    assert [d.R for d in over] == sorted((d.R for d in over), reverse=True)


def test_donor_count_gate_inside_and_outside(policy: PolicyConfig) -> None:
    _, target = make_target(policy)
    need = policy.donor.min_count
    donors = fuse.rank_donors([make_donor(target, 20 + i, policy) for i in range(need)], policy)
    inside = fuse.donor_count_gate(donors, policy)
    assert inside.ok
    assert inside.rule_code is ReasonCode.INSUFFICIENT_COMPATIBLE_DONORS
    assert (inside.stage, inside.observed, inside.operator, inside.threshold) == (
        PolicyStage.RANK,
        need,
        ">=",
        need,
    )
    assert (inside.units, inside.policy_key) == ("donors", "donor.min_count")
    outside = fuse.donor_count_gate(donors[: need - 1], policy)
    assert not outside.ok
    assert outside.failure_code is ReasonCode.INSUFFICIENT_COMPATIBLE_DONORS
    assert outside.observed == need - 1


def test_empty_pool_ranks_to_empty_and_fails_count(policy: PolicyConfig) -> None:
    assert fuse.rank_donors([], policy) == ()
    assert not fuse.donor_count_gate((), policy).ok


@pytest.mark.parametrize("field", ["video_id", "track_id"])
def test_never_ranks_across_tracks_or_videos(policy: PolicyConfig, field: str) -> None:
    _, target = make_target(policy)
    other = {"video_id": VIDEO, "track_id": TRACK, field: "01a1219b-7a80-7bd1-b62c-000000000000"}
    donors = [make_donor(target, 20, policy), make_donor(target, 22, policy, **other)]
    with pytest.raises(ValidationFailed, match="video/track"):
        fuse.rank_donors(donors, policy)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bridged": True}, "detector-backed"),
        ({"detector_backed": False}, "detector-backed"),
        ({"aligned": False}, "alignment and color"),
        ({"colored": False}, "alignment and color"),
        ({"R": 0.99}, "formula"),
    ],
)
def test_rejects_donors_that_did_not_pass_upstream_gates(
    policy: PolicyConfig, kwargs: dict[str, object], match: str
) -> None:
    _, target = make_target(policy)
    with pytest.raises(ValidationFailed, match=match):
        fuse.rank_donors([make_donor(target, 20, policy, **kwargs)], policy)  # type: ignore[arg-type]


def test_rejects_duplicate_frames(policy: PolicyConfig) -> None:
    _, target = make_target(policy)
    with pytest.raises(ValidationFailed, match="twice"):
        fuse.rank_donors([make_donor(target, 20, policy), make_donor(target, 20, policy)], policy)
