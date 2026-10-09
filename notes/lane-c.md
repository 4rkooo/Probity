# Lane C notes: ranking and tile fusion

Owner files: `src/probity/reconstruction/fuse.py`, `tests/unit/reconstruction/lane_c/**`.
Contract: `docs/person2-lanes.md`.

## Interpretations

(Number each decision the spec leaves open, as in `NOTES-person2.md`.)

### Ranking (step 8)

C1. `T = exp(-|dt_s| / donor.temporal_decay_s)` exactly; a non-finite `dt_s` raises `ValueError`
    (a plumbing error, not a refusal). `dt_s` is computed by the caller; `rank_donors` cannot see
    the target, so it checks `R` against `rank_score(Q, A, T)` but cannot re-derive `T`.
C2. `R = 0.45 Q + 0.35 A + 0.20 T` with `Q = quality.Q` and `A = alignment.A`. Inputs outside
    `[0, 1]` or non-finite raise `ValueError` rather than being clipped: clipping would hide an
    upstream bug in a score that orders evidence.
C3. LUT order is `(R desc, frame_number asc)` and is the only tie-break. `rank_donors` sorts with
    the key `(-R, frame_number)`, so the result is independent of input order. `max_count` is applied
    after sorting (the lowest-ranked donors are dropped).
C4. **Conservative input checks (fail closed, `ValidationFailed`).** `rank_donors` and `fuse_tiles`
    refuse a pool that mixes videos or tracks, repeats a frame, contains a bridged-only or
    non-detector observation, contains a donor whose `alignment.accepted` or `color.accepted` is
    False, or whose stored `R` disagrees with the formula (tolerance 1e-9). These are invariants
    upstream lanes already enforce; checking them again means a mis-built `AlignedDonor` can never
    donate pixels. They raise instead of logging because they are plumbing errors (same reading
    as `NOTES-person2.md` #23), not evidence decisions.
C5. `min_count` is a gate, not an exception: `donor_count_gate(ranked, cfg)` (new public function)
    returns the `INSUFFICIENT_COMPATIBLE_DONORS` / `RANK` gate (`observed = len(ranked) >= donor.min_count`,
    units `donors`, subject `donors`), exactly the row the hand-built golden logs. It is applied to
    the capped list, so the count is of donors that can actually be used. `rank_donors` itself never
    refuses; the integrator logs the gate and refuses on failure.

## Status

- [x] T, R, rank order (R desc, frame asc), max_count cap, min_count gate
- [ ] 8x8 winner-take-all selection with keep/no-gain/residual reasons
- [ ] `tile_gates` rule-coded rows
- [ ] `uv run python scripts/check_lane.py --lane c` passes
