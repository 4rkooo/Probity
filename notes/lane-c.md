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

### Fusion (step 9)

C6. Keep-original uses `s_target >= fusion.keep_original_sharpness` (spec: "at least 0.45") and
    short-circuits before any donor is scored. A tile that is already clear is never replaced,
    even if a donor would score higher.
C7. Relative improvement is `(S_donor - S_target) / max(S_target, fusion.sharpness_ratio_epsilon)`.
    The min-gain gate is `improvement >= fusion.min_sharpness_improvement` (spec: "at least 15%"),
    implemented as reject when `improvement <` the policy value. A donor below that gate is not a
    candidate; if no donor clears it, the reason is `NO_TILE_IMPROVED`.
C8. Coverage is the fraction of tile pixels whose `valid_mask` is True. The gate is
    `coverage >= fusion.required_valid_tile_coverage` (1.00: every pixel). Residual is the mean
    absolute BGR difference in 8-bit units over the tile, measured only when coverage passes
    (otherwise it is not a number and cannot be scored). Residual accepts `<= color.max_mean_abs_residual_8bit`
    (spec: "<=18/255"). A donor that cleared the gain gate but failed coverage or residual yields
    `PHOTOMETRIC_INCOMPATIBLE` when no other donor is eligible.
C9. Selection score is `0.50 * improvement + 0.30 * R + 0.20 * (1 - residual / color.max_mean_abs_residual_8bit)`.
    The residual limit comes from config, not a literal 18. The winner is the strict maximum; a
    tied score keeps the earlier LUT index (donors are already in LUT order). Never average or
    blend: the whole tile is copied from exactly one warped crop, or left as the original.
C10. `fuse_tiles` requires donors already in LUT order and already capped at `max_count`. It
    refuses a donor that is the target frame, a pool over `max_count`, mixed video/track, or a
    crop/mask whose shape is not the target expanded box. `target_frame` must be a uint8 BGR copy
    of the pixels in `target.crop` / `target.expanded_crop`. These are plumbing errors.
C11. Sharpness is `quality.tile_sharpness_scores` on subject luma (the same resized-grid S as
    `NOTES-person2.md` #4). Tiles are `quality.tiles(target.bbox_px, fusion.tile_px)`, row-major,
    edge tiles clipped. Only subject-box pixels are overwritten; the rest of the full-frame copy
    is the unchanged target. The function never reads a previous result.
C12. `tile_gates` emits one `BORROW_TILE_ACCEPTED` / `FUSE` row per borrowed tile
    (`subject_ref = tile:{x1},{y1}`, observed = improvement, `>= fusion.min_sharpness_improvement`,
    units `ratio`) and always one aggregate `KEEP_ORIGINAL_CLEAR` / `FUSE` row on subject `tiles`
    (`observed = n_kept >= 0`, units `tiles`, `fusion.keep_original_sharpness`) whose accept
    reason is the hand-built count string: already clear / not improved / failed residual-or-coverage.
    `REVERT_TILE_VALIDATION` (integrator validation pass) counts as the residual/coverage bucket
    so the three counts still sum to every kept tile. `tile_gates` does **not** emit the run-level
    `NO_TILE_IMPROVED` refusal when zero tiles are borrowed; that is the integrator's job after
    it sees `n_borrowed == 0` (hand-built `handbuilt_runs.py`).
C13. `FusionResult.donor_lut_order` is frame numbers only, so a borrow accept-reason cannot
    reconstruct `{video_id}:f{n}`. The reason cites `f{frame_number}` (the `frame_id` suffix) and
    the residual to two decimals, matching the hand-built residual phrase. No `types.py` change
    requested: the integrator has the video id when it writes the log.

## Status

- [x] T, R, rank order (R desc, frame asc), max_count cap, min_count gate
- [x] 8x8 winner-take-all selection with keep/no-gain/residual reasons
- [x] `tile_gates` rule-coded rows
- [x] `uv run python scripts/check_lane.py --lane c` passes
