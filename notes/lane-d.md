# Lane D notes: provenance writer and integrity

Owner files: `src/probity/reconstruction/provenance.py`, `src/probity/reconstruction/integrity.py`,
`tests/unit/reconstruction/lane_d/**`. Contract: `docs/person2-lanes.md`.

## Interpretations

1. **Never class 2.** The writer initialises every pixel as `ORIGINAL` with identity coordinates
   and overwrites only accepted subject tiles as `BORROWED`. It never writes
   `GENERATED_BLEND`. A single class-2 pixel in arrays offered to `validate_arrays`,
   `provenance_record`, or `compute_integrity` raises `GeneratedPixelError`
   (`GENERATED_SEMANTIC_PIXEL`).
2. **Never partial provenance.** `build_provenance_arrays` and `provenance_record` collect every
   problem they can and raise `ProvenanceIncomplete` with the full list. They never return arrays
   or a `PixelProvenance` when any check failed. Identity arrays are copied before mutation so the
   temporary `ProvenanceArrays.identity` object is not mutated after construction.
3. **LUT matrix width.** AKAZE homography stores all 9 row-major `float64` values. ECC affine
   stores the top two rows (6 values). The 3x3 form must be finite, invertible, and (for ECC) end
   with `(0, 0, 1)`. `transform_id` is `h_{frame}` or `a_{frame}`. Interpolation is always
   `LANCZOS4` for donors and `IDENTITY` for the target row.
4. **LUT order is rank order.** Donors must already be sorted by `R` descending, then frame number
   ascending. `fusion.donor_lut_order` must match that sequence. A mismatch is
   `PROVENANCE_INCOMPLETE`, not a silent re-sort.
5. **Donor lineage is a hard stop.** A donor from another video or track, the target frame itself,
   a bridged/non-detector observation, or a different frame size raises `DonorLineageError`
   (`DONOR_TRACK_MISMATCH`) before any LUT row is built. Preflight should have rejected these;
   the writer does not repair them.
6. **Same-video frames share the output size.** `provenance_record` and `compute_integrity` pass
   `(width, height)` of the class map as every LUT row's source size. Lineage already requires
   `donor.frame_size == target.frame_size`.
7. **Borrowed tiles must be exact warped samples.** Each `BORROW_TILE_ACCEPTED` tile must lie
   inside the subject box, name a donor LUT row, have 100% valid warp coverage, not overlap
   another borrowed tile, and match the donor `warped_crop` byte-for-byte. Changed pixels in the
   expanded box that are not claimed as borrowed are `PROVENANCE_INCOMPLETE`. Kept tiles must
   name `source_index` 0.
8. **`C` is full-frame supported coverage.** `supported_coverage = (ORIGINAL + BORROWED) / (W*H)`.
   After `validate_arrays` this is always `1.0`. Incomplete coverage is refused rather than
   scored (`integrity.required_provenance_coverage = 1.00`).
9. **`Qd` / `Ad` are borrowed-pixel-weighted.** Weight is the integer count of `BORROWED` pixels
   whose `source_index` is that LUT row. Extra unused donors do not change the means. A borrowed
   row with no donor in the mapping is `PROVENANCE_INCOMPLETE`. Components are rounded to 6
   decimal places to match the hand-built goldens (`handbuilt_runs.py`).
10. **No borrowed pixels refuse, never invent `Qd`.** `compute_integrity` raises
    `IntegrityRefusal` with `NO_TILE_IMPROVED`. An all-original map is a valid provenance
    artifact; it is not a valid integrity score.
11. **`G` is the generated fraction inside the subject box**, stored as
    `semantic_generated_pct` in `[0, 100]`. `validate_arrays` refuses any class-2 pixel first, so
    a successful score always has `G = 0` (`integrity.max_semantic_generated_fraction = 0.00`).
12. **`Au` is presence of expected material decisions**, order-independent. Expected items are
    (a) every `decision_id` on every DONOR LUT row, and (b) the run-stage pairs
    `(INPUT, SOURCE_HASH_VERIFIED)`, `(TRACK, TRACK_CONFIRMED)`, `(QUALITY, TARGET_TOO_SMALL)`,
    `(QUALITY, TARGET_QUALITY_TOO_LOW)`, `(RANK, INSUFFICIENT_COMPATIBLE_DONORS)`,
    `(FUSE, KEEP_ORIGINAL_CLEAR)`, `(PROVENANCE, PROVENANCE_COMPLETE)`,
    `(PROVENANCE, GENERATED_SEMANTIC_PIXEL)`. Extra log rows do not raise `Au`. An empty
    expected set yields `0.0`, not `1.0`. `INTEGRITY` is omitted so `Au` is not circular.
    `MODE` / fixture disclosure is not a reconstruction gate. Presence is enough; outcome is not
    scored.
13. **`Tc` is the caller's track continuity**, required to be finite and in `[0, 1]`. Out-of-range
    values are `PROVENANCE_INCOMPLETE` (a plumbing error, not a clip).
14. **Integrity is order-independent.** `Qd`/`Ad` sum by LUT index; `Au` uses set membership;
    `C`/`G` are array reductions. Shuffling `decisions` or the insertion order of `donors` must
    not change the `IntegrityScore`.
15. **`integrity_gate`** is `score_0_100 >= integrity.min_score` (70 points), rule
    `INTEGRITY_BELOW_MINIMUM`, stage `INTEGRITY`, units `points`. Score 70 accepts; score 69
    rejects with that reason code. The score itself is `integrity_score_0_100` (half-up).
16. **Writer tests never call lanes B or C.** Donor warps are `cv2.remap(..., INTER_LANCZOS4,
    BORDER_REFLECT_101)` over a seeded translated scene. Golden fixtures are read-only.

## Status

- [x] `build_source_lut` (9-value homography / 6-value affine, decision ids)
- [x] `build_provenance_arrays` (identity init, borrowed tiles, never class 2)
- [x] `provenance_record` (validated, never partial)
- [x] integrity-v1 components, Au definition, order-independence shuffle test
- [x] `uv run python scripts/check_lane.py --lane d` passes
