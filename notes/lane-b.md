# Lane B notes: alignment and color

Owner files: `src/probity/reconstruction/align.py`, `src/probity/reconstruction/color.py`,
`tests/unit/reconstruction/lane_b/**`. Contract: `docs/person2-lanes.md`.

## Interpretations

1. **AKAZE keypoints are the min of the two expanded crops.** `akaze.min_keypoints` is one gate
   whose observed value is `min(n_target, n_donor)` after CLAHE + AKAZE. Either crop under 20
   fails closed as `ALIGNMENT_FAILED` / `akaze.min_keypoints`.
2. **Ratio test is strict `<`.** A match is kept only when `d1 < akaze.ratio_test * d2`. Equality
   at 0.75 is discarded. Pairs with fewer than two Hamming neighbours are discarded. This is a
   matcher filter, not its own gate; the gated count is `akaze.min_good_matches`.
3. **ECC fallback only on feature starvation.** `align()` runs ECC iff the first failing AKAZE
   gate's `policy_key` is `akaze.min_keypoints` or `akaze.min_good_matches`. A homography that is
   estimated and then fails inlier / reprojection / scale / corner / coverage is **not** retried
   with ECC (spec: "If AKAZE fails only because it lacks enough features or matches").
4. **Unmeasured AKAZE metrics are 0.0** when there is no matrix (`types.py`). `A` on those rejects
   is therefore `0.30` (the zero-reprojection term). `A` is only used to rank accepted donors.
5. **ECC does not measure inliers or reprojection.** `inlier_ratio = 0.0`, `median_reproj_px = 0.0`
   by the unmeasured-metric rule. That zeros the inlier term of `A` and gives the ECC donor
   `A = 0.30 + 0.30 * coverage` (max 0.60), so a fallback never outranks a full AKAZE accept on
   the inlier/reproj terms. Geometry gates (scale, corner, coverage) still apply after ECC.
6. **Scale gates use requested policy keys that are not on this branch.**
   `alignment.scale_ratio_min = 0.67` and `alignment.scale_ratio_max = 1.50` via
   `getattr(..., default)`. `transform_scale` is `sqrt(projected_donor_box_area / donor_box_area)`
   and returns 0.0 for a mirror or a box that straddles the horizon, which cannot pass the min
   gate. Two gates (min then max); one operator each.
7. **CLAHE / AKAZE / RANSAC / ECC knobs** that are not on `PolicyConfig` use the requested
   defaults: `clahe.clip_limit=2.0`, `clahe.tile_grid=4`, `akaze.detector_threshold=0.001`,
   `ransac.max_iters=2000`, `ransac.confidence=0.995`, `ransac.rng_seed=20261009`,
   `ecc.gauss_filt_size=5`. CLAHE is skipped when a crop is not strictly larger than the tile
   grid (OpenCV would throw); AKAZE then runs on raw luma.
8. **Keypoints live in full-frame coordinates.** AKAZE runs on the expanded crop; `kp.pt` is
   offset by `expanded_box` origin before RANSAC. The homography maps donor full-frame -> target
   full-frame, matching `types.py`.
9. **Inlier ratio** is `mask.mean()` over the ratio-test matches fed to `findHomography`. If
   OpenCV returns no matrix, `inlier_ratio = 0.0` and later geometry gates are not recorded
   (no estimate). Median reprojection is the median inlier error in target pixels; if every
   projection is non-finite, a finite `1e9` is used so `A` stays defined and the reproj gate fails.
10. **ECC runs on expanded-crop luma in [0, 1]**, initialized from the axis-aligned affine that
    maps the donor subject box onto the target subject box, converted into crop coordinates
    because `findTransformECC` samples template (target crop) pixels in the input (donor crop).
    The refined 2x3 is converted back to a full-frame 3x3 with last row `(0, 0, 1)`. `cv2.error`
    or a non-finite correlation is `observed = 0.0` on `ecc.min_correlation`.
11. **OpenCV threads.** `cv2.setNumThreads(1)` at the start of AKAZE and ECC; `cv2.setRNGSeed`
    immediately before `findHomography`. Never pick between AKAZE and ECC by appearance.
12. **Warp sampling** uses inverse-homography source maps, `cv2.INTER_LANCZOS4` and
    `BORDER_REFLECT_101`, valid on the closed interval `[0, W-1] x [0, H-1]` of the **float32**
    maps. Points with homogeneous `w <= 0` are invalid and stored as `-1`.
13. **Color accept code is `PHOTOMETRIC_INCOMPATIBLE`.** `types.py` says so. Gain/bias failures
    reject as `LIGHTING_OUT_OF_RANGE` ("bounded color fit failed"); sample-count and residual
    failures reject as `PHOTOMETRIC_INCOMPATIBLE`. Short-circuit: too few samples returns identity
    gain/bias and does not invent a residual.
14. **Sobel is on target luma in 8-bit units**, `ksize=3`, compared to
    `color.max_context_gradient_8bit` (32). This is the same comparison as `<= 32/255` on luma in
    `[0, 1]` because Sobel is linear. The gradient is a sample selector, not a logged gate.
15. **Stable pixels** = context ring (expanded minus subject) AND valid warp AND Sobel <= 32.
    Fit is independent per BGR channel: three-iteration Huber IRLS (`k = 1.345 * 1.4826` MAD) of
    `target = gain * donor + bias`. Residual is the mean absolute 8-bit error of that fit on the
    same pixels. `apply_color` is `clip(rint(gain * crop + bias), 0, 255)` in float64, whole crop.
16. **Identity color when IRLS is unbounded.** If the unweighted IRLS design `[donor, 1]` is
    rank-deficient on any channel (near-constant context ring) **or** the fitted affine is
    outside gain/bias bounds while identity residual is already
    `<= color.max_mean_abs_residual_8bit`, return identity `(1,1,1)/(0,0,0)`. Applying a
    rank-deficient slope (translate car-body G gain ~0.10) is less conservative than
    identity. Sample and residual gates are unchanged and are not loosened. Identity
    residual outside the residual gate still rejects as `PHOTOMETRIC_INCOMPATIBLE` (flat
    ring) or, when the design is full rank, `LIGHTING_OUT_OF_RANGE` as today. Gates
    document the choice: a rank gate (`observed <= 1`) when deficient, and the residual
    accept reason names identity. A well-conditioned in-range affine is still estimated.
17. **Thin-strip / demo plate.** Unchanged from `NOTES-person2.md`: a 76x26 demo plate will
    `TOO_FEW_FEATURES` into ECC. Tests use the 220x72 synthetic plate size so AKAZE is exercisable.

## Status

- [x] AKAZE + RANSAC homography with every gate rule-coded
- [x] single ECC affine fallback (keypoints/matches failures only)
- [x] `warp_donor` source maps + LANCZOS4 remap
- [x] bounded robust color fit and `apply_color`
- [x] `uv run python scripts/check_lane.py --lane b` passes
