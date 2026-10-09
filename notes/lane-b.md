# Lane B notes: alignment and color

Owner files: `src/probity/reconstruction/align.py`, `src/probity/reconstruction/color.py`,
`tests/unit/reconstruction/lane_b/**`. Contract: `docs/person2-lanes.md`.

## Interpretations

(Number each decision the spec leaves open, as in `NOTES-person2.md`.)

## Status

- [ ] AKAZE + RANSAC homography with every gate rule-coded
- [ ] single ECC affine fallback (keypoints/matches failures only)
- [ ] `warp_donor` source maps + LANCZOS4 remap
- [ ] bounded robust color fit and `apply_color`
- [ ] `uv run python scripts/check_lane.py --lane b` passes
