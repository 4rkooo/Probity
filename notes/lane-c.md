# Lane C notes: ranking and tile fusion

Owner files: `src/probity/reconstruction/fuse.py`, `tests/unit/reconstruction/lane_c/**`.
Contract: `docs/person2-lanes.md`.

## Interpretations

(Number each decision the spec leaves open, as in `NOTES-person2.md`.)

## Status

- [ ] T, R, rank order (R desc, frame asc), max_count cap
- [ ] 8x8 winner-take-all selection with keep/no-gain/residual reasons
- [ ] `tile_gates` rule-coded rows
- [ ] `uv run python scripts/check_lane.py --lane c` passes
