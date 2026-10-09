# Lane D notes: provenance writer and integrity

Owner files: `src/probity/reconstruction/provenance.py`, `src/probity/reconstruction/integrity.py`,
`tests/unit/reconstruction/lane_d/**`. Contract: `docs/person2-lanes.md`.

## Interpretations

(Number each decision the spec leaves open, as in `NOTES-person2.md`.)

## Status

- [ ] `build_source_lut` (9-value homography / 6-value affine, decision ids)
- [ ] `build_provenance_arrays` (identity init, borrowed tiles, never class 2)
- [ ] `provenance_record` (validated, never partial)
- [ ] integrity-v1 components, Au definition, order-independence shuffle test
- [ ] `uv run python scripts/check_lane.py --lane d` passes
