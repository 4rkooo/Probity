# Lane E notes: baseline isolation and evaluation metrics

Owner files: `src/probity/reconstruction/baseline.py`, `src/probity/eval/metrics.py`,
`tests/unit/reconstruction/lane_e/**`. Contract: `docs/person2-lanes.md`.

## Interpretations

(Number each decision the spec leaves open, as in `NOTES-person2.md`.)

## Status

- [ ] baseline URI layout, BASELINE AssetRef with non_evidentiary=true
- [ ] loader/tracker/reconstructor rejection of baseline output (tests only; io.py is frozen)
- [ ] PSNR/SSIM on linearized luma, provenance percentages, changed-pixel rates
- [ ] OCR character accuracy (strings only), alignment rejection rate, EvaluationRow
- [ ] `uv run python scripts/check_lane.py --lane e` passes
