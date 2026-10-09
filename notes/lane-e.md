# Lane E notes: baseline isolation and evaluation metrics

Owner files: `src/probity/reconstruction/baseline.py`, `src/probity/eval/metrics.py`,
`tests/unit/reconstruction/lane_e/**`. Contract: `docs/person2-lanes.md`.

## Interpretations

Numbered like `NOTES-person2.md`. Conservative / refuse-leaning where the spec is open.

1. **Baseline path.** `baseline_uri` is exactly `derived/{video_id}/baseline/{run_id}/{filename}`.
   `video_id` must be a UUIDv7; `run_id` and `filename` are a single safe path segment
   (`[A-Za-z0-9._-]+`, not starting with `.`). No `source/` path can be produced.
2. **`run_baseline` reads `target.crop` only**, never `expanded_crop`, donors, or a prior result.
   The upscaler must return BGR uint8 at one integer scale of that crop (Real-ESRGAN x4plus is
   scale 4 in live mode; tests use `cv2.resize` INTER_NEAREST). Non-integer or anisotropic scales
   are refused.
3. **`write_baseline` always emits `AssetKind.BASELINE` with `non_evidentiary=True`.** PNG and
   `baseline.json` go through `store.derived_path(video_id, "baseline", run_id, ...)`. The store
   path is checked to end with `derived/{video_id}/baseline/{run_id}/{filename}`. Sidecar records
   model id, model SHA-256, and license. Live Real-ESRGAN is not a test dependency.
4. **Loader / tracker / reconstructor isolation** is the frozen `assert_evidentiary_input` gate
   (`GENERATED_INPUT_REJECTED`). Lane E tests it; `io.py` is not edited. A `frames/` URI is
   accepted; a `baseline/` URI or `BASELINE` AssetRef is rejected. Reconstruction modules other
   than `baseline.py` must not import it.
5. **Linear luma** is sRGB decode (knee 0.04045) then Rec. 709 weights (0.2126, 0.7152, 0.0722)
   on RGB after BGR swap, clipped to [0, 1]. PSNR/SSIM are defined only on this luma.
6. **PSNR** uses peak 1.0 (linearized range). Identical images return `+inf`. An empty mask is
   refused: there is no registered ground truth. PSNR is never written into an `EvaluationRow`
   when non-finite (interpretation 12).
7. **SSIM** is mean SSIM on linearized luma: Gaussian window 11, sigma 1.5, K1=0.01, K2=0.03,
   L=1.0, `BORDER_REFLECT_101`. Images smaller than 11x11 are refused. Variance is floored at 0
   after the blur so float noise cannot invert the map.
8. **Provenance percentages** are 100 * count / N, rounded to 4 decimal places, keys
   `ORIGINAL` / `BORROWED` / `GENERATED_BLEND`, matching `CoveragePct`. `subject_box` is
   full-frame half-open and must lie wholly inside the array (no silent clip).
9. **Supported changed-pixel rate** = (changed AND class BORROWED AND `source_index` is a donor
   LUT row with `index >= 1`) / changed. **1.0 if nothing changed** (complement of lane D's
   unsupported rate, which is 0.0 when nothing changed). A BORROWED label without a donor LUT
   row does not count as supported. Lane E does not import `provenance.py`.
10. **OCR character accuracy** is exact, case-sensitive, per-position over `len(truth)`. Extra
    predicted characters do not change the denominator. Empty truth is refused (undefined). No
    OCR engine runs here; strings come from the eval harness / generator only.
11. **Alignment rejection rate** counts unique `subject_ref` among **ALIGN-stage** decisions
    only. `"total"` is rejected / considered. Per-key rates use the first REJECT `rule_code`
    for that donor, divided by considered. PREFLIGHT/COLOR rejects are not alignment
    rejections. No ALIGN decisions => `{"total": 0.0}`.
12. **`evaluation_row`** sorts metric names, requires a UUIDv7 `correlation_id`, and refuses
    non-finite values (including PSNR inf) and bools. Ground-truth metrics and provenance
    metrics are separate keys; this module never ranks methods by sharpness.
13. **No live model.** Tests inject `FakeUpscaler` (`cv2.resize`). An optional `baseline` extra
    for Real-ESRGAN is requested of Person 1 only when live eval is wired; not needed for this
    lane.

## Thresholds covered (inside / outside)

Lane E has no numeric policy table of its own. The isolation gate is binary; metric fractions
are checked just inside a perfect score and just outside it.

| Gate / metric | Just inside | Just outside | Reason / value |
| --- | --- | --- | --- |
| Evidentiary URI | `derived/{vid}/frames/f1/frame.png` accepted | `derived/{vid}/baseline/{run}/baseline.png` rejected | `GENERATED_INPUT_REJECTED` |
| Baseline AssetRef as loader/tracker/reconstructor input | source `FrameReference` accepted | `AssetKind.BASELINE` rejected | `GENERATED_INPUT_REJECTED` |
| `BASELINE` must be `non_evidentiary` | `True` constructs | `False` fails domain validation | (Pydantic; domain frozen) |
| OCR | `"PRB 4K7"` vs truth → 1.0 | one substitution → 6/7 | strings only |
| Supported changed pixels | BORROWED + donor LUT row → 1.0 | BORROWED but LUT has no donor → 0.0 | 1.0 if none changed |
| Alignment rejection | two ALIGN ACCEPTs → `total` 0.0 | one ACCEPT + one `ALIGNMENT_FAILED` → 0.5 | key `ALIGNMENT_FAILED` |
| PSNR mask | mask excludes the changed pixel → inf | mask is only that pixel → finite | empty mask refused |

## Status

- [x] baseline URI layout, BASELINE AssetRef with non_evidentiary=true
- [x] loader/tracker/reconstructor rejection of baseline output (tests only; io.py is frozen)
- [x] PSNR/SSIM on linearized luma, provenance percentages, changed-pixel rates
- [x] OCR character accuracy (strings only), alignment rejection rate, EvaluationRow
- [x] `uv run python scripts/check_lane.py --lane e` passes
