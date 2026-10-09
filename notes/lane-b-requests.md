# Lane B requests

Changes this lane needs in files it does not own (types.py, conftest.py, quality.py, policy YAML,
pyproject, another lane's module). Never edit those files; add a row and keep working around it.

| # | File | Exact change requested | Why | Status |
| --- | --- | --- | --- | --- |
| 1 | `config/policy.demo.yaml` + `PolicyConfig` (team) | `ransac.max_iters: 2000`, `ransac.confidence: 0.995`, `ransac.rng_seed: 20261009`, `clahe.clip_limit: 2.0`, `clahe.tile_grid: 4`, `akaze.detector_threshold: 0.001`, `ecc.gauss_filt_size: 5` (already pending in `NOTES-person2.md`) | Determinism knobs absent from policy; Lane B uses `getattr` / module defaults matching these values | pending |
| 2 | `config/policy.demo.yaml` + `AlignmentPolicy` (team) | `alignment.scale_ratio_min: 0.67`, `alignment.scale_ratio_max: 1.50` (already pending in `NOTES-person2.md`) | Homography/ECC scale gates have no policy fields on this branch; `getattr(cfg.alignment, ...)` defaults to 0.67 / 1.50 | pending |
| 3 | `src/probity/reconstruction/color.py` (Lane B) | If the IRLS design is rank-deficient (near-constant ring) **or** identity residual is already `<= color.max_mean_abs_residual_8bit` while the affine gain/bias is out of range, return identity `(1,1,1)/(0,0,0)` and keep the residual/sample gates. Do not loosen those gates. | Integrator: four AKAZE-accepted translate donors fail `color.gain_min` (G gain ~0.10) on the flat car-body ring; identity residual is ~0.7–5/255. Unbounded slope is less conservative than identity. | pending |
