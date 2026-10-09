# Lane B requests

Changes this lane needs in files it does not own (types.py, conftest.py, quality.py, policy YAML,
pyproject, another lane's module). Never edit those files; add a row and keep working around it.

| # | File | Exact change requested | Why | Status |
| --- | --- | --- | --- | --- |
| 1 | `config/policy.demo.yaml` + `PolicyConfig` (team) | `ransac.max_iters: 2000`, `ransac.confidence: 0.995`, `ransac.rng_seed: 20261009`, `clahe.clip_limit: 2.0`, `clahe.tile_grid: 4`, `akaze.detector_threshold: 0.001`, `ecc.gauss_filt_size: 5` (already pending in `NOTES-person2.md`) | Determinism knobs absent from policy; use module constants in `align.py` with these values until added | pending |
