# Person 2 parallel lanes (TrueFrame)

Up to five agents work in separate git worktrees without touching each other, Person 1's API, or
Person 3's UI. `PROBITY_TECHNICAL_DESIGN.md` stays authoritative; `NOTES-person2.md` records every
interpretation. Every kickoff rule still applies in every lane (invariants, determinism, pathlib,
atomic LF JSON, no shell=True, no edits to Person 1 files, no dependency edits).

## Branches

- `person2/trueframe` is the integration branch. Kickoff steps 1-3 (synthetic goldens, loader and
  provenance validator; track confirmation; quality and preflight hard gates) landed directly on
  it, followed by the type move and "p2: parallel scaffolding".
- Each lane works on `person2/lane-<x>` in `..\probity-p2-<x>`, created by
  `scripts/make_worktrees.ps1` (lanes B-E). Lane A (tracking/quality) is complete and has none.
- Merge order into `person2/trueframe`: any order for B, C, D, E (they share no files); then the
  INTEGRATOR, sequentially; then lane F.

## Lanes

| Lane | Scope | Owns |
| --- | --- | --- |
| A (done) | Tracking, identity and quality gates, preflight (kickoff steps 2-3) | merged; now shared read-only |
| B | AKAZE/RANSAC, one ECC fallback, bounded color fit (steps 4-5) | `reconstruction/align.py`, `reconstruction/color.py`, `tests/unit/reconstruction/lane_b/**` |
| C | Donor ranking, 8x8 winner-take-all fusion (step 6) | `reconstruction/fuse.py`, `tests/unit/reconstruction/lane_c/**` |
| D | Provenance writer/validator, integrity-v1 | `reconstruction/provenance.py`, `reconstruction/integrity.py`, `tests/unit/reconstruction/lane_d/**` |
| E | Real-ESRGAN baseline isolation, eval metrics (step 9) | `reconstruction/baseline.py`, `eval/metrics.py`, `tests/unit/reconstruction/lane_e/**` |
| INTEGRATOR | After B-E merge: `trueframe.py`, the revert-only validation pass, golden regeneration, `tests/integration/test_trueframe.py` | see `scripts/lanes.json` |
| F | Only after the integrator is green: `adapters/live/yolo.py`, `adapters/fixture/yolo.py`, `tests/contract/test_detector_tracker.py` | see `scripts/lanes.json` |

Every lane also owns `notes/lane-<x>.md` (interpretations, status) and `notes/lane-<x>-requests.md`
(changes it needs in files it does not own). `scripts/lanes.json` is the machine-readable
allowlist; the checker enforces it.

## Frozen for every lane

- Person 1 contract: `src/probity/domain/**`, `src/probity/ports.py`, `config/**`,
  `contracts/**`, `fixtures/schema/**`, `fixtures/demo/manifest.json` and `source/`,
  `pyproject.toml`, `uv.lock`.
- Person 2 shared interfaces: `reconstruction/types.py` (all internal dataclasses),
  `tests/unit/reconstruction/conftest.py`, `tests/unit/reconstruction/test_guards.py`,
  `eval/synth.py`, `eval/window.py`, and the step 1-3 modules `decisions.py`, `determinism.py`,
  `io.py`, `quality.py`, `tracking.py`, `preflight.py`.
- Fixtures: `fixtures/synthetic/**` and `fixtures/demo/**` (pinned by `scripts/golden_hashes.json`).
- Public signatures of every lane module stub (pinned by `scripts/frozen_hashes.json`). A lane
  implements the bodies and may add private helpers or new public functions; it may not rename,
  remove, or re-type a frozen one.

Need a change to a frozen file? Write the exact change in `notes/lane-<x>-requests.md` and keep
working around it. The integrator decides, edits once, and regenerates the hashes.

## Interfaces (`src/probity/reconstruction/types.py`)

Images are BGR uint8 `(H, W, 3)`; boxes are full-frame half-open `(x1, y1, x2, y2)`; `*_crop`
arrays are in the target expanded-box frame; matrices are float64 3x3 donor->target full-frame;
`source_x/source_y` are float32 donor full-frame coordinates. The flow:

```
preflight (done)  -> Obs + QualityScores per target/donor        [integrator builds Obs]
align.align       -> Alignment (gates, matrix, A)                [lane B]
align.warp_donor  -> Warp (crop, valid_mask, source_x/y)         [lane B]
color.fit_color   -> ColorFit; color.apply_color -> warped_crop  [lane B]
fuse.temporal_preference / rank_score / rank_donors -> AlignedDonor(T, R), LUT order [lane C]
fuse.fuse_tiles   -> FusionResult(result_frame, TileDecisions, donor_lut_order)     [lane C]
provenance.build_source_lut / build_provenance_arrays / provenance_record         [lane D]
integrity.compute_integrity / integrity_gate                                         [lane D]
baseline.* (evaluation only, never imported by the reconstruction path)              [lane E]
eval.metrics.*                                                                         [lane E]
```

Lanes return `Gate`s (in `Alignment.gates`, `ColorFit.gates`, `fuse.tile_gates`,
`integrity.integrity_gate`); the integrator records them through `DecisionLog.check/apply`, so
every material accept/reject becomes a rule-coded `PolicyDecision`.

## Isolation rule

Lane tests live in their lane folder and use only hand-built or seeded inputs plus the read-only
fixtures from `conftest.py` (`policy`, `translate_window`, `single_donor_window`,
`translate_images`, `single_donor_images`, `golden_completed`, `golden_refused`, `rng`,
`make_obs`). They never call another lane's implementation (they may use its types), never use the
network, never download models, and use deterministic seeds. Each lane folder keeps its
`__init__.py` so test module names stay unique.

## Before every merge

```
uv run python scripts/check_lane.py --lane <x> --base person2/trueframe
```

It fails if any change since the merge base (committed, staged, unstaged, or untracked) is
outside the lane allowlist, if any frozen file changed, if golden or frozen hashes or signatures
mismatch, or if the full `uv run pytest -q` is not green. `--skip-tests` is for debugging and
always reports failure. `--write-hashes` is integrator-only (the hash files are frozen for lanes,
so a lane that runs it still fails the scope check).

## Line endings

`core.autocrlf=true` on this machine and the repo has no root `.gitattributes`. The checker and
guards hash text files with CRLF normalized to LF, and `fixtures/synthetic/.gitattributes`
(`* -text`) keeps golden bytes identical in every worktree. Root `.gitattributes` is still
requested from the repo owner (`NOTES-person2.md`).

## Setup

```
powershell -ExecutionPolicy Bypass -File scripts\make_worktrees.ps1
```

Sets `core.longpaths`, creates `person2/trueframe` from HEAD if missing, adds the B-E worktrees,
runs `uv sync --frozen --extra recon` in each (OpenCV lives in the `recon` extra), and warns if the
checkout is inside OneDrive.
