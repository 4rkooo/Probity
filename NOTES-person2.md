# Person 2 decision log (TrueFrame, provenance, integrity, eval)

Each entry: the ambiguity, the reading taken (conservative / refuse-leaning where it matters), and
where it lives. `PROBITY_TECHNICAL_DESIGN.md` stays authoritative; anything here that turns out to
contradict it is a bug in this file.

## Open requests to other owners

| To | Request | Status |
| --- | --- | --- |
| Person 1 | `pyproject.toml`: `recon = ["opencv-contrib-python-headless>=4.10,<5", "scikit-image>=0.24,<0.25"]`, `live-yolo = ["ultralytics>=8.3,<9", "lap>=0.5.12"]`, and `[tool.uv] override-dependencies = ["opencv-python; sys_platform == 'never'", "opencv-python-headless; sys_platform == 'never'"]` (CSRT needs contrib; ultralytics pulls the non-headless wheel) | pending |
| Person 1 | Regenerate the demo clip so frame 417 is the blurriest plate view and at least 3 sharper detector-backed frames fall inside +/-2 s; keep the bit-exact encode flags | pending |
| Person 1 | Register `fixtures/demo/frames/` as frame_png; frame entity folders are `f{frame_number}` (':' is an NTFS stream separator) | pending |
| Person 1 | Confirm the worker does not re-register reconstruction AssetRefs I emit; reuse my `pixel_sha256` definition (below) for extracted frames | pending |
| Team | Add to `config/policy.demo.yaml` (and `PolicyConfig`): `alignment.scale_ratio_min: 0.67`, `alignment.scale_ratio_max: 1.50`, `ransac.max_iters: 2000`, `ransac.confidence: 0.995`, `ransac.rng_seed: 20261009`, `clahe.clip_limit: 2.0`, `clahe.tile_grid: 4`, `akaze.detector_threshold: 0.001`, `ecc.gauss_filt_size: 5`. `config_sha256` changes | pending |
| User | `.gitattributes`: `*.png binary`, `*.npz binary`, `*.npy binary`, `*.mp4 binary`, `*.json text eol=lf` | pending |
| User | ffmpeg on PATH (`winget install Gyan.FFmpeg`), needed for demo frame extraction | pending |

## Findings to share

- **AKAZE needs a large crop.** On the expanded plate crop, AKAZE returns 0 keypoints at 76x26 px
  and about 47 keypoints (35 good matches) at 220x72 px. The demo sedan plate
  (`[216,470,292,496)`, 76x26) will therefore always take the ECC fallback
  (`TOO_FEW_FEATURES`). This is expected behaviour, not a bug; the demo narrative should say so.
- **Thin-strip homographies are ill-conditioned.** A full homography fitted to a plate strip can
  show good inlier ratios while its translation terms drift. Step 4 relies on the corner-outside
  and scale gates to catch this; nothing is loosened.
- **Sharpest donor vs. residual gate.** Where a donor is much sharper than a blurred target, its
  8-bit residual against the target is larger, so the tile residual gate can hand the tile to a
  less sharp donor. This is the spec's behaviour (residual is a hard gate, then score).
- **Tile seams are visible before validation.** Winner-take-all tiles from different donors show
  block edges; the step-8 boundary-discontinuity pass may only revert such tiles.

## Interpretations

### Quality (section 9)

1. `crop.alignment_context_expand = 0.15` means the box grows 15 % in total (7.5 % per side),
   floor/ceil to integers, clipped to the frame. `quality.expand_box`.
2. Resizing to `resize_long_edge_px`: `INTER_AREA` when shrinking, `INTER_LINEAR` when enlarging.
3. Laplacian is `cv2.Laplacian(luma_float64, CV_64F, ksize=1)` on luma in [0, 255].
4. **Tile sharpness is measured on the same resized grid as S** (tile rectangles scaled into the
   160-px long-edge crop), normalised by `laplacian_variance_normalizer`. At native resolution
   the score saturates on almost every tile and the keep-original gate becomes meaningless.
5. Exposure delta (stops) = `|log2(mean_luma_donor / mean_luma_target)|` over the subject boxes,
   with each mean floored at 1.0 (8-bit) to avoid log(0).
6. Scale ratio = `sqrt(area_donor / area_target)`; aspect change = `|aspect_d / aspect_t - 1|`;
   both are measured against the target box, not the track median, because donors map to the target.
7. P (pose) uses the track median aspect of accepted observations.

### Preflight and decisions

8. Gate order per donor: window, scale min, scale max, aspect, obstruction, donor quality floor,
   quality gain, exposure. A rejected donor records only its first failing gate (one REJECT row);
   a passing donor records every gate as ACCEPT. Short-circuiting keeps the log readable and never
   hides a reject.
9. A bridged-only observation (`CSRT_BRIDGE`, `ANALYST_ROI` without a detection) cannot donate
   pixels; it is rejected as `IDENTITY_GEOMETRY_MISMATCH` (observed `CSRT_BRIDGE`, threshold `DETECTOR`).
10. `observed` is recorded at full precision (no rounding before comparison), so the decision row
    reproduces the comparison exactly. Integer observations stay integers.
11. Kept tiles get one aggregate `KEEP_ORIGINAL_CLEAR` row with counts per reason; each borrowed
    tile gets its own `BORROW_TILE_ACCEPTED` row referenced from that donor's LUT `decision_ids`.

### Track (step 2, `reconstruction/tracking.py`)

12. `continuity_score` = accepted detector-backed observations / sampled frames in the window.
    `mean_confidence` is rounded to 4 dp and the stored value is the one gated.
13. Appearance uses a hue-saturation histogram (30x32 bins, `HISTCMP_CORREL`) of the subject box
    against the seed crop; it ignores brightness, which the lighting gate handles. The obstructed
    synthetic frame scores 0.819 against 0.80, a thin but honest margin pinned by a test.
14. Identity gates, in order, each against the previous accepted observation (association IoU,
    center step) or the seed box (scale, aspect, appearance), then collision with any box of a
    different or no ByteTrack ID. Center step is per sampled step and is not scaled by elapsed
    frames (a bridged frame counts as the previous step).
15. Frozen reason codes: every identity-gate failure is `IDENTITY_GEOMETRY_MISMATCH`; observations
    after continuity breaks are `TRACK_NOT_CONFIRMED`. The specific gate key, value, and threshold
    live in `ObservationAudit`. Finer codes (collision, appearance, gap) would need interface review.
16. Gap rule: a sampled frame without an accepted detection is covered only by a bridge box that
    passes every identity gate. A lost or rejected bridge, a gap over
    `track.max_bridge_gap_frames`, or no bridger ends the track in that direction; later same-ID
    detections are recorded as rejected. Gaps count sampled frames, not source frames.
17. **Without CSRT (current environment: no opencv-contrib) the live tracker cannot cross any gap.**
    Fixture mode replays recorded bridge boxes and is unaffected.
18. Bridged observations carry `confidence = 0.0` (no detector evidence); they never count toward
    confirmation or `mean_confidence` and never donate (preflight rejects them).
19. Frames whose detection was rejected are not bridged in the record, but the bridger is stepped
    and must still produce a passing box for the gap to continue.
20. A ByteTrack ID switch is not followed: only the seed's ID is associated. Duplicate boxes with
    the seed's ID in one frame: the best IoU against the previous box wins, others are rejected.
21. Seeds: an explicit `seed_detection_id` must be in the seed frame and overlap `seed_bbox_px`
    (IoU >= 0.30), else `ValidationFailed`. An ROI-only seed is confirmed by the best-overlapping
    detection in the seed frame (becomes a DETECTOR seed). Otherwise it stays `ANALYST_ROI` with
    D = 0.50 and adopts the ByteTrack ID of the nearest overlapping detection within
    `max_bridge_gap_frames + 1` sampled frames; with none, the track is NOT_CONFIRMED. A seed
    box that collides with another box makes the track NOT_CONFIRMED.
22. Window radius = min(request `window_radius_us`, `track.window_radius_s`). Sampling stride =
    `ceil((1e6 / max_sample_fps) / (median_dt_us + 1))`, anchored at the seed, plus the seed and
    every frame with a stored detection; the +1 us absorbs integer PTS rounding (30 fps -> 2).
23. Frames or detections from another video raise `ValidationFailed` (a plumbing error, not a
    refusal); non-`FrameReference` inputs raise `NonEvidentiaryInput`.
24. Synthetic tracks are built by `confirm_track` from `tracker_inputs.json` (ByteTrack ID 1 for
    the single subject; ground-truth bridge boxes standing in for CSRT, disclosed in
    `bridge_source`); `tracker_version = probity-track-v1+synthetic-replay`.

### Hashing and files

25. `pixel_sha256` (`probity-pixel-v1`): SHA-256 over the ASCII header
    `probity-pixel-v1|uint8|{H},{W},3|RGB\n` followed by the RGB bytes (row-major, C-contiguous).
    Decoded pixels are hashed separately from file bytes.
26. npz files are written by our own zip writer (fixed 1980 timestamp, `create_system=3`,
    deflate level 6, sorted names). `np.savez` embeds wall-clock times and is not byte-stable.
27. File-byte SHA-256 is recorded for every artifact, but golden comparisons across machines use
    pixel / array digests, because zlib output may differ between builds. Byte equality is asserted
    only when the OpenCV version matches the one recorded in `truth.json`.
28. JSON is UTF-8 with `\n` line endings, 2-space indent, trailing newline, written via temp file
    + fsync + `os.replace` (handle closed first).

### Synthetic fixtures (`fixtures/synthetic/`)

29. Synthetic windows have no MP4. Their `source_sha256` is the canonical SHA-256 of
    `{"synth_version", "pixel_sha256": [ordered frame hashes]}`.
30. Ground truth (clean target, occlusion masks, plate text, true geometry) lives only under
    `ground_truth/`, is flagged `evaluation_only`, and is unreachable through the frame loader.
31. Scene is 480x288 @ 15 fps, 75 frames; plate 220x72 so AKAZE has features (finding above).
    Target sigma 2.5: at sigma 1.0-2.0 the quality gains were too small or the obstructed frame
    failed the quality floor first, which would have hidden the `DONOR_OBSTRUCTED` path.
32. **Fixture-design fix (not a threshold change):** the single-donor variant's background frames
    were blurred to 0.92-0.999x the target sigma. Sub-pixel plate phase then made some of them
    measure blurrier than the target, breaking "target is the blurriest". The range is now
    0.80-0.95x, still far below the 0.10 quality-gain threshold.

### Hand-built golden runs (step 1 only; removed in step 8)

33. `src/probity/eval/handbuilt_runs.py` builds the goldens with ground-truth translations in
    place of AKAZE/ECC. Every run carries the hand-built note in `uncertainty`, and every alignment
    decision says so. A uses inlier=1 and error=0, so `mean_alignment_confidence = 1.0` and the
    integrity score (96) is optimistic; step 8 replaces these files with real pipeline output.
34. Color in the hand-built run is identity gain/bias with the context-ring residual and sample
    gates still enforced. Audit completeness Au = 1.0 by construction.

### Preflight hard gates (step 3, `reconstruction/preflight.py`)

35. Track gates (all logged): run video and case match the track (`DONOR_TRACK_MISMATCH`), then
    state `CONFIRMED`, detector-backed count >= 5 and mean confidence >= 0.55 (all reject as
    `TRACK_NOT_CONFIRMED`). Each rejected track observation also gets a TRACK-stage REJECT row.
36. Target selection (first failure): the target frame must be a track observation
    (`TRACK_NOT_CONFIRMED`), detector-backed (`IDENTITY_GEOMETRY_MISMATCH`), and the request box
    must overlap the tracked box with IoU >= 0.30 (`IDENTITY_GEOMETRY_MISMATCH`). Target size and
    target Q are then all logged (`TARGET_TOO_SMALL`, `TARGET_QUALITY_TOO_LOW`).
37. Donor lineage runs before any pixel load, first failure: donor video == run video, donor is an
    accepted member of this track (`DONOR_TRACK_MISMATCH`, observed `NOT_MEMBER`), donor source is
    `DETECTOR`. A foreign-video or foreign-track candidate is therefore never decoded.
38. O_i = max(mask fraction, detector `occlusion_score`). The mask is the union of out-of-frame
    area and overlap with other detections in the same frame, excluding the subject's own
    detection and any detection that fully contains the subject box (its carrier, e.g. the car).
39. Inputs go through `assert_evidentiary_input` before loading. A generated/baseline asset is a
    `GENERATED_INPUT_REJECTED` row and a pixel-hash mismatch is `DECODE_NOT_DETERMINISTIC`; for the
    target that refuses the run, for a donor it rejects the donor. Neither is an exception.
40. Cancellation is checked once per donor candidate; `JobCancelled` propagates (cancellation is
    not a refusal).
41. Thresholds compare full-precision floats, so a computed gain like 0.60 - 0.50 sits just under
    0.10 and rejects. Real Q values are continuous; tests check just inside and just outside.
