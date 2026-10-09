# Lane F notes: YOLO/ByteTrack live and fixture adapters

Owner files: `src/probity/adapters/live/yolo.py`, `src/probity/adapters/fixture/yolo.py`,
`tests/contract/test_detector_tracker.py`, `tests/unit/reconstruction/lane_f/**`.
Contract: `docs/person2-lanes.md`.

## Interpretations

1. **One protocol, two adapters.** `FixtureYoloAdapter` and `LiveYoloAdapter` both satisfy
   `DetectorTracker`. The contract test is parametrized over both. Fixture always runs against
   `fixtures/synthetic/plate_translate_v1`. Live inference skips unless ultralytics and a local
   checkpoint file are present; it never downloads weights.
2. **Fixture reads stored records only.** Detections come from `detections.json` / `.jsonl`;
   tracks from `track.json`, `tracks.json`, `track_confirmed.json`, `track_not_confirmed.json`,
   or `tracks/*.json`. No network, no weights, no `ground_truth/` (OCR lives there). Output order
   is `(video_id, frame_number, detection_id)` / `(video_id, track_id)`.
3. **`detect` filters.** A detection is returned only when its `frame_id` is in the requested
   frames and `class_name` is in `classes`. An empty class set returns no rows.
4. **`track` is a lookup, not a recompute.** Match `track_id`, else `(video_id, seed_frame_id,
   seed_bbox_px)`. Missing records are `FixtureNotFound`. Identity gates stay in
   `reconstruction/tracking.py`; this adapter does not re-run them.
5. **No plate characters.** Domain `Detection` / `Track` have no OCR fields. The adapters never
   copy `quoted_text` or ground-truth plate strings. `class_name` is a taxonomy label
   (`license_plate`), not rendered plate text.
6. **Live ultralytics is lazy.** The module does not import ultralytics at import time.
   `find_spec` is used for health. `YOLO(...)` is called only after `weights_path.is_file()`.
   A hub name such as `yolov8n.pt` with no local file is `NotConfigured`.
7. **CoreWeave is NotConfigured.** `coreweave=True` or `run_on_coreweave()` raises
   `NotConfigured` (`SPONSOR_UNAVAILABLE`). No endpoint, queue URL, or API path is defined on
   this branch; none is invented.
8. **`NotConfigured` lives on the live adapter** (domain/errors.py is frozen). It is a
   `ProbityError` with `SPONSOR_UNAVAILABLE`. `health()` never raises; it reports
   `DISABLED` (missing extra/weights) or `UNAVAILABLE` (CoreWeave requested).
9. **Live `track` binds frames at construction.** The protocol `track(request)` has no frame
   list. Local inference needs a `frame_loader` and `frames`; otherwise `ValidationFailed`.
   CSRT bridging is omitted when opencv-contrib is absent (same as NOTES-person2 #17).
10. **Frame entity folders.** Any frame path this lane would write uses `f{frame_number}`
    (`:` is an NTFS stream). This lane does not write frames; it only reads committed fixtures.
11. **`live-yolo` extra is not on this branch.** Requested of Person 1; already on
    `person2/requests-for-p1` (`c1f8c0e`). See `notes/lane-f-requests.md`.

## Status

- [x] Fixture adapter replays synthetic goldens and contract tracks
- [x] Live adapter lazy-imports ultralytics; missing extra/weights -> NotConfigured / skip
- [x] CoreWeave remote path is NotConfigured (no guessed URLs)
- [x] Shared contract test for live + fixture
- [x] No OCR / plate characters
- [x] Live `detect` order is `(video_id, frame_number, detection_id)` (not string `frame_id`)
- [x] Empty class set, seed-bbox track lookup, bound-frame ValidationFailed, and no-import tests
