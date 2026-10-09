# Lane F requests

Changes this lane needs in files it does not own. Never edit those files; add a row and keep
working around it.

| # | File | Exact change requested | Why | Status |
| --- | --- | --- | --- | --- |
| 1 | `pyproject.toml` (Person 1) | Optional extra `live-yolo = ["ultralytics>=8.3,<9", "lap>=0.5.12"]` plus the existing `[tool.uv] override-dependencies` so ultralytics cannot pull `opencv-python` | Local YOLO/ByteTrack only. Import stays optional; tests skip without it. Never download weights. | already on `person2/requests-for-p1` (`c1f8c0e`); not merged here |
| 2 | `src/probity/domain/errors.py` | Optional: a first-class `NotConfigured` error code if Person 1 wants it distinct from `SPONSOR_UNAVAILABLE` | Live adapter currently subclasses `ProbityError` with `SPONSOR_UNAVAILABLE` because domain is frozen | working around |
