# Lane E requests

Changes this lane needs in files it does not own (types.py, conftest.py, quality.py, policy YAML,
pyproject, another lane's module). Never edit those files; add a row and keep working around it.

| # | File | Exact change requested | Why | Status |
| --- | --- | --- | --- | --- |
| 1 | `pyproject.toml` (Person 1) | Live baseline only: an optional `baseline = [...]` extra for Real-ESRGAN; tests never need it | Real-ESRGAN x4plus runtime | not yet needed |
