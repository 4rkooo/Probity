from __future__ import annotations

import json
from pathlib import Path

from probity.api.main import create_app

ROOT = Path(__file__).resolve().parents[2]


def test_openapi_matches_committed_snapshot() -> None:
    committed = json.loads((ROOT / "contracts/openapi.json").read_text())
    assert create_app().openapi() == committed, (
        "OpenAPI drifted from contracts/openapi.json; contract changes need coordinator review"
    )


def test_every_mutating_route_requires_idempotency_key() -> None:
    spec = json.loads((ROOT / "contracts/openapi.json").read_text())
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            if method == "post":
                headers = {p["name"] for p in op.get("parameters", []) if p.get("in") == "header"}
                assert "Idempotency-Key" in headers, path


def test_exported_contract_artifacts_have_no_drift() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_contracts.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout
