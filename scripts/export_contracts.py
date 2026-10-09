"""Export frozen contract artifacts: OpenAPI snapshot and JSON Schemas.

Usage: uv run python scripts/export_contracts.py [--check]
--check exits non-zero if any committed artifact differs from the generated output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel

from probity.api.main import create_app
from probity.domain import models
from probity.domain.fixtures import FixtureCatalog, FixtureManifest
from probity.domain.policy import PolicyConfig

ROOT = Path(__file__).resolve().parents[1]

SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    name: getattr(models, name)
    for name in (
        "CaseWorkspace",
        "SourceVideo",
        "FrameManifest",
        "FrameReference",
        "VideoSegment",
        "SegmentDescription",
        "Embedding",
        "Detection",
        "Track",
        "SearchEvidence",
        "PolicyDecision",
        "PixelProvenance",
        "PixelOrigin",
        "ReconstructionRun",
        "HumanReview",
        "EvidenceReport",
        "LineageRecord",
        "JobView",
        "JobEvent",
        "AssetRef",
        "AdapterHealth",
    )
}


def render(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def artifacts() -> dict[Path, str]:
    out: dict[Path, str] = {ROOT / "contracts/openapi.json": render(create_app().openapi())}
    for name, model in SCHEMA_MODELS.items():
        out[ROOT / f"contracts/schemas/{name}.schema.json"] = render(model.model_json_schema())
    out[ROOT / "contracts/schemas/PolicyConfig.schema.json"] = render(
        PolicyConfig.model_json_schema()
    )
    out[ROOT / "fixtures/schema/manifest.schema.json"] = render(FixtureManifest.model_json_schema())
    out[ROOT / "fixtures/schema/catalog.schema.json"] = render(FixtureCatalog.model_json_schema())
    return out


def main(check: bool) -> int:
    drift = []
    for path, text in artifacts().items():
        if check:
            if not path.exists() or path.read_text() != text:
                drift.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    if drift:
        print("Contract drift:\n  " + "\n  ".join(drift))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
