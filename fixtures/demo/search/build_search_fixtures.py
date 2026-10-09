"""Deterministic search fixtures for demo-plate-90s (Agent C).

Keyed by (source_sha256, ordinal, start/end pts) so descriptions can be re-id'd to any video_id.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from probity.adapters.fixture.cosmos import EMBED_MODEL_ID, FIXTURE_ID, FixtureVideoUnderstanding
from probity.adapters.fixture.vast import FixtureEvidenceStore
from probity.domain.ids import parse_segment_id, sha256_hex
from probity.domain.models import Embedding, SearchPlan, SegmentDescription, SourceVideo, VideoSegment
from probity.domain.policy import default_policy
from probity.domain.prompts import COSMOS_INGESTION_PROMPT_SHA256
from probity.ports import QueryPlanRequest, SearchQuery
from probity.search.embedding import EMBED_DIM, embedding_input_text, fixture_text_embedding, input_sha256
from probity.search.plan import RuleBasedPlanner
from probity.search.service import LocalSearchService

ROOT = Path(__file__).resolve().parents[3]
CONTRACTS = ROOT / "fixtures" / "contracts"
OUT = Path(__file__).resolve().parent
SOURCE_SHA256 = "9806895754d3af16f50ba66a99514f4b7c1b6659b278fdf7f8260cfd12bf05e5"
PREPARED_QUERY = "Find the blue sedan when its rear plate is most visible"
CLARIFY_QUERY = "Who is the guilty driver and why did they flee?"
CORRELATION_ID = "01a12166-60c0-7e9e-8227-4170b5a9bc3d"


def _sha(path: Path) -> str:
    return sha256_hex(path.read_bytes())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_descriptions = json.loads((CONTRACTS / "segment_descriptions.json").read_text(encoding="utf-8"))
    descriptions = [SegmentDescription.model_validate(row) for row in raw_descriptions]
    if len(descriptions) != 15:
        raise SystemExit(f"expected 15 segment descriptions, got {len(descriptions)}")

    jsonl_path = OUT / "descriptions.jsonl"
    lines: list[str] = []
    texts: list[str] = []
    refs: list[dict[str, object]] = []
    matrix = np.zeros((len(descriptions), EMBED_DIM), dtype=np.float32)
    for desc in descriptions:
        _video_id, ordinal = parse_segment_id(desc.segment_id)
        if desc.source_sha256 != SOURCE_SHA256:
            raise SystemExit("source_sha256 mismatch in contract descriptions")
        record = {
            "source_sha256": desc.source_sha256,
            "ordinal": ordinal,
            "start_pts_us": desc.time_range.start_pts_us,
            "end_pts_us": desc.time_range.end_pts_us,
            "model_id": desc.model_id,
            "prompt_sha256": desc.prompt_sha256,
            "raw_output_sha256": desc.raw_output_sha256,
            "summary": desc.summary,
            "rigid_subjects": [item.model_dump(mode="json") for item in desc.rigid_subjects],
            "actions": [item.model_dump(mode="json") for item in desc.actions],
            "visibility": [item.model_dump(mode="json") for item in desc.visibility],
            "scene_changes": list(desc.scene_changes),
            "search_terms": list(desc.search_terms),
            "uncertainty": list(desc.uncertainty),
            "confidence": desc.confidence,
            "mode": desc.mode.value,
        }
        if record["prompt_sha256"] != COSMOS_INGESTION_PROMPT_SHA256:
            raise SystemExit("prompt_sha256 drifted from COSMOS_INGESTION_PROMPT_SHA256")
        lines.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        text = embedding_input_text(desc.summary, desc.search_terms)
        texts.append(text)
        vector = fixture_text_embedding(text, dim=EMBED_DIM)
        matrix[ordinal] = vector
        refs.append(
            {
                "ordinal": ordinal,
                "start_pts_us": desc.time_range.start_pts_us,
                "end_pts_us": desc.time_range.end_pts_us,
                "embedding_ref": f"{FIXTURE_ID}/embeddings.npy#{ordinal}",
                "input_sha256": input_sha256(text),
            }
        )

    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    npy_path = OUT / "embeddings.npy"
    np.save(npy_path, matrix)
    desc_sha = _sha(jsonl_path)
    npy_sha = _sha(npy_path)
    refs_payload = {
        "fixture_id": FIXTURE_ID,
        "source_sha256": SOURCE_SHA256,
        "model_id": EMBED_MODEL_ID,
        "dimension": EMBED_DIM,
        "normalized": True,
        "prompt_sha256": COSMOS_INGESTION_PROMPT_SHA256,
        "embeddings_path": "search/embeddings.npy",
        "embeddings_sha256": npy_sha,
        "descriptions_path": "search/descriptions.jsonl",
        "descriptions_sha256": desc_sha,
        "count": len(descriptions),
        "refs": refs,
    }
    refs_path = OUT / "embedding_refs.json"
    refs_path.write_text(json.dumps(refs_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    policy = default_policy()
    understanding = FixtureVideoUnderstanding(ROOT / "fixtures" / "demo", ROOT / "fixtures" / "demo" / "manifest.json")
    store = FixtureEvidenceStore()
    video = SourceVideo.model_validate_json((CONTRACTS / "source_video.json").read_text(encoding="utf-8"))
    segments = [
        VideoSegment.model_validate(row)
        for row in json.loads((CONTRACTS / "video_segments.json").read_text(encoding="utf-8"))
    ]
    embeddings: list[Embedding] = []
    for segment, desc in zip(segments, descriptions, strict=True):
        ordinal = segment.ordinal
        embeddings.append(
            Embedding(
                embedding_ref=f"{FIXTURE_ID}/embeddings.npy#{ordinal}",
                model_id=EMBED_MODEL_ID,
                dimension=EMBED_DIM,
                vector=tuple(float(x) for x in matrix[ordinal].tolist()),
                normalized=True,
                input_sha256=input_sha256(texts[ordinal]),
                mode=desc.mode,
            )
        )
    async def _index() -> None:
        await store.upsert_segments(segments, embeddings)

    asyncio.run(_index())

    planner = RuleBasedPlanner(policy)

    prepared_plan = planner.plan(
        QueryPlanRequest(
            query=PREPARED_QUERY,
            video_id=video.video_id,
            allowed_subject_classes=("car", "truck", "license_plate", "sign"),
            indexed_range=video.indexed_range,
        )
    )
    clarify_plan = planner.plan(
        QueryPlanRequest(
            query=CLARIFY_QUERY,
            video_id=video.video_id,
            allowed_subject_classes=("car", "truck", "license_plate", "sign"),
            indexed_range=video.indexed_range,
        )
    )
    service = LocalSearchService(understanding, store, policy)

    async def _search():
        return await service.search(
            video, SearchQuery(query=PREPARED_QUERY, max_results=5), CORRELATION_ID
        )

    evidence = asyncio.run(_search())
    golden = []
    for result in evidence.results:
        _vid, ordinal = parse_segment_id(result.segment_id)
        golden.append(
            {
                "ordinal": ordinal,
                "segment_suffix": f"s{ordinal:04d}",
                "score": result.score,
                "similarity": next(
                    c.similarity for c in evidence.candidates if c.segment_id == result.segment_id
                ),
                "components": result.components.model_dump(mode="json"),
                "start_pts_us": result.start_pts_us,
                "end_pts_us": result.end_pts_us,
                "explanation": result.explanation,
            }
        )
    queries = {
        "schema_version": "1.0",
        "source_sha256": SOURCE_SHA256,
        "prepared_query": {
            "query": PREPARED_QUERY,
            "expected_plan": _plan_dump(prepared_plan),
            "golden_ranks": golden,
        },
        "clarification_query": {
            "query": CLARIFY_QUERY,
            "expected_plan": _plan_dump(clarify_plan),
        },
    }
    queries_path = OUT / "queries.json"
    queries_path.write_text(json.dumps(queries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    files = [
        jsonl_path,
        npy_path,
        refs_path,
        queries_path,
        Path(__file__),
    ]
    print("search fixture sha256s:")
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name
        print(f"  {rel}: {digest} ({path.stat().st_size} bytes)")
    print("prepared golden ordinals:", [row["ordinal"] for row in golden])
    print("prepared plan:", json.dumps(_plan_dump(prepared_plan)))
    print("clarify plan:", json.dumps(_plan_dump(clarify_plan)))


def _plan_dump(plan: SearchPlan) -> dict[str, object]:
    return plan.model_dump(mode="json")


if __name__ == "__main__":
    sys.exit(main() or 0)
