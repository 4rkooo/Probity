"""Fixture NVIDIA Cosmos adapter: verified descriptions and hashed embeddings."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from probity.domain.enums import AdapterMode, HealthStatus, InferenceMode
from probity.domain.errors import FixtureNotFound, ValidationFailed
from probity.domain.fixtures import FixtureCatalog
from probity.domain.ids import parse_segment_id, sha256_hex, utc_now
from probity.domain.models import AdapterHealth, Embedding, SegmentDescription
from probity.domain.prompts import COSMOS_INGESTION_PROMPT, COSMOS_INGESTION_PROMPT_SHA256
from probity.search.embedding import (
    EMBED_DIM,
    as_unit_tuple,
    embedding_input_text,
    fixture_text_embedding,
    input_sha256,
)

DESCRIBE_MODEL_ID = "fixture/cosmos-reason-describe-v1"
EMBED_MODEL_ID = "fixture/cosmos-embed-v1"
FIXTURE_ID = "demo-plate-90s"


def _file_sha(path: Path) -> str:
    return sha256_hex(path.read_bytes())


class FixtureVideoUnderstanding:
    """VideoUnderstanding + Adapter. ``adapter_name='cosmos-fixture'``, mode FIXTURE."""

    adapter_name = "cosmos-fixture"
    schema_version = "1.0"

    def __init__(
        self,
        fixture_root: Path | str,
        catalog_path: Path | str | None = None,
        *,
        allow_synthetic: bool = False,
    ) -> None:
        self._root = Path(fixture_root)
        if catalog_path is not None:
            self._catalog_path = Path(catalog_path)
        else:
            self._catalog_path = self._root / "manifest.json"
        self._search_dir = self._root / "search"
        self._hashes_ok = False
        self._detail: str | None = "fixture search files not verified"
        self._catalog: FixtureCatalog | None = None
        self._by_key: dict[tuple[str, int, int, int], dict[str, Any]] = {}
        self._vectors: np.ndarray | None = None
        self._refs: dict[str, Any] = {}
        self._allow_synthetic = allow_synthetic
        self._load()

    @property
    def model_id(self) -> str:
        return DESCRIBE_MODEL_ID

    @property
    def mode(self) -> AdapterMode:
        return AdapterMode.FIXTURE

    def _load(self) -> None:
        try:
            if self._catalog_path.is_file():
                raw = self._catalog_path.read_text(encoding="utf-8")
                self._catalog = FixtureCatalog.model_validate_json(raw)
            desc_path = self._search_dir / "descriptions.jsonl"
            npy_path = self._search_dir / "embeddings.npy"
            refs_path = self._search_dir / "embedding_refs.json"
            if not (desc_path.is_file() and npy_path.is_file() and refs_path.is_file()):
                self._detail = "missing fixtures/demo/search files"
                return
            refs = json.loads(refs_path.read_text(encoding="utf-8"))
            desc_sha = _file_sha(desc_path)
            npy_sha = _file_sha(npy_path)
            desc_ok = refs.get("descriptions_sha256") == desc_sha
            npy_ok = refs.get("embeddings_sha256") == npy_sha
            if not desc_ok or not npy_ok:
                self._detail = "search fixture hash mismatch"
                return
            if refs.get("prompt_sha256") != COSMOS_INGESTION_PROMPT_SHA256:
                self._detail = "ingestion prompt hash mismatch"
                return
            if self._catalog is not None:
                source_sha = self._catalog.fixtures[0].source.sha256
                if refs.get("source_sha256") != source_sha:
                    self._detail = "source_sha256 does not match catalog"
                    return
                expected = {f.path: f.sha256 for f in self._catalog.fixtures[0].files}
                for rel, digest in (
                    ("search/descriptions.jsonl", desc_sha),
                    ("search/embeddings.npy", npy_sha),
                    ("search/embedding_refs.json", _file_sha(refs_path)),
                ):
                    if rel in expected and expected[rel] != digest:
                        self._detail = f"catalog hash mismatch for {rel}"
                        return
            vectors = np.load(npy_path)
            if vectors.dtype != np.float32 or vectors.ndim != 2 or vectors.shape[1] != EMBED_DIM:
                self._detail = "embeddings.npy must be float32 [N,64]"
                return
            norms = np.linalg.norm(vectors, axis=1)
            if not np.all(np.abs(norms - 1.0) <= 1e-4):
                self._detail = "embeddings.npy rows must be L2-normalized"
                return
            records: dict[tuple[str, int, int, int], dict[str, Any]] = {}
            with desc_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    key = (
                        str(row["source_sha256"]),
                        int(row["ordinal"]),
                        int(row["start_pts_us"]),
                        int(row["end_pts_us"]),
                    )
                    records[key] = row
            if len(records) != vectors.shape[0]:
                self._detail = "description count does not match embeddings.npy rows"
                return
            self._by_key = records
            self._vectors = vectors
            self._refs = refs
            self._hashes_ok = True
            self._detail = None
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
            self._hashes_ok = False
            self._detail = f"fixture load failed: {exc}"

    async def health(self) -> AdapterHealth:
        started = time.perf_counter()
        status = HealthStatus.OK if self._hashes_ok else HealthStatus.UNAVAILABLE
        return AdapterHealth(
            adapter_name=self.adapter_name,
            model_id=self.model_id,
            mode=self.mode,
            status=status,
            checked_at=utc_now(),
            latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
            detail=self._detail,
        )

    def _require_ready(self) -> None:
        if not self._hashes_ok:
            raise FixtureNotFound(self._detail or "fixture search cache is not verified")

    def _lookup(
        self, source_sha256: str, ordinal: int, start_pts_us: int, end_pts_us: int
    ) -> dict[str, Any]:
        self._require_ready()
        key = (source_sha256, ordinal, start_pts_us, end_pts_us)
        row = self._by_key.get(key)
        if row is None:
            if self._allow_synthetic:
                return self._synthetic_row(source_sha256, ordinal, start_pts_us, end_pts_us)
            raise FixtureNotFound(
                "no fixture description for "
                f"source_sha256={source_sha256} ordinal={ordinal} "
                f"pts=[{start_pts_us},{end_pts_us})"
            )
        return row

    def _synthetic_row(
        self, source_sha256: str, ordinal: int, start_pts_us: int, end_pts_us: int
    ) -> dict[str, Any]:
        """Deterministic placeholder description for custom uploads (demo only)."""
        start_s = start_pts_us / 1_000_000
        end_s = end_pts_us / 1_000_000
        summary = (
            f"Custom upload segment {ordinal}: observable scene content between "
            f"{start_s:.1f}s and {end_s:.1f}s. Synthetic description for demo ingest "
            f"(no live Cosmos). Source {source_sha256[:12]}."
        )
        search_terms = ["custom", "upload", "video", "segment", f"s{ordinal:04d}"]
        raw = json.dumps(
            {
                "summary": summary,
                "ordinal": ordinal,
                "start_pts_us": start_pts_us,
                "end_pts_us": end_pts_us,
                "source_sha256": source_sha256,
            },
            sort_keys=True,
        )
        return {
            "source_sha256": source_sha256,
            "ordinal": ordinal,
            "start_pts_us": start_pts_us,
            "end_pts_us": end_pts_us,
            "model_id": DESCRIBE_MODEL_ID,
            "prompt_sha256": COSMOS_INGESTION_PROMPT_SHA256,
            "raw_output_sha256": sha256_hex(raw.encode("utf-8")),
            "summary": summary,
            "rigid_subjects": [],
            "actions": [],
            "visibility": [
                {
                    "kind": "CLEAR",
                    "observable_source": None,
                    "confidence": 0.5,
                    "time_range": None,
                }
            ],
            "scene_changes": [],
            "search_terms": search_terms,
            "uncertainty": [
                "Synthetic description generated locally because this upload is not in the "
                "verified fixture catalog and live Cosmos is unavailable."
            ],
            "confidence": 0.5,
            "mode": InferenceMode.FIXTURE.value,
        }

    def _to_description(
        self, row: dict[str, Any], video_id: str, segment_id: str
    ) -> SegmentDescription:
        payload = dict(row)
        payload["video_id"] = video_id
        payload["segment_id"] = segment_id
        payload["time_range"] = {
            "start_pts_us": row["start_pts_us"],
            "end_pts_us": row["end_pts_us"],
        }
        payload.pop("ordinal", None)
        payload.pop("start_pts_us", None)
        payload.pop("end_pts_us", None)
        payload.setdefault("mode", InferenceMode.FIXTURE.value)
        payload.setdefault("prompt_sha256", COSMOS_INGESTION_PROMPT_SHA256)
        return SegmentDescription.model_validate(payload)

    async def describe(self, clip: Any, prompt: str) -> SegmentDescription:
        if prompt != COSMOS_INGESTION_PROMPT:
            raise ValidationFailed("describe requires the exact Cosmos ingestion prompt")
        row = self._lookup(clip.source_sha256, clip.ordinal, clip.start_pts_us, clip.end_pts_us)
        expected_id = f"{clip.video_id}:s{clip.ordinal:04d}"
        segment_id = clip.segment_id if clip.segment_id == expected_id else expected_id
        return self._to_description(row, clip.video_id, segment_id)

    def _embed_text(self, text: str, ref: str) -> Embedding:
        vector = fixture_text_embedding(text, dim=EMBED_DIM)
        return Embedding(
            embedding_ref=ref,
            model_id=EMBED_MODEL_ID,
            dimension=EMBED_DIM,
            vector=as_unit_tuple(vector),
            normalized=True,
            input_sha256=input_sha256(text),
            mode=InferenceMode.FIXTURE,
        )

    async def embed_segments(self, items: Sequence[SegmentDescription]) -> Sequence[Embedding]:
        self._require_ready()
        out: list[Embedding] = []
        for item in items:
            known = any(item.source_sha256 == key[0] for key in self._by_key)
            if not known and not self._allow_synthetic:
                raise FixtureNotFound(f"unknown source_sha256 {item.source_sha256}")
            ordinal = parse_segment_id(item.segment_id)[1]
            text = embedding_input_text(item.summary, item.search_terms)
            ref = (
                f"{FIXTURE_ID}/embeddings.npy#{ordinal}"
                if known
                else f"synthetic/{item.source_sha256[:16]}#{ordinal}"
            )
            out.append(self._embed_text(text, ref))
        return tuple(out)

    async def embed_query(self, query: str) -> Embedding:
        self._require_ready()
        return self._embed_text(query, "query")
