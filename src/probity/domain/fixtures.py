"""Fixture manifest schema. A fixture adapter only serves data whose manifest hashes verify."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, model_validator

from probity.domain.models import FrozenModel, ModelId, Sha256Hex, UtcTimestamp


def _safe_relative(value: str) -> str:
    if value.startswith("/") or ".." in value.split("/"):
        raise ValueError("fixture paths must be relative and must not traverse upward")
    return value


FixturePath = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9._/-]+$", description="Relative to fixtures/demo/"),
    AfterValidator(_safe_relative),
]


class FixtureSource(FrozenModel):
    path: FixturePath
    sha256: Sha256Hex
    byte_length: Annotated[int, Field(gt=0)]
    duration_us: Annotated[int, Field(gt=0)]
    license_note: Annotated[str, Field(min_length=1, max_length=500)]
    synthetic: bool
    generator: Annotated[str, Field(max_length=500)] | None = None


class FixtureModelRef(FrozenModel):
    role: Literal[
        "cosmos_describe", "cosmos_embed", "yolo_detect", "bytetrack", "wandb_plan", "wandb_explain"
    ]
    model_id: ModelId
    model_sha256: Sha256Hex | None = None
    embedding_dimension: Annotated[int, Field(gt=0)] | None = None


class FixtureFile(FrozenModel):
    path: FixturePath
    sha256: Sha256Hex
    kind: Literal[
        "segments_jsonl",
        "descriptions_jsonl",
        "embeddings_npy",
        "embedding_refs_json",
        "detections_jsonl",
        "tracks_json",
        "search_plans_json",
        "expected_ranks_json",
        "thumbnail",
        "frame_png",
        "reconstruction_json",
        "provenance_npz",
        "other",
    ]


class FixtureManifest(FrozenModel):
    manifest_version: Literal["1.0"] = "1.0"
    schema_version: Literal["1.0"] = "1.0"
    fixture_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")]
    display_name: Annotated[str, Field(min_length=1, max_length=120)]
    source: FixtureSource
    policy_config_sha256: Sha256Hex
    ingestion_prompt_sha256: Sha256Hex
    models: tuple[FixtureModelRef, ...]
    files: tuple[FixtureFile, ...]
    created_at: UtcTimestamp

    @model_validator(mode="after")
    def _unique_paths(self) -> Self:
        paths = [f.path for f in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate fixture file path")
        return self


class FixtureCatalog(FrozenModel):
    """``fixtures/demo/manifest.json``: all verified demo fixtures."""

    catalog_version: Literal["1.0"] = "1.0"
    fixtures: Annotated[tuple[FixtureManifest, ...], Field(min_length=1)]
