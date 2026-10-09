"""Contract-freeze guards: fixtures validate, hashes verify, models are immutable, IDs are exact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from probity.domain import models as m
from probity.domain.enums import ALLOWED_JOB_TRANSITIONS, TERMINAL_JOB_STATES, JobState
from probity.domain.fixtures import FixtureCatalog
from probity.domain.ids import (
    canonical_json,
    frame_id,
    is_uuid7,
    new_uuid7,
    parse_frame_id,
    segment_id,
)
from probity.domain.policy import default_policy
from probity.domain.prompts import COSMOS_INGESTION_PROMPT, COSMOS_INGESTION_PROMPT_SHA256

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "fixtures/contracts"

FIXTURE_MODELS: dict[str, type[m.FrozenModel]] = {
    "case_workspace.json": m.CaseWorkspace,
    "source_video.json": m.SourceVideo,
    "source_video_partial.json": m.SourceVideo,
    "frame_manifest.json": m.FrameManifest,
    "video_segments.json": m.VideoSegment,
    "segment_descriptions.json": m.SegmentDescription,
    "detections.json": m.Detection,
    "search_evidence.json": m.SearchEvidence,
    "search_evidence_needs_clarification.json": m.SearchEvidence,
    "track_confirmed.json": m.Track,
    "track_not_confirmed.json": m.Track,
    "policy_decisions_succeeded.json": m.PolicyDecision,
    "policy_decisions_refused.json": m.PolicyDecision,
    "pixel_provenance.json": m.PixelProvenance,
    "pixel_origin_borrowed.json": m.PixelOrigin,
    "reconstruction_run_succeeded.json": m.ReconstructionRun,
    "reconstruction_run_refused.json": m.ReconstructionRun,
    "reconstruction_run_failed.json": m.ReconstructionRun,
    "human_review_approve.json": m.HumanReview,
    "human_review_veto.json": m.HumanReview,
    "evidence_report.json": m.EvidenceReport,
    "job_views.json": m.JobView,
    "asset_refs.json": m.AssetRef,
}


def load(name: str) -> list[dict[str, object]]:
    data = json.loads((CONTRACTS / name).read_text())
    return data if isinstance(data, list) else [data]


def test_every_contract_fixture_is_covered() -> None:
    on_disk = {p.name for p in CONTRACTS.glob("*.json")}
    assert on_disk == set(FIXTURE_MODELS)


@pytest.mark.parametrize("name", sorted(FIXTURE_MODELS))
def test_fixture_validates_and_round_trips(name: str) -> None:
    model = FIXTURE_MODELS[name]
    for raw in load(name):
        obj = model.model_validate(raw)
        assert obj.model_dump(mode="json") == raw
        assert model.model_validate_json(obj.model_dump_json()) == obj


@pytest.mark.parametrize("name", sorted(FIXTURE_MODELS))
def test_record_hash_is_enforced(name: str) -> None:
    model = FIXTURE_MODELS[name]
    if not issubclass(model, m.Record):
        pytest.skip("value object")
    raw = dict(load(name)[0])
    tampered = dict(raw, created_at="2030-01-01T00:00:00.000000Z")
    with pytest.raises(ValidationError, match="content_sha256 mismatch"):
        model.model_validate(tampered)


def test_content_hash_excludes_itself_and_is_canonical() -> None:
    case = m.CaseWorkspace.model_validate(load("case_workspace.json")[0])
    payload = case.model_dump(mode="json")
    payload.pop("content_sha256")
    import hashlib

    assert hashlib.sha256(canonical_json(payload)).hexdigest() == case.content_sha256


def test_models_are_frozen_and_reject_extra_fields() -> None:
    video = m.SourceVideo.model_validate(load("source_video.json")[0])
    with pytest.raises(ValidationError):
        video.sha256 = "0" * 64  # type: ignore[misc]
    raw = dict(load("source_video.json")[0], unexpected="x")
    with pytest.raises(ValidationError):
        m.SourceVideo.model_validate(raw)


def test_frozen_config_on_every_model() -> None:
    for name in m.__all__:
        obj = getattr(m, name)
        if isinstance(obj, type) and issubclass(obj, m.FrozenModel):
            assert obj.model_config.get("frozen") is True, name
            assert obj.model_config.get("extra") == "forbid", name


def test_revise_creates_new_hash_without_mutating_original() -> None:
    video = m.SourceVideo.model_validate(load("source_video.json")[0])
    before = video.model_dump(mode="json")
    revised = video.revise(ingest_state="QUARANTINED")
    assert video.model_dump(mode="json") == before
    assert revised.content_sha256 != video.content_sha256


def test_uuid7_and_deterministic_ids() -> None:
    value = new_uuid7()
    assert is_uuid7(value) and value == value.lower()
    vid = "0199a51e-43bf-7aa2-86e1-b2a0bb287492"
    assert frame_id(vid, 417) == f"{vid}:f417"
    assert segment_id(vid, 4) == f"{vid}:s0004"
    assert parse_frame_id(f"{vid}:f0") == (vid, 0)
    with pytest.raises(ValueError, match="invalid frame_id"):
        parse_frame_id(f"{vid}:f007")


def test_segment_and_frame_identity_validation() -> None:
    raw = dict(load("video_segments.json")[3])
    raw["ordinal"] = 9
    with pytest.raises(ValidationError):
        m.VideoSegment.model_validate(raw)


def test_job_transition_table_matches_spec() -> None:
    assert ALLOWED_JOB_TRANSITIONS[JobState.CREATED] == {JobState.QUEUED}
    assert ALLOWED_JOB_TRANSITIONS[JobState.QUEUED] == {JobState.RUNNING, JobState.CANCELLING}
    assert ALLOWED_JOB_TRANSITIONS[JobState.CANCELLING] == {JobState.CANCELLED}
    for state in TERMINAL_JOB_STATES:
        assert ALLOWED_JOB_TRANSITIONS[state] == frozenset()


def test_job_fixtures_cover_every_state() -> None:
    states = {v["state"] for v in load("job_views.json")}
    assert states == {s.value for s in JobState}


def test_partial_requires_ingest_and_committed_segment() -> None:
    raw = next(v for v in load("job_views.json") if v["state"] == "PARTIAL")
    bad = m.JobView.model_validate(raw).model_dump(mode="python", exclude={"content_sha256"})
    bad["partial"] = None
    with pytest.raises(ValidationError, match="PARTIAL requires"):
        m.JobView.create(**bad)


def test_refused_run_has_no_result_and_succeeded_has_full_provenance() -> None:
    ok = m.ReconstructionRun.model_validate(load("reconstruction_run_succeeded.json")[0])
    assert ok.provenance is not None and ok.provenance.coverage_complete
    assert ok.provenance.coverage_pct.GENERATED_BLEND == 0.0
    refused = m.ReconstructionRun.model_validate(load("reconstruction_run_refused.json")[0])
    assert refused.result_png_uri is None and refused.refusal_reasons


def test_integrity_formula_enforced() -> None:
    with pytest.raises(ValidationError, match="integrity-v1"):
        m.IntegrityScore(
            score_0_100=91,
            supported_coverage=1.0,
            mean_source_confidence=0.86,
            mean_alignment_confidence=0.93,
            track_continuity=0.94,
            audit_completeness=1.0,
            semantic_generated_pct=0.0,
        )


def test_search_results_are_timestamped_and_cited() -> None:
    ev = m.SearchEvidence.model_validate(load("search_evidence.json")[0])
    assert 1 <= len(ev.results) <= 5
    for r in ev.results:
        assert r.end_pts_us > r.start_pts_us
        assert r.source_sha256 == load("source_video.json")[0]["sha256"]


def test_policy_and_prompt_are_pinned() -> None:
    policy = default_policy()
    assert policy.segment.step_us == 6_000_000
    assert policy.video.max_bytes == 262_144_000
    assert policy.search.top_k == 12 and policy.search.max_results == 5
    assert COSMOS_INGESTION_PROMPT.startswith("You are creating a factual search index")
    catalog = FixtureCatalog.model_validate_json((ROOT / "fixtures/demo/manifest.json").read_text())
    manifest = catalog.fixtures[0]
    assert manifest.policy_config_sha256 == policy.config_sha256
    assert manifest.ingestion_prompt_sha256 == COSMOS_INGESTION_PROMPT_SHA256


def test_demo_source_matches_manifest_hash() -> None:
    import hashlib

    catalog = FixtureCatalog.model_validate_json((ROOT / "fixtures/demo/manifest.json").read_text())
    src = catalog.fixtures[0].source
    data = (ROOT / "fixtures/demo" / src.path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == src.sha256 and len(data) == src.byte_length


def test_frame_reference_rejects_baseline_paths() -> None:
    vid = "0199a51e-43bf-7aa2-86e1-b2a0bb287492"
    with pytest.raises(ValidationError):
        m.FrameReference.create(
            frame_id=f"{vid}:f1",
            video_id=vid,
            frame_number=1,
            pts_us=33333,
            is_keyframe=False,
            width_px=10,
            height_px=10,
            lossless_png_uri=f"derived/{vid}/baseline/x/out.png",
            pixel_sha256="0" * 64,
        )
