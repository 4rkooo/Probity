"""Unit tests for the review/export gate and the evidence bundle export service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from probity.adapters.fixture.weave import sanitize_attributes
from probity.api.mock_client import MockApiClient
from probity.domain.enums import ExplanationSource, VetoReason
from probity.ports import ReportNarrative
from probity.reports.export import (
    ExportBlockedError,
    SourceIdentity,
    export_evidence_bundle,
    verify_bundle,
)
from probity.reports.gate import approval_blockers, export_blockers, review_is_current


@pytest.fixture
def client() -> MockApiClient:
    return MockApiClient()


def _approve(client: MockApiClient):
    run = client.run_succeeded
    return client.submit_review(
        run_id=run.run_id,
        decision="APPROVE",
        reviewer_alias="analyst-1",
        comment="ok",
        reviewed_result_sha256=run.result_png_sha256,
        reviewed_provenance_sha256=run.provenance_sha256,
    )


def _export(client: MockApiClient, tmp_path: Path, review):
    run = client.run_succeeded
    artifacts = client.export_artifacts()
    narrative, fell_back = client.draft_narrative(client.report_facts(run, review))
    return export_evidence_bundle(
        case=client.case,
        source=SourceIdentity(**client.run_source_identity()),
        observed_source_sha256=client.verify_run_source()[1],
        run=run,
        review=review,
        decisions=client.list_decisions(run.run_id),
        artifacts=artifacts,
        trace_spans=client.trace.spans,
        narrative=narrative,
        narrative_fell_back=fell_back,
        output_dir=tmp_path / "bundle",
    )


def test_succeeded_run_is_approvable(client: MockApiClient) -> None:
    assert approval_blockers(client.run_succeeded, source_verified=True) == []


@pytest.mark.parametrize("outcome", ["REFUSED", "FAILED"])
def test_non_succeeded_runs_are_not_approvable(client: MockApiClient, outcome: str) -> None:
    run = client.get_reconstruction(outcome=outcome)
    codes = {b.code for b in approval_blockers(run, source_verified=True)}
    assert f"RUN_{outcome}" in codes
    assert "PROVENANCE_INCOMPLETE" in codes


def test_low_integrity_and_incomplete_provenance_block(client: MockApiClient) -> None:
    run = client.run_succeeded
    low = run.model_copy(update={"integrity": run.integrity.model_copy(update={"score_0_100": 69})})
    assert "INTEGRITY_BELOW_MINIMUM" in {
        b.code for b in approval_blockers(low, source_verified=True)
    }
    holey = run.model_copy(
        update={"provenance": run.provenance.model_copy(update={"coverage_complete": False})}
    )
    assert "PROVENANCE_INCOMPLETE" in {
        b.code for b in approval_blockers(holey, source_verified=True)
    }
    gen = run.model_copy(
        update={
            "provenance": run.provenance.model_copy(
                update={
                    "subject_coverage_pct": run.provenance.subject_coverage_pct.model_copy(
                        update={"GENERATED_BLEND": 0.5}
                    )
                }
            )
        }
    )
    assert "GENERATED_SEMANTIC_PIXEL" in {
        b.code for b in approval_blockers(gen, source_verified=True)
    }


def test_source_mismatch_blocks_approval(client: MockApiClient) -> None:
    client.source_tampered = True
    assert client.verify_source()[0] is False
    with pytest.raises(ValueError, match="Approval blocked"):
        _approve(client)


def test_export_gate_states(client: MockApiClient) -> None:
    run = client.run_succeeded
    assert {b.code for b in export_blockers(run, None, source_verified=True)} == {"REVIEW_REQUIRED"}

    review = _approve(client)
    assert review_is_current(run, review)
    assert export_blockers(run, review, source_verified=True) == []

    regenerated = client.regenerate_artifacts()
    assert not review_is_current(regenerated, review)
    assert "STALE_APPROVAL" in {
        b.code for b in export_blockers(regenerated, review, source_verified=True)
    }


def test_veto_blocks_export(client: MockApiClient) -> None:
    run = client.run_succeeded
    veto = client.submit_review(
        run_id=run.run_id,
        decision="VETO",
        reviewer_alias="analyst-1",
        comment="misaligned",
        reviewed_result_sha256=run.result_png_sha256,
        reviewed_provenance_sha256=run.provenance_sha256,
        veto_reason=VetoReason.MISALIGNMENT,
    )
    assert "HUMAN_VETOED" in {b.code for b in export_blockers(run, veto, source_verified=True)}
    with pytest.raises(ValueError, match="Export blocked"):
        client.create_report(run.run_id)


def test_export_bundle_contents_and_hashes_verify(client: MockApiClient, tmp_path: Path) -> None:
    review = _approve(client)
    bundle = _export(client, tmp_path, review)
    assert verify_bundle(bundle.directory) == []
    assert bundle.zip_path.exists()

    names = {p.name for p in bundle.directory.iterdir()}
    assert {
        "report.html",
        "report.json",
        "manifest.json",
        "result.png",
        "provenance.npz",
        f"target_f{client.target_frame_number}.png",
        "donor_f43.png",
        "policy_decisions.json",
        "trace_summary.json",
    } <= names

    manifest = json.loads((bundle.directory / "manifest.json").read_text())
    by_role = {f["role"]: f for f in manifest["files"]}
    assert by_role["result"]["sha256"] == client.run_succeeded.result_png_sha256
    assert by_role["provenance"]["sha256"] == client.run_succeeded.provenance_sha256
    assert by_role["source"]["sha256"] == client.window.source_sha256

    html = bundle.html_path.read_text()
    for needle in (
        str(client.run_succeeded.target_pts_us),
        *(str(e.pts_us) for e in client.run_succeeded.provenance.source_lut),
        "ALIGNMENT_AKAZE_ACCEPTED",
        client.window.source_sha256,
        "Recorded uncertainty",
        "FIXTURE",
        "Artifact Manifest",
    ):
        assert needle in html

    # Tampering with an exported file is detected.
    (bundle.directory / "result.png").write_bytes(b"tampered")
    assert any("result.png" in p for p in verify_bundle(bundle.directory))


def test_export_refuses_tampered_source(client: MockApiClient, tmp_path: Path) -> None:
    review = _approve(client)
    client.source_tampered = True
    with pytest.raises(ExportBlockedError, match="Source bytes"):
        _export(client, tmp_path, review)


def test_hallucinated_narrative_falls_back(client: MockApiClient) -> None:
    review = _approve(client)
    facts = client.report_facts(client.run_succeeded, review)
    client.hallucinate_narrative = True
    narrative, fell_back = client.draft_narrative(facts)
    assert fell_back is True
    assert narrative.source is ExplanationSource.TEMPLATE
    assert not any("ABC-1234" in p for p in narrative.paragraphs)
    assert isinstance(narrative, ReportNarrative)


def test_traced_spans_are_redacted(client: MockApiClient) -> None:
    client.search("Find the blue sedan when its rear plate is most visible")
    _approve(client)
    names = [s["span_name"] for s in client.trace.spans]
    assert names[:2] == ["search.plan", "wandb.explain"]
    assert "review.record" in names
    blob = json.dumps(client.trace.spans)
    assert "blue sedan" not in blob  # query text only as a hash
    assert '"comment"' not in blob
    for span in client.trace.spans:
        assert span["attributes"]["correlation_id"] == client.correlation_id


def test_sanitizer_keeps_token_counts_only() -> None:
    safe = sanitize_attributes(
        {
            "token_count": 5,
            "access_token": "x",
            "frame_bytes": b"\x00",
            "analyst_notes": "n",
            "run_id": "r",
        }
    )
    assert safe == {"token_count": 5, "run_id": "r"}
