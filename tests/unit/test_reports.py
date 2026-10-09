"""Unit tests for Person 3 report rendering and export gates."""

import tempfile
from pathlib import Path

import pytest

from probity.api.mock_client import MockApiClient
from probity.domain.enums import ReviewDecision, VetoReason
from probity.domain.ids import new_uuid7
from probity.domain.models import HumanReview
from probity.reports.render import ReportRenderer, verify_source_hash


@pytest.fixture
def client() -> MockApiClient:
    return MockApiClient()


def test_source_hash_verification(client: MockApiClient) -> None:
    source_file = client.get_asset_path("demo-plate-90s.mp4")
    assert source_file.exists()
    assert verify_source_hash(source_file, client.source_video.sha256) is True
    assert verify_source_hash(source_file, "bad_hash_00000000000000000000000000000000") is False


def test_approved_report_bundle_renders(client: MockApiClient) -> None:
    renderer = ReportRenderer()
    run = client.get_reconstruction()
    review = HumanReview.create(
        review_id=new_uuid7(),
        run_id=run.run_id,
        reviewer_alias="analyst-lead",
        decision=ReviewDecision.APPROVE,
        reason_code=None,
        comment="Rigid plate observations verified across frames f409 and f424.",
        reviewed_result_sha256=run.result_png_sha256,
        reviewed_provenance_sha256=run.provenance_sha256,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        report, html_path, json_path = renderer.render_bundle(
            case=client.case,
            video=client.source_video,
            run=run,
            review=review,
            output_dir=tmpdir,
            source_path=client.get_asset_path("demo-plate-90s.mp4"),
        )
        assert html_path.exists()
        assert json_path.exists()
        assert report.bundle_sha256 is not None
        assert report.source_verified is True

        html_text = html_path.read_text()
        assert "Provity Video Evidence & Provenance Report" in html_text
        assert "RESEARCH / DEMO PROTOTYPE - NOT FOR LEGAL CONCLUSIONS" in html_text
        assert str(run.integrity.score_0_100) in html_text


def test_veto_blocks_export(client: MockApiClient) -> None:
    renderer = ReportRenderer()
    run = client.get_reconstruction()
    veto_review = HumanReview.create(
        review_id=new_uuid7(),
        run_id=run.run_id,
        reviewer_alias="analyst-lead",
        decision=ReviewDecision.VETO,
        reason_code=VetoReason.MISALIGNMENT,
        comment="Alignment failure on donor frame f409.",
        reviewed_result_sha256=run.result_png_sha256,
        reviewed_provenance_sha256=run.provenance_sha256,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Export blocked: Review decision is VETO"):
            renderer.render_bundle(
                case=client.case,
                video=client.source_video,
                run=run,
                review=veto_review,
                output_dir=tmpdir,
            )


def test_stale_hash_blocks_export(client: MockApiClient) -> None:
    renderer = ReportRenderer()
    run = client.get_reconstruction()
    stale_review = HumanReview.create(
        review_id=new_uuid7(),
        run_id=run.run_id,
        reviewer_alias="analyst-lead",
        decision=ReviewDecision.APPROVE,
        reason_code=None,
        comment="Approved previous hash.",
        reviewed_result_sha256="0" * 64,  # Stale hash!
        reviewed_provenance_sha256=run.provenance_sha256,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Stale approval for result hash"):
            renderer.render_bundle(
                case=client.case,
                video=client.source_video,
                run=run,
                review=stale_review,
                output_dir=tmpdir,
            )
