"""UI smoke tests for the Person 3 Streamlit application (design section 14, "UI smoke")."""

from __future__ import annotations

import random

from probity.domain.enums import VetoReason
from probity.ui.components.provenance_canvas import build_canvas_html, resolve_from_exceptions
from tests.ui.conftest import click, texts

# ---------------------------------------------------------------------------------------------
# Persistent shell
# ---------------------------------------------------------------------------------------------


def test_home_renders_persistent_disclosure(make_app, client) -> None:
    at = make_app("Source", progress=False).run()
    assert not at.exception
    page = texts(at)
    assert "Research/demo prototype - not for legal conclusions" in page
    assert "VERIFIED CACHE" in page
    assert client.manifest_created_at in page
    assert client.source_video.sha256[:12] in page
    assert "VIDEO: NOT INGESTED" in page
    assert any(c.value == client.source_video.sha256 for c in at.code)  # copy action


def test_every_step_renders_with_progress(make_app) -> None:
    for step in ("Source", "Ingest", "Search", "Subject", "TrueFrame", "Review", "Report"):
        at = make_app(step).run()
        assert not at.exception, f"Failed on step {step}"


def test_locked_step_explains_prerequisite(make_app) -> None:
    at = make_app("Search", progress=False).run()
    assert not at.exception
    assert any("finish ingestion first" in i.value for i in at.info)


# ---------------------------------------------------------------------------------------------
# Upload / ingestion states
# ---------------------------------------------------------------------------------------------


def test_upload_empty_state(make_app) -> None:
    at = make_app("Source", progress=False).run()
    at.radio[0].set_value("Upload New MP4 (Experimental)").run()
    assert "No video yet" in texts(at)


def test_ingest_polls_to_success(make_app) -> None:
    at = make_app("Ingest", progress=False).run()
    click(at, "Start Ingestion")
    assert not at.exception
    assert at.session_state["ingest_state"] == "SUCCEEDED"
    assert "VAST AI OS" in texts(at)
    assert "VIDEO: SEARCHABLE" in texts(at)


def test_ingest_partial_limits_search(make_app, client) -> None:
    client.ingest_outcome = "PARTIAL"
    at = make_app("Ingest", progress=False).run()
    click(at, "Start Ingestion")
    assert at.session_state["ingest_state"] == "PARTIAL"
    assert "PARTIAL - indexed 0.0s to 74.0s" in texts(at)

    click(at, "Proceed to Search")
    click(at, "🔎 Search")
    page = texts(at)
    assert "PARTIAL INDEX" in page
    assert "Rank #1" in page


def test_ingest_failure_shows_correlation_and_retry(make_app, client) -> None:
    client.ingest_outcome = "FAILED"
    at = make_app("Ingest", progress=False).run()
    click(at, "Start Ingestion")
    page = texts(at)
    assert at.session_state["ingest_state"] == "FAILED"
    assert "WORKER_INTERRUPTED" in page
    assert "Correlation ID" in page
    assert "Traceback" not in page
    click(at, "Retry Ingestion")
    assert at.session_state["ingest_state"] == "SUCCEEDED"


def test_sponsor_timeout_offers_verified_cache(make_app, client) -> None:
    client.ingest_outcome = "SPONSOR_TIMEOUT"
    at = make_app("Ingest", progress=False).run()
    click(at, "Start Ingestion")
    assert "SPONSOR_TIMEOUT" in texts(at)
    click(at, "Continue with verified demo cache")
    assert at.session_state["ingest_state"] == "SUCCEEDED"


def test_ingest_cancellation(make_app) -> None:
    at = make_app(
        "Ingest", progress=False, ingest_state="RUNNING", ingest_cancel=True, ingest_tick=0
    )
    at.run()
    assert at.session_state["ingest_state"] == "CANCELLED"
    assert any("CANCELLED" in w.value for w in at.warning)


# ---------------------------------------------------------------------------------------------
# Search states
# ---------------------------------------------------------------------------------------------


def test_search_no_results_and_filtered_terms(make_app) -> None:
    at = make_app("Search", query_input="weather over the harbor").run()
    click(at, "🔎 Search")
    assert "No indexed moments matched" in texts(at)

    at.text_input[0].set_value("Find the fleeing stolen car").run()
    click(at, "🔎 Search")
    assert any("Policy filtered" in w.value for w in at.warning)


def test_empty_query_disables_search(make_app) -> None:
    at = make_app("Search").run()
    at.text_input[0].set_value("   ").run()
    search = next(b for b in at.button if "🔎 Search" in b.label)
    assert search.disabled


# ---------------------------------------------------------------------------------------------
# Subject / TrueFrame
# ---------------------------------------------------------------------------------------------


def test_unconfirmed_track_disables_reconstruction(make_app) -> None:
    at = make_app("Subject").run()
    at.radio[0].set_value(at.radio[0].options[1]).run()
    run_btn = next(b for b in at.button if "Run TrueFrame" in b.label)
    assert run_btn.disabled
    assert any("track not confirmed" in w.value for w in at.warning)


def test_refusal_state(make_app) -> None:
    at = make_app("TrueFrame", target_frame=414, recon_done=False, run_outcome=None).run()
    assert not at.exception
    page = texts(at)
    assert "No defensible enhancement produced" in page
    assert "INSUFFICIENT_COMPATIBLE_DONORS" in page
    click(at, "Choose a different target frame")
    assert at.session_state["step"] == "Subject"


def test_reconstruction_failure_state(make_app, client) -> None:
    client.run_outcome = "FAILED"
    at = make_app("TrueFrame", recon_done=False, run_outcome=None).run()
    page = texts(at)
    assert "SOURCE_HASH_MISMATCH" in page
    assert "Retryable: no" in page


def test_comparison_labels_baseline_and_resolves_pixels(make_app) -> None:
    at = make_app("TrueFrame").run()
    assert not at.exception
    assert "BASELINE IS NON-EVIDENTIARY" in texts(at)

    at.number_input(key="inspect_x").set_value(230).run()
    at.number_input(key="inspect_y").set_value(481).run()
    page = texts(at)
    assert "BORROWED" in page and "f409" in page and "13.633s" in page

    at.number_input(key="inspect_x").set_value(260).run()
    page = texts(at)
    assert "f424" in page and "14.133s" in page

    click(at, "Open source frame")
    assert at.session_state["open_source"]["frame"] == 424
    assert "Original player seeked to 14.133s" in texts(at)


def test_canvas_lookup_matches_npz_everywhere(client) -> None:
    exceptions = client.provenance_exceptions()
    w, h = client.provenance_shape()
    rng = random.Random(7)
    points = [(int(r[0]), int(r[1])) for r in exceptions[::17]]
    points += [(rng.randrange(w), rng.randrange(h)) for _ in range(200)]
    for x, y in points:
        origin = client.resolve_pixel(x, y)
        cls, idx, sx, sy = resolve_from_exceptions(x, y, exceptions)
        assert origin is not None
        assert (cls, idx, sx, sy) == (
            ["ORIGINAL", "BORROWED", "GENERATED_BLEND"].index(origin.provenance_class),
            origin.source_index,
            origin.source_x,
            origin.source_y,
        )

    html = build_canvas_html(
        target_img_uri="data:,",
        result_img_uri="data:,",
        donor_img_uris={},
        source_video_uri="data:,",
        lut_entries=[],
        exceptions=exceptions,
        frame_size=(w, h),
        subject_bbox=(216, 470, 292, 496),
        decision_codes={},
        target_pts_us=13_900_000,
        nominal_fps=30.0,
        initial_pixel=(224, 481),
    )
    assert "RECOMPRESSED PREVIEW - NOT THE CANONICAL RESULT" in html
    assert "Open source" in html
    assert "__DATA__" not in html


# ---------------------------------------------------------------------------------------------
# Review gate and export
# ---------------------------------------------------------------------------------------------


def _approve_button(at):
    return next(b for b in at.button if "Approve" in b.label)


def test_approval_requires_inspection(make_app) -> None:
    at = make_app("Review").run()
    assert _approve_button(at).disabled
    next(c for c in at.checkbox if "I have inspected" in c.label).check().run()
    assert not _approve_button(at).disabled


def test_source_mismatch_disables_approval(make_app, client) -> None:
    client.source_tampered = True
    at = make_app("Review", review_inspected=True).run()
    assert _approve_button(at).disabled
    assert any("SOURCE_HASH_MISMATCH" in e.value for e in at.error)


def test_no_review_then_veto_blocks_export(make_app, client) -> None:
    at = make_app("Report").run()
    assert not at.exception
    assert any("Prerequisite Required" in w.value for w in at.warning)

    run = client.run_succeeded
    client.submit_review(
        run_id=run.run_id,
        decision="VETO",
        reviewer_alias="analyst-lead",
        comment="Incompatible donor alignment.",
        reviewed_result_sha256=run.result_png_sha256,
        reviewed_provenance_sha256=run.provenance_sha256,
        veto_reason=VetoReason.MISALIGNMENT,
    )
    at = make_app("Report").run()
    assert any("EXPORT PERMANENTLY BLOCKED" in e.value for e in at.error)
    export = next(b for b in at.button if "Export Evidence Bundle" in b.label)
    assert export.disabled


def test_stale_approval_blocks_export(make_app, client) -> None:
    at = make_app("Review", review_inspected=True).run()
    click(at, "Approve")
    assert at.session_state["step"] == "Report"

    client.regenerate_artifacts()
    at = make_app("Report").run()
    assert any("approval was recorded for different artifact hashes" in e.value for e in at.error)
    assert next(b for b in at.button if "Export Evidence Bundle" in b.label).disabled

    at = make_app("Review").run()
    assert any("STALE APPROVE" in w.value for w in at.warning)


def test_approved_export_bundle_verifies(make_app) -> None:
    at = make_app("Review", review_inspected=True).run()
    click(at, "Approve")
    assert any("Active Approval Confirmed" in s.value for s in at.success)
    click(at, "Export Evidence Bundle")
    assert not at.exception
    assert "Bundle re-verified: every manifest hash matches" in texts(at)
    assert len(at.get("download_button")) == 3


def test_hallucinated_narrative_falls_back(make_app, client) -> None:
    client.hallucinate_narrative = True
    at = make_app("Review", review_inspected=True).run()
    click(at, "Approve")
    page = texts(at)
    assert "failed grounding validation" in page
    assert "ABC-1234" not in page
