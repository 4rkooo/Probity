"""Screen 8: Evidence Report Preview & Delivery View."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from probity.api.mock_client import MockApiClient
from probity.reports.export import (
    ExportBlockedError,
    SourceIdentity,
    export_evidence_bundle,
    verify_bundle,
)
from probity.reports.gate import export_blockers


def _export_dir(run_id: str) -> Path:
    return Path(tempfile.gettempdir()) / "probity_exports" / run_id


def render_report_view(client: MockApiClient) -> None:
    ss = st.session_state
    st.subheader("7. Evidence Report Delivery & Export Manifest")

    run = (
        client.get_reconstruction()
        if client.uses_live_upload
        else client.run_succeeded
    )
    review = client.get_latest_review(run.run_id)
    if client.uses_live_upload:
        st.info(
            f"Evidence report for uploaded clip **{client.source_video.original_name}**."
        )
    verified, observed = client.verify_run_source()
    blockers = export_blockers(run, review, source_verified=verified)
    codes = {b.code for b in blockers}

    # Source re-verification is always shown, before any export decision.
    color, label = ("#4ade80", "✓ SOURCE HASH RE-VERIFIED NOW") if verified else ("#f87171", "✗ SOURCE HASH MISMATCH")
    st.markdown(
        f"""
        <div style="background:#1e293b; color:#e2e8f0; padding:14px; border-radius:6px; font-size:12px; border:1px solid #334155; margin-bottom:12px;">
          <div style="color:{color}; font-weight:700;">{label}</div>
          <div style="font-family:monospace; margin-top:4px; word-break:break-all;">ingest:&nbsp;&nbsp;&nbsp;{client.source_video.sha256}</div>
          <div style="font-family:monospace; word-break:break-all;">observed: {observed}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "REVIEW_REQUIRED" in codes:
        st.warning("⚠️ Prerequisite Required: complete the human review gate before exporting.")
        if st.button("⬅ Go to Review Gate"):
            ss["step"] = "Review"
            st.rerun()
    elif "HUMAN_VETOED" in codes and review is not None:
        st.error(
            f"🛑 EXPORT PERMANENTLY BLOCKED: reconstruction was vetoed by `{review.reviewer_alias}` "
            f"(reason: {review.reason_code}). No evidence bundle may be generated."
        )
    elif "STALE_APPROVAL" in codes:
        st.error(
            "🛑 EXPORT BLOCKED: the approval was recorded for different artifact hashes. "
            "Review the current artifacts again."
        )
    if blockers:
        with st.expander("Why export is locked", expanded="SOURCE_HASH_MISMATCH" in codes):
            for b in blockers:
                st.markdown(f"- `{b.code}`: {b.message}")
        st.button("📦 Export Evidence Bundle", disabled=True, width="stretch")
        return

    assert review is not None
    st.success(
        f"✅ Active Approval Confirmed: reviewed by `{review.reviewer_alias}` at {review.created_at} "
        "for the exact current artifact hashes."
    )

    facts = client.report_facts(run, review)
    narrative, fell_back = client.draft_narrative(facts)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Structured Report Preview")
        integ = run.integrity
        prov = run.provenance
        st.dataframe(
            [
                {"Field": "Case", "Value": f"{client.case.display_name} ({client.case.case_id})"},
                {"Field": "Source", "Value": f"{client.run_source_identity()['original_name']} · {client.run_source_identity()['width_px']}x{client.run_source_identity()['height_px']}"},
                {"Field": "Target frame / PTS", "Value": f"{run.target_frame_id.split(':')[-1]} @ {run.target_pts_us} µs"},
                {"Field": "Donors", "Value": ", ".join(d.split(":")[-1] for d in run.accepted_donor_frame_ids)},
                {"Field": "Integrity", "Value": f"{integ.score_0_100}/100 ({integ.formula_version})" if integ else "n/a"},
                {"Field": "Original / Borrowed / Generated (subject)", "Value": f"{prov.subject_coverage_pct.ORIGINAL:.2f}% / {prov.subject_coverage_pct.BORROWED:.2f}% / {prov.subject_coverage_pct.GENERATED_BLEND:.2f}%" if prov else "n/a"},
                {"Field": "Review", "Value": f"{review.decision} by {review.reviewer_alias} at {review.created_at}"},
                {
                    "Field": "Mode",
                    "Value": (
                        f"{run.mode} (custom upload)"
                        if client.uses_live_upload
                        else f"{run.mode} (verified cached inference)"
                    ),
                },
            ],
            width="stretch",
            hide_index=True,
        )
        st.markdown("#### Narrative")
        if fell_back:
            st.warning(
                "W&B draft failed grounding validation (uncited timestamp, identity, or confidence claim). "
                "Showing the deterministic template narrative instead."
            )
        st.caption(f"Narrative source: {narrative.source}")
        for p in narrative.paragraphs:
            st.markdown(f"> {p}")

        st.markdown("#### Included Artifact Manifest")
        st.markdown(
            """
            - `report.html` / `report.json` - standalone and machine-readable reports
            - target frame, `result.png`, donor frames - lossless canonical stills
            - `provenance.npz` - authoritative class / source-index / source-coordinate arrays
            - `policy_decisions.json` - ordered audit log
            - `trace_summary.json` - sanitized Weave spans (no media bytes, crops, secrets, or notes)
            - `manifest.json` - SHA-256 of every file for re-verification
            """
        )

    with col2:
        st.markdown("### Export Bundle")
        if st.button("📦 Export Evidence Bundle", type="primary", width="stretch"):
            artifacts = client.export_artifacts()
            with client.trace.span(
                "report.render",
                {
                    "correlation_id": client.correlation_id,
                    "run_id": run.run_id,
                    "review_id": review.review_id,
                    "narrative_source": str(narrative.source),
                    "narrative_fell_back": fell_back,
                },
            ):
                try:
                    bundle = export_evidence_bundle(
                        case=client.case,
                        source=SourceIdentity(**client.run_source_identity()),
                        # Re-hash every source frame immediately before export.
                        observed_source_sha256=client.verify_run_source()[1],
                        run=run,
                        review=review,
                        decisions=client.list_decisions(run.run_id),
                        artifacts=artifacts,
                        trace_spans=client.trace.spans,
                        narrative=narrative,
                        narrative_fell_back=fell_back,
                        output_dir=_export_dir(run.run_id),
                    )
                except ExportBlockedError as exc:
                    st.error(str(exc))
                    return
            ss["export_dir"] = str(bundle.directory)
            ss["export_zip"] = str(bundle.zip_path)
            ss["export_bundle_sha256"] = bundle.report.bundle_sha256

        export_dir = ss.get("export_dir")
        if export_dir and Path(export_dir).exists():
            problems = verify_bundle(export_dir)
            st.markdown(
                f"""
                <div style="background:#1e293b; color:#e2e8f0; padding:10px; border-radius:4px; font-size:11px; margin-bottom:12px;">
                  <div style="color:#94a3b8; font-weight:600;">BUNDLE SHA-256:</div>
                  <div style="color:#38bdf8; font-family:monospace; word-break:break-all;">{ss['export_bundle_sha256']}</div>
                  <div style="margin-top:6px; color:{'#4ade80' if not problems else '#f87171'}; font-weight:700;">
                    {'✓ Bundle re-verified: every manifest hash matches' if not problems else '✗ ' + '; '.join(problems)}</div>
                  <div style="margin-top:4px;">Mode: {"LIVE (custom upload)" if client.uses_live_upload else "FIXTURE (verified cached inference)"}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            out = Path(export_dir)
            st.download_button(
                "📥 Download Bundle (.zip)",
                data=Path(ss["export_zip"]).read_bytes(),
                file_name=f"probity_evidence_{client.case.case_id}.zip",
                mime="application/zip",
                width="stretch",
                type="primary",
            )
            st.download_button(
                "📥 Download HTML Report",
                data=(out / "report.html").read_bytes(),
                file_name=f"probity_report_{client.case.case_id}.html",
                mime="text/html",
                width="stretch",
            )
            st.download_button(
                "📥 Download JSON Report",
                data=(out / "report.json").read_bytes(),
                file_name=f"probity_report_{client.case.case_id}.json",
                mime="application/json",
                width="stretch",
            )
            with st.expander("Weave trace summary (sanitized)"):
                st.json(client.trace.spans)
