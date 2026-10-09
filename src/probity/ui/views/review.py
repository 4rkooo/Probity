"""Screen 6 & 7: Integrity, Policy Audit & Human Review Gate View."""

from __future__ import annotations

import html

import streamlit as st

from probity.api.mock_client import MockApiClient
from probity.domain.enums import ReviewDecision, VetoReason
from probity.reports.gate import MIN_INTEGRITY_SCORE, approval_blockers, review_is_current

SCORE_HELP = (
    "Integrity measures lineage completeness and compatibility. It is not a claim of "
    "accuracy, identity, truth, or legal admissibility."
)


def _integrity_panel(client: MockApiClient) -> None:
    run = client.run_succeeded
    integrity = run.integrity
    st.markdown("### Deterministic Integrity Score")
    col_score, col_bars = st.columns([1, 2])
    with col_score:
        score_val = integrity.score_0_100 if integrity else 0
        passes = score_val >= MIN_INTEGRITY_SCORE
        st.metric(
            label=f"Integrity score ({integrity.formula_version if integrity else 'n/a'})",
            value=f"{score_val} / 100",
            delta=f"{'PASS' if passes else 'BELOW MINIMUM'} (>= {MIN_INTEGRITY_SCORE} required)",
            delta_color="normal" if passes else "inverse",
            help=SCORE_HELP,
        )
        st.caption("ℹ️ " + SCORE_HELP)
        st.caption("Formula: round(100·(0.30C + 0.25Qd + 0.20Ad + 0.15Tc + 0.10Au)·(1−G))")
    with col_bars:
        st.markdown("**Integrity components**")
        if integrity:
            for label, val in (
                ("Supported coverage (C)", integrity.supported_coverage),
                ("Source confidence (Qd)", integrity.mean_source_confidence),
                ("Alignment confidence (Ad)", integrity.mean_alignment_confidence),
                ("Track continuity (Tc)", integrity.track_continuity),
                ("Audit completeness (Au)", integrity.audit_completeness),
            ):
                st.progress(float(val), text=f"{label}: {val * 100:.1f}%")

    st.markdown("### Provenance & Uncertainty")
    u_col1, u_col2 = st.columns(2)
    with u_col1:
        if run.provenance:
            full, subj = run.provenance.coverage_pct, run.provenance.subject_coverage_pct
            st.markdown(
                f"""
                <div style="background:#1e293b; color:#e2e8f0; padding:12px; border-radius:6px; font-size:12px; border:1px solid #334155;">
                  <div style="font-weight:600; color:#38bdf8;">Provenance percentages (full frame / subject region)</div>
                  <div style="margin-top:6px;">■ Original: <strong>{full.ORIGINAL:.2f}% / {subj.ORIGINAL:.2f}%</strong></div>
                  <div>■ Borrowed: <strong>{full.BORROWED:.2f}% / {subj.BORROWED:.2f}%</strong></div>
                  <div>■ Generated blend: <strong>{full.GENERATED_BLEND:.2f}% / {subj.GENERATED_BLEND:.2f}%</strong></div>
                  <div style="margin-top:6px; color:#94a3b8;">Coverage complete: <strong>{'YES' if run.provenance.coverage_complete else 'NO'}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with u_col2:
        items = "".join(f"<li>{html.escape(u)}</li>" for u in run.uncertainty)
        st.markdown(
            f"""
            <div style="background:#78350f; color:#fef3c7; padding:12px; border-radius:6px; font-size:12px; border-left:4px solid #f59e0b;">
              <div style="font-weight:700;">Uncertainty (plain language)</div>
              <ul style="margin:6px 0 0 16px; padding:0;">{items}
                <li>Characters remain unresolved where no compatible donor was clearer; no plate text is claimed.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _policy_log(client: MockApiClient) -> None:
    st.markdown("### Ordered Policy Decision Log")
    decisions = sorted(client.list_decisions(client.run_succeeded.run_id), key=lambda d: d.sequence)
    f1, f2, f3 = st.columns(3)
    with f1:
        codes = st.multiselect("Reason codes", sorted({str(d.rule_code) for d in decisions}))
    with f2:
        stages = st.multiselect("Stages", sorted({str(d.stage) for d in decisions}))
    with f3:
        outcomes = st.multiselect("Outcomes", sorted({str(d.outcome) for d in decisions}))
    rows = [
        d
        for d in decisions
        if (not codes or str(d.rule_code) in codes)
        and (not stages or str(d.stage) in stages)
        and (not outcomes or str(d.outcome) in outcomes)
    ]
    st.dataframe(
        [
            {
                "#": d.sequence,
                "Rule Code": str(d.rule_code),
                "Stage": str(d.stage),
                "Subject": d.subject_ref.split(":")[-1],
                "Outcome": str(d.outcome),
                "Observed": str(d.observed),
                "Threshold": f"{d.operator} {d.threshold} {d.units or ''}".strip(),
                "Reason": d.reason,
            }
            for d in rows
        ],
        width="stretch",
        hide_index=True,
    )


def render_review_view(client: MockApiClient) -> None:
    ss = st.session_state
    st.subheader("6. Integrity Metrics, Policy Audit & Review Gate")

    run = client.run_succeeded
    _integrity_panel(client)
    _policy_log(client)

    st.markdown("---")
    st.markdown("### 🔒 Human Review & Hash Binding Gate")

    current_review = client.get_latest_review(run.run_id)
    current = review_is_current(run, current_review)
    if current_review:
        when = current_review.created_at
        if not current:
            st.warning(
                f"⚠️ STALE {current_review.decision}: recorded by `{current_review.reviewer_alias}` at {when} "
                "for different artifact hashes. The artifacts changed; a new review is required."
            )
        elif current_review.decision is ReviewDecision.APPROVE:
            st.success(f"✅ APPROVED by `{current_review.reviewer_alias}` at {when}. Report export is unlocked.")
        else:
            st.error(
                f"🛑 VETOED by `{current_review.reviewer_alias}` at {when} "
                f"(reason: {current_review.reason_code}). Report export is permanently locked."
            )

    verified, _ = client.verify_run_source()
    blockers = approval_blockers(run, source_verified=verified)
    # Exactly one current review per artifact-hash pair; a stale review does not lock.
    locked = current

    r_col1, r_col2 = st.columns([1, 1])
    with r_col1:
        st.markdown(
            f"""
            <div style="background:#1e293b; padding:12px; border-radius:6px; font-size:11px; font-family:monospace; border:1px solid #334155;">
              <div style="color:#94a3b8; font-weight:700;">EXACT RESULT SHA-256 (result.png):</div>
              <div style="color:#38bdf8; word-break:break-all;">{run.result_png_sha256}</div>
              <div style="color:#94a3b8; font-weight:700; margin-top:8px;">EXACT PROVENANCE SHA-256 (provenance.npz):</div>
              <div style="color:#38bdf8; word-break:break-all;">{run.provenance_sha256}</div>
              <div style="color:#94a3b8; margin-top:8px;">Run {run.run_id} · {run.state}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if blockers:
            st.error(
                "Approval disabled:\n"
                + "\n".join(f"- `{b.code}`: {b.message}" for b in blockers)
            )

    with r_col2:
        inspected = st.checkbox(
            "I have inspected the provenance overlay, donor frame alignment, and policy decision log.",
            value=ss.get("review_inspected", False),
        )
        ss["review_inspected"] = inspected
        reviewer = st.text_input("Reviewer Alias", value=client.case.owner_alias)
        comment = st.text_area("Review Findings / Comment", value="Verified rigid plate geometry across donors f409 and f424.")

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            approve_disabled = not inspected or bool(blockers) or locked or not reviewer.strip()
            if st.button("✅ Approve Enhancement", type="primary", disabled=approve_disabled, width="stretch"):
                client.submit_review(
                    run_id=run.run_id,
                    decision="APPROVE",
                    reviewer_alias=reviewer,
                    comment=comment,
                    reviewed_result_sha256=run.result_png_sha256,  # type: ignore[arg-type]
                    reviewed_provenance_sha256=run.provenance_sha256,  # type: ignore[arg-type]
                )
                ss["step"] = "Report"
                st.rerun()

        with btn_c2:
            veto_reason = st.selectbox("Veto Reason Code:", [r.value for r in VetoReason])
            veto_disabled = not comment.strip() or not reviewer.strip() or locked
            if st.button("🛑 Veto (Block Export)", disabled=veto_disabled, width="stretch"):
                client.submit_review(
                    run_id=run.run_id,
                    decision="VETO",
                    reviewer_alias=reviewer,
                    comment=comment,
                    reviewed_result_sha256=run.result_png_sha256,  # type: ignore[arg-type]
                    reviewed_provenance_sha256=run.provenance_sha256,  # type: ignore[arg-type]
                    veto_reason=VetoReason(veto_reason),
                )
                st.rerun()
            if not comment.strip():
                st.caption("A veto requires a comment.")
