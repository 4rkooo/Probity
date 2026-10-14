"""Probity: Video Evidence & Insight Agent (Streamlit Shell).

Person 3 implementation: Analyst workflow, Probity provenance inspection,
review gate, and evidence delivery.

Run with ``streamlit run src/probity/ui/Home.py``.
"""

from __future__ import annotations

import html

import streamlit as st

from probity.api.mock_client import INGEST_OUTCOMES, RUN_OUTCOMES
from probity.ui.state import STEPS, get_client, reset_client, step_prerequisite
from probity.ui.views.comparison import render_comparison_view
from probity.ui.views.ingest import render_ingest_view
from probity.ui.views.report import render_report_view
from probity.ui.views.review import render_review_view
from probity.ui.views.search import render_search_view
from probity.ui.views.subject import render_subject_view
from probity.ui.views.upload import render_upload_view

st.set_page_config(
    page_title="Probity - Video Evidence Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ss = st.session_state
ss.setdefault("step", "Source")
client = get_client(ss)

# ---------------------------------------------------------------------------------------------
# Persistent Shell: Top Disclosure Bar
# ---------------------------------------------------------------------------------------------

ingest_label = {
    None: "NOT INGESTED",
    "SUCCEEDED": "SEARCHABLE",
    "PARTIAL": "PARTIAL",
    "FAILED": "INGEST FAILED",
    "CANCELLED": "CANCELLED",
    "RUNNING": "INDEXING",
}.get(ss.get("ingest_state"), str(ss.get("ingest_state")))
source_ok, _ = client.verify_source()  # 250 KB fixture: cheap to re-hash on every render
sha = client.source_video.sha256

top_cols = st.columns([6, 1])
with top_cols[0]:
    st.markdown(
        f"""
        <div style="background:#0f172a; color:#f8fafc; border-radius:8px; padding:10px 14px; display:flex; flex-wrap:wrap; gap:16px; align-items:center; font-size:12px;">
          <div style="font-size:14px; font-weight:700;">🛡️ Probity
            <span style="font-weight:400; color:#94a3b8;">| {html.escape(client.case.display_name)}</span></div>
          <div style="font-family:monospace; color:#cbd5e1;">SHA-256 <strong style="color:#38bdf8;">{sha[:12]}…{sha[-8:]}</strong>
            {'' if source_ok else '<strong style="color:#f87171;"> MISMATCH</strong>'}</div>
          <span style="background:#78350f; color:#fef3c7; border:1px solid #f59e0b; padding:2px 8px; border-radius:9999px; font-size:10px; font-weight:700;"
            title="Fixture manifest verified">VERIFIED CACHE · manifest {html.escape(client.manifest_created_at)}</span>
          <span style="background:#1e293b; color:#a7f3d0; border:1px solid #065f46; padding:2px 8px; border-radius:9999px; font-size:10px; font-weight:700;">
            VIDEO: {html.escape(ingest_label)}</span>
          <div style="color:#f59e0b; font-weight:600; margin-left:auto;">⚠️ Research/demo prototype - not for legal conclusions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_cols[1]:
    with st.popover("Copy hash", width="stretch"):
        st.caption("Canonical source SHA-256")
        st.code(sha, language=None)

st.markdown("")

# ---------------------------------------------------------------------------------------------
# Sidebar Workflow Navigation (Stepper)
# ---------------------------------------------------------------------------------------------

st.sidebar.title("Analyst Workflow")
current_step = ss["step"]

for idx, s in enumerate(STEPS, start=1):
    blocked = step_prerequisite(s, ss)
    label = f"{idx}. {s}" + (" 🔒" if blocked else "")
    if current_step == s:
        st.sidebar.markdown(
            f"""
            <div style="background:#0284c7; color:#fff; padding:8px 12px; border-radius:6px; font-weight:700; font-size:13px; margin-bottom:4px;">
              ➔ {label}
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif st.sidebar.button(label, key=f"nav_{s}", width="stretch"):
        ss["step"] = s
        st.rerun()
    if blocked:
        st.sidebar.caption(blocked)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
st.sidebar.caption(
    f"""
    • **Mode**: Verified cache (fixture manifest {client.manifest_created_at})
    • **Stack**: VAST AI OS • Cosmos • YOLO • W&B
    • **Observability**: Safe Weave sink ({len(client.trace.spans)} spans)
    • **Correlation ID**: `{client.correlation_id}`
    """
)

with st.sidebar.expander("Demo scenario controls (fault injection)"):
    client.ingest_outcome = st.selectbox(
        "Ingest outcome", INGEST_OUTCOMES, index=INGEST_OUTCOMES.index(client.ingest_outcome)
    )
    client.run_outcome = st.selectbox(
        "Reconstruction outcome", RUN_OUTCOMES, index=RUN_OUTCOMES.index(client.run_outcome)
    )
    client.source_tampered = st.checkbox(
        "Simulate altered source bytes", value=client.source_tampered
    )
    client.hallucinate_narrative = st.checkbox(
        "Inject ungrounded W&B narrative", value=client.hallucinate_narrative
    )
    if st.button("Regenerate artifacts (invalidates approval)"):
        client.regenerate_artifacts()
        st.rerun()
    if st.button("Reset demo"):
        reset_client()
        for key in list(ss.keys()):
            del ss[key]
        st.rerun()

# ---------------------------------------------------------------------------------------------
# Dispatch Active Workflow Screen
# ---------------------------------------------------------------------------------------------

VIEWS = {
    "Source": render_upload_view,
    "Ingest": render_ingest_view,
    "Search": render_search_view,
    "Subject": render_subject_view,
    "Probity": render_comparison_view,
    "Review": render_review_view,
    "Report": render_report_view,
}

blocked = step_prerequisite(current_step, ss)
if blocked and current_step != "Report":
    st.subheader(current_step)
    st.info(f"🔒 {blocked}")
    prior = STEPS[max(STEPS.index(current_step) - 1, 0)]
    if st.button(f"⬅ Go to {prior}"):
        ss["step"] = prior
        st.rerun()
else:
    VIEWS[current_step](client)
