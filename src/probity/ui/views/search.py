"""Screen 3: Semantic Search View."""

from __future__ import annotations

import html

import streamlit as st

from probity.api.mock_client import MockApiClient
from probity.domain.enums import SearchStatus
from probity.domain.models import SearchEvidence, TimeRangeUs
from probity.ui.components.states import empty_state, partial_bar

EXAMPLES = (
    "Find the blue sedan when its rear plate is most visible",
    "Find vehicle moving left to right",
    "Find the fleeing stolen car",
)


def _indexed_range(client: MockApiClient) -> TimeRangeUs | None:
    job = st.session_state.get("ingest_job")
    if st.session_state.get("ingest_state") == "PARTIAL" and job is not None and job.partial:
        return job.partial.indexed_ranges[0]
    return None


def render_search_view(client: MockApiClient) -> None:
    ss = st.session_state
    st.subheader("3. Semantic Search & Objective Query Planning")

    covered = _indexed_range(client)
    if covered is not None:
        partial_bar(covered.start_pts_us, covered.end_pts_us, client.source_video.duration_us)

    st.caption("Use observable terms (object, color, direction, visibility). Intent and identity terms are filtered.")
    ex_cols = st.columns(len(EXAMPLES))
    for i, example in enumerate(EXAMPLES):
        with ex_cols[i]:
            if st.button(f"Example {i + 1}: '{example}'", key=f"example_{i}", width="stretch"):
                ss["query_input"] = example

    query_text = st.text_input(
        "Natural-Language Query",
        value=ss.get("query_input", EXAMPLES[0]),
        placeholder="e.g. Find the blue sedan when its rear plate is most visible",
    )
    if st.button("🔎 Search", type="primary", disabled=not query_text.strip()):
        ss["query_input"] = query_text
        ss["search_result"] = client.search(query_text, indexed_range=covered)

    result: SearchEvidence | None = ss.get("search_result")
    if result is None:
        empty_state("No search yet", "Enter an objective query and press Search.")
        return

    plan = result.query_plan
    with st.expander("🤖 Weights & Biases Structured Query Plan", expanded=True):
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            st.markdown(f"**Semantic Target:** `{plan.semantic_query or '-'}`")
            st.markdown(f"**Objective Terms:** `{', '.join(plan.objective_terms) or 'none'}`")
        with p_col2:
            st.markdown(f"**Subject Classes:** `{', '.join(plan.subject_classes) or 'none'}`")
            st.markdown(f"**Planner:** `{result.planner_model_id}` ({result.mode})")
        with p_col3:
            if plan.filtered_terms:
                st.warning(
                    f"⚠️ Policy filtered: `{', '.join(plan.filtered_terms)}` "
                    f"({', '.join(plan.policy_reason_codes)}) - non-evidentiary intent removed"
                )
            else:
                st.success("✅ Policy gate: objective terms only")

    if result.status is SearchStatus.NEEDS_CLARIFICATION:
        empty_state("Query needs clarification", "Describe something observable, e.g. a vehicle color or a sign.")
        return
    if result.status is SearchStatus.NO_RESULTS or not result.results:
        empty_state(
            "No indexed moments matched",
            "Try observable terms such as 'blue sedan', 'rear plate', or 'street sign'.",
        )
        return

    st.markdown("### Ranked Grounded Evidence Results")
    for res in result.results:
        badges = " ".join(
            f'<span style="background:#334155; color:#e2e8f0; padding:1px 8px; border-radius:9999px; font-size:11px;">'
            f"{html.escape(c.class_name)} ×{c.detection_count} max {c.max_confidence:.2f}</span>"
            for c in res.detected_classes
        )
        comp = res.components
        warn = (
            '<div style="margin-top:8px; color:#f59e0b; font-size:11px; font-weight:700;">⚠ PARTIAL INDEX - results limited to the covered time range</div>'
            if covered is not None or res.outside_indexed_range_warning
            else ""
        )
        st.markdown(
            f"""
            <div style="background:#1e293b; border:1px solid #334155; border-radius:8px; padding:16px; margin-bottom:8px;">
              <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                <div style="font-weight:700; font-size:16px; color:#38bdf8;">
                  Rank #{res.rank} - Segment {res.segment_id.split(':')[-1]} ({res.start_pts_us / 1e6:.2f}s - {res.end_pts_us / 1e6:.2f}s)
                </div>
                <div style="background:#0369a1; color:#f8fafc; font-weight:700; padding:2px 10px; border-radius:9999px; font-size:12px;">
                  Score {res.score:.3f}
                </div>
              </div>
              <div style="margin-top:8px; font-size:13px; color:#e2e8f0;">{html.escape(res.description)}</div>
              <div style="margin-top:6px; font-size:12px; color:#cbd5e1;"><em>{html.escape(res.explanation)}</em>
                <span style="color:#94a3b8;">({res.explanation_source})</span></div>
              <div style="margin-top:8px;">{badges}</div>
              <div style="margin-top:8px; display:flex; flex-wrap:wrap; gap:16px; font-size:11px; color:#94a3b8;">
                <div>PTS: <strong>{res.start_pts_us} - {res.end_pts_us} µs</strong></div>
                <div>Frames: <strong>{res.start_frame} - {res.end_frame}</strong></div>
                <div>cosine {comp.cosine:.2f} · lexical {comp.lexical_overlap:.2f} · detection {comp.detection_match:.2f} · visibility {comp.visibility_match:.2f}</div>
                <div>Evidence IDs: <strong>{len(res.evidence_detection_ids)} detections</strong></div>
              </div>
              {warn}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            f"Select Result #{res.rank} for Tracking & Reconstruction ➔",
            key=f"select_res_{res.rank}",
            type="primary" if res.rank == 1 else "secondary",
        ):
            ss["selected_segment"] = res.segment_id
            ss["seek_us"] = res.start_pts_us
            ss["step"] = "Subject"
            st.rerun()
