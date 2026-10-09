"""Screen 2: Ingestion Progress View.

Polls the ingest job while it is visible and renders every job state: queued/running with
named stages, PARTIAL coverage, FAILED (retry or verified-cache fallback), and cooperative
cancellation.
"""

from __future__ import annotations

import time

import streamlit as st

from probity.api.mock_client import INGEST_STAGE_PLAN, MockApiClient
from probity.domain.enums import TERMINAL_JOB_STATES, JobStage, JobState
from probity.domain.models import JobView
from probity.ui.components.states import (
    cancellation_notice,
    failure_card,
    loading_state,
    partial_bar,
)
from probity.ui.state import poll_interval

STAGE_TEXT = {
    JobStage.VALIDATE: "Validating container",
    JobStage.HASH: "Hashing source bytes",
    JobStage.EXTRACT: "Extracting frames and PTS manifest",
    JobStage.DESCRIBE: "Describing segment {n} of {t} (NVIDIA Cosmos)",
    JobStage.EMBED: "Embedding segment {n} of {t} (Cosmos embeddings)",
    JobStage.DETECT: "Detecting objects in segment {n} of {t} (YOLO)",
    JobStage.INDEX: "Indexing segment {n} of {t} (VAST AI OS)",
}


def _stage_label(job: JobView) -> str:
    if job.state is JobState.CREATED:
        return "Job created"
    if job.state is JobState.QUEUED:
        return "Queued - waiting for a worker"
    if job.stage is None:
        return str(job.state)
    return STAGE_TEXT[job.stage].format(n=job.completed_units + 1, t=job.total_units)


def _stage_strip(job: JobView) -> None:
    order = [stage for stage, _ in INGEST_STAGE_PLAN]
    current = order.index(job.stage) if job.stage in order else -1
    terminal_ok = job.state in (JobState.SUCCEEDED,)
    cols = st.columns(len(order))
    for i, stage in enumerate(order):
        if terminal_ok or i < current:
            mark, color = "✓ done", "#4ade80"
        elif i == current and job.state is JobState.RUNNING:
            mark, color = "● running", "#38bdf8"
        elif i == current:
            mark, color = f"■ {job.state}", "#f59e0b"
        else:
            mark, color = "○ pending", "#64748b"
        with cols[i]:
            st.markdown(
                f"""
                <div style="background:#1e293b; color:{color}; padding:8px 4px; text-align:center; border-radius:4px; font-size:11px; font-weight:700; border:1px solid #334155;">
                  {stage.name}<br><span style="font-weight:500;">{mark}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _restart(client: MockApiClient) -> None:
    ss = st.session_state
    ss["ingest_state"] = "RUNNING"
    ss["ingest_tick"] = 0
    ss["ingest_cancel"] = False
    ss.pop("ingest_job", None)


def render_ingest_view(client: MockApiClient) -> None:
    ss = st.session_state
    st.subheader("2. Ingestion Progress & Multi-Model Indexing")

    state = ss.get("ingest_state")
    if state is None:
        if client.using_custom_source():
            st.markdown(
                f"`{client.source_video.original_name}` is hashed. "
                "Start ingestion to search the 8 indexed segments from this recording."
            )
        else:
            st.markdown("The bundled clip is stored and hashed. Start ingestion to index it for search.")
        if st.button("▶ Start Ingestion", type="primary"):
            _restart(client)
            st.rerun()
        return

    col1, col2 = st.columns([3, 1])

    if state == "RUNNING":
        seq = client.cancel_job_sequence() if ss.get("ingest_cancel") else client.ingest_job_sequence()
        tick = min(ss.get("ingest_tick", 0), len(seq) - 1)
        job = seq[tick]
        with col1:
            _stage_strip(job)
            if job.state is JobState.CANCELLING:
                cancellation_notice("CANCELLING")
            else:
                loading_state(
                    _stage_label(job),
                    job.completed_units / max(job.total_units, 1),
                    f"{job.completed_units} / {job.total_units} units committed",
                )
        with col2:
            st.markdown("### Job Status")
            st.markdown(f"**State:** `{job.state}`  \n**Attempt:** {job.attempt}")
            if job.state is not JobState.CANCELLING and st.button(
                "■ Cancel Ingestion", width="stretch"
            ):
                ss["ingest_cancel"] = True
                ss["ingest_tick"] = 0
                st.rerun()

        if job.state in TERMINAL_JOB_STATES:
            ss["ingest_state"] = str(job.state)
            ss["ingest_job"] = job
            st.rerun()
        time.sleep(poll_interval(ss))
        ss["ingest_tick"] = tick + 1
        st.rerun()
        return

    job: JobView = ss.get("ingest_job") or client.ingest_job_sequence()[-1]

    with col1:
        _stage_strip(job)
        if state == "SUCCEEDED":
            st.progress(
                1.0,
                text=f"Ingestion succeeded: {job.completed_units} / {job.total_units} units indexed (100%)",
            )
        elif state == "PARTIAL" and job.partial:
            rng = job.partial.indexed_ranges[0]
            partial_bar(rng.start_pts_us, rng.end_pts_us, client.source_video.duration_us)
            st.caption(
                f"{len(job.partial.committed_segment_ids)} segments searchable; "
                f"{len(job.partial.failed_segment_ids)} failed. Search is limited to the covered range."
            )
        elif state == "FAILED" and job.error:
            is_timeout = job.error.code == "SPONSOR_TIMEOUT"
            failure_card(
                title="Sponsor timeout" if is_timeout else "Ingestion interrupted",
                code=str(job.error.code),
                message=job.error.message,
                correlation_id=job.correlation_id,
                retryable=job.error.retryable,
            )
            if is_timeout:
                if st.button("Continue with verified demo cache", type="primary"):
                    client.ingest_outcome = "SUCCEEDED"
                    _restart(client)
                    st.rerun()
            elif job.error.retryable and st.button("↻ Retry Ingestion", type="primary"):
                client.ingest_outcome = "SUCCEEDED"
                _restart(client)
                st.rerun()
        elif state == "CANCELLED":
            cancellation_notice("CANCELLED")
            if job.partial:
                rng = job.partial.indexed_ranges[0]
                partial_bar(rng.start_pts_us, rng.end_pts_us, client.source_video.duration_us)
            if st.button("↻ Restart Ingestion", type="primary"):
                _restart(client)
                st.rerun()

        st.markdown("#### Sponsor Attribution")
        st.markdown(
            """
            - **VAST AI OS**: Canonical source storage, segment metadata, and vector index.
            - **NVIDIA Cosmos**: Segment descriptions & semantic embeddings.
            - **YOLOv8 + ByteTrack**: Object detection and rigid continuity tracking.
            - **Weights & Biases**: Reasoner query planning and safe observability sink.
            """
        )

        if state in ("SUCCEEDED", "PARTIAL"):
            segments = client.list_segments()
            with st.expander(f"View Indexed Segments ({len(segments)} segments)", expanded=False):
                for s in segments:
                    st.caption(
                        f"**Segment {s.ordinal}** ({s.start_pts_us / 1e6:.1f}s - {s.end_pts_us / 1e6:.1f}s): {s.description}"
                    )

    with col2:
        st.markdown("### Job Status")
        rng_text = (
            f"{job.partial.indexed_ranges[0].start_pts_us / 1e6:.1f}s - {job.partial.indexed_ranges[0].end_pts_us / 1e6:.1f}s"
            if job.partial
            else (
                f"0.0s - {client.source_video.duration_us / 1e6:.1f}s"
                if state == "SUCCEEDED"
                else "none"
            )
        )
        st.markdown(
            f"""
            <div style="background:#1e293b; color:#f8fafc; padding:12px; border-radius:6px; font-size:12px; border:1px solid #334155;">
              <div style="color:#94a3b8; font-size:10px; font-weight:600; text-transform:uppercase;">Status</div>
              <div style="font-weight:700; font-size:14px; margin-top:2px;">{state}</div>
              <div style="margin-top:8px;">Units: <strong>{job.completed_units} / {job.total_units}</strong></div>
              <div>Indexed Range: <strong>{rng_text}</strong></div>
              <div>Mode: <strong>{job.mode}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        if state in ("SUCCEEDED", "PARTIAL") and st.button(
            "Proceed to Search ➔", type="primary", width="stretch"
        ):
            ss["step"] = "Search"
            st.rerun()
