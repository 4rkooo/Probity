"""Screen 4: Tracked-Subject & Target Frame Selection View."""

from __future__ import annotations

import streamlit as st

from probity.api.mock_client import MockApiClient
from probity.domain.enums import TrackState
from probity.domain.ids import parse_frame_id


def _render_custom_subject(client: MockApiClient) -> None:
    """Playback for a user clip. Plate tracking belongs to the synthetic sedan fixture."""
    ss = st.session_state
    seek_us = int(ss.get("seek_us") or 0)
    selected = str(ss.get("selected_segment") or "")
    match = next((segment for segment in client.list_segments() if segment.segment_id == selected), None)
    st.video(str(client.source_path()), start_time=int(seek_us // 1_000_000))
    if match is not None and match.description:
        shown = match.description.replace("$", r"\$")
        st.info(
            f"**Segment {match.ordinal}** · {match.start_pts_us / 1e6:.1f}s–{match.end_pts_us / 1e6:.1f}s  \n"
            f"{shown}"
        )
    st.warning(
        "This recording is a screen capture, not the synthetic sedan clip. "
        "There is no license-plate track to reconstruct. Search results above are the evidence."
    )


SUBJECT_OPTIONS = (
    "Detected LICENSE_PLATE box (YOLO detection at seed frame)",
    "Analyst-drawn rigid ROI (requires detector-backed continuity)",
)


def render_subject_view(client: MockApiClient) -> None:
    ss = st.session_state
    st.subheader("4. Tracked-Subject Confirmation & Target Frame Selection")
    if client.using_custom_source():
        _render_custom_subject(client)
        return

    subject_choice = st.radio("Subject", SUBJECT_OPTIONS, index=0, horizontal=True)
    track = client.get_track(confirmed=subject_choice == SUBJECT_OPTIONS[0])
    seed_frame = parse_frame_id(track.seed_frame_id)[1]

    col1, col2 = st.columns([2, 1])

    with col1:
        seek_us = ss.get("seek_us", track.window_start_us)
        st.markdown(f"#### Source Player (seeked to cited time {seek_us / 1e6:.2f}s)")
        st.video(str(client.source_path()), start_time=int(seek_us // 1_000_000))
        st.caption(
            f"Segment `{str(ss.get('selected_segment', '')).split(':')[-1]}` · "
            "browser playback is a convenience view; lossless stills are canonical."
        )

        st.info(
            f"**Subject Type:** `{track.subject_type}` | **Seed Frame:** `f{seed_frame}` | "
            f"**Seed BBox:** `{list(track.seed_bbox_px)}` | **Continuity:** `{track.continuity_score * 100:.1f}%`"
        )

        st.markdown(f"#### Track Observations ({track.window_start_us / 1e6:.2f}s - {track.window_end_us / 1e6:.2f}s window)")
        obs_cols = st.columns(max(len(track.observations), 1))
        for idx, obs in enumerate(track.observations):
            with obs_cols[idx]:
                frame_no = parse_frame_id(obs.frame_id)[1]
                accepted = "ACCEPTED" if obs.accepted else "REJECTED"
                color = "#4ade80" if obs.accepted else "#f87171"
                reason = f"<div style='color:#fca5a5; font-size:9px;'>{obs.reason_code}</div>" if obs.reason_code else ""
                st.markdown(
                    f"""
                    <div style="background:#1e293b; border:1px solid #334155; border-radius:6px; padding:6px; text-align:center; font-size:11px; color:#e2e8f0;">
                      <div style="font-weight:700; color:#38bdf8;">f{frame_no}</div>
                      <div style="color:#94a3b8; font-size:10px;">{obs.pts_us / 1e6:.2f}s · {obs.confidence:.2f}</div>
                      <div style="color:{color}; font-weight:700; margin-top:4px;">{'✓' if obs.accepted else '✗'} {accepted}</div>
                      {reason}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        accepted_frames = [
            parse_frame_id(o.frame_id)[1] for o in track.observations if o.accepted
        ]
        default_idx = accepted_frames.index(seed_frame) if seed_frame in accepted_frames else 0
        st.markdown("")
        target_frame = st.selectbox(
            "Reconstruction target frame (accepted observations only):",
            accepted_frames,
            index=default_idx,
            format_func=lambda f: f"Frame {f}" + (" - seed frame, best target geometry" if f == seed_frame else ""),
            disabled=track.state is not TrackState.CONFIRMED,
        )

    with col2:
        st.markdown("### Tracking Integrity Gate")
        confirmed = track.state is TrackState.CONFIRMED
        st.markdown(
            f"""
            <div style="background:#1e293b; color:#f8fafc; padding:12px; border-radius:6px; font-size:12px; border:1px solid #334155;">
              <div style="color:#94a3b8; font-size:10px; font-weight:600; text-transform:uppercase;">Tracker State</div>
              <div style="color:{'#4ade80' if confirmed else '#f59e0b'}; font-weight:700; font-size:14px; margin-top:2px;">{track.state}</div>
              <div style="margin-top:8px;">Tracker: <strong>{track.tracker} {track.tracker_version}</strong></div>
              <div>Mean Conf: <strong>{track.mean_confidence:.2f}</strong></div>
              <div>Detector-backed observations: <strong>{track.detector_observation_count}</strong></div>
              <div>Track ID: <code>{track.track_id[:13]}…</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        if not confirmed:
            st.warning(
                "Reconstruction disabled: track not confirmed "
                f"({', '.join(track.reason_codes)}). Only plate/sign detections or a rigid ROI with "
                "detector-backed continuity may be reconstructed."
            )
        if st.button(
            "Run Probity Reconstruction ➔",
            type="primary",
            width="stretch",
            disabled=not confirmed,
        ):
            ss["target_frame"] = int(target_frame)
            ss["run_outcome"] = None
            ss["recon_tick"] = 0
            ss["recon_done"] = False
            ss["step"] = "Probity"
            st.rerun()
