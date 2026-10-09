"""Screen 4: Tracked-Subject & Target Frame Selection View."""

from __future__ import annotations

import streamlit as st
from PIL import Image, ImageDraw

from probity.api.mock_client import MockApiClient
from probity.domain.enums import TrackState
from probity.domain.ids import parse_frame_id

SUBJECT_OPTIONS = (
    "Detected LICENSE_PLATE box (YOLO detection at seed frame)",
    "Analyst-drawn rigid ROI (requires detector-backed continuity)",
)


def _frame_with_box(client: MockApiClient, frame_number: int, bbox, accepted: bool) -> Image.Image:
    with Image.open(client.frame_path(frame_number)) as img:
        frame = img.convert("RGB")
    draw = ImageDraw.Draw(frame)
    draw.rectangle(list(bbox), outline=(0, 255, 255) if accepted else (248, 113, 113), width=2)
    return frame


def render_subject_view(client: MockApiClient) -> None:
    ss = st.session_state
    st.subheader("4. Tracked-Subject Confirmation & Target Frame Selection")

    subject_choice = st.radio("Subject", SUBJECT_OPTIONS, index=0, horizontal=True)
    track = client.get_track(confirmed=subject_choice == SUBJECT_OPTIONS[0])
    confirmed = track.state is TrackState.CONFIRMED

    st.info(
        f"Reconstruction window: **{client.window.fixture_id}** - Person 2's synthetic evaluation "
        "window with real Probity output. The bundled demo clip has not been re-rendered for "
        "reconstruction yet, so the tracked subject below comes from this window."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        if confirmed:
            observations = sorted(track.observations, key=lambda o: o.pts_us)
            by_frame = {parse_frame_id(o.frame_id)[1]: o for o in observations}
            frame_numbers = list(by_frame)
            seed_frame = parse_frame_id(track.seed_frame_id)[1]
            shown = st.select_slider(
                "Frame step (track window)",
                options=frame_numbers,
                value=ss.get("subject_frame", seed_frame),
                format_func=lambda f: f"f{f}",
            )
            ss["subject_frame"] = shown
            obs = by_frame[shown]
            st.image(
                _frame_with_box(client, shown, obs.bbox_px, obs.accepted),
                caption=(
                    f"f{shown} @ {obs.pts_us / 1e6:.3f}s · {obs.source} · confidence "
                    f"{obs.confidence:.2f} · {'ACCEPTED' if obs.accepted else f'REJECTED ({obs.reason_code})'}"
                ),
                width="stretch",
            )

            accepted = [o for o in observations if o.accepted]
            rejected = [o for o in observations if not o.accepted]
            st.markdown(
                f"**Track observations:** {len(accepted)} accepted · {len(rejected)} rejected "
                f"({track.window_start_us / 1e6:.2f}s - {track.window_end_us / 1e6:.2f}s)"
            )
            with st.expander("Observation strip"):
                st.dataframe(
                    [
                        {
                            "Frame": f"f{parse_frame_id(o.frame_id)[1]}",
                            "PTS (s)": round(o.pts_us / 1e6, 3),
                            "Source": str(o.source),
                            "Confidence": o.confidence,
                            "Status": "ACCEPTED" if o.accepted else "REJECTED",
                            "Reason": str(o.reason_code or ""),
                        }
                        for o in observations
                    ],
                    hide_index=True,
                    width="stretch",
                )

            accepted_frames = [parse_frame_id(o.frame_id)[1] for o in accepted]
            default_idx = accepted_frames.index(seed_frame) if seed_frame in accepted_frames else 0
            target_frame = st.selectbox(
                "Reconstruction target frame (accepted observations only):",
                accepted_frames,
                index=default_idx,
                format_func=lambda f: f"Frame {f}"
                + (" - evaluated target (verified cached run)" if f == client.target_frame_number else ""),
            )
        else:
            target_frame = None
            st.image(str(client.target_frame_path()), caption="Seed frame (no confirmed track)")

        with st.expander("Cited search moment in the demo clip"):
            seek_us = ss.get("seek_us", 0)
            st.video(str(client.source_path()), start_time=int(seek_us // 1_000_000))
            st.caption(f"Seeked to {seek_us / 1e6:.2f}s from the selected search result.")

    with col2:
        st.markdown("### Tracking Integrity Gate")
        st.markdown(
            f"""
            <div style="background:#1e293b; color:#f8fafc; padding:12px; border-radius:6px; font-size:12px; border:1px solid #334155;">
              <div style="color:#94a3b8; font-size:10px; font-weight:600; text-transform:uppercase;">Tracker State</div>
              <div style="color:{'#4ade80' if confirmed else '#f59e0b'}; font-weight:700; font-size:14px; margin-top:2px;">{track.state}</div>
              <div style="margin-top:8px;">Subject: <strong>{track.subject_type}</strong></div>
              <div>Tracker: <strong>{track.tracker} {track.tracker_version}</strong></div>
              <div>Mean Conf: <strong>{track.mean_confidence:.2f}</strong> · Continuity: <strong>{track.continuity_score * 100:.1f}%</strong></div>
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
