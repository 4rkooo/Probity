"""Screen 1: Upload & Case Setup View."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import httpx
import streamlit as st

from probity.api.live_client import LiveApiClient
from probity.api.mock_client import MockApiClient
from probity.ui.components.states import empty_state

MAX_UPLOAD_MB = 250


def _live_client() -> LiveApiClient:
    return LiveApiClient()


def render_upload_view(client: MockApiClient) -> None:
    st.subheader("1. Case Setup & Source Video Selection")

    col1, col2 = st.columns([2, 1])
    live = _live_client()
    live_ok = live.health_ok()

    with col1:
        case_name = st.text_input(
            "Fictional Case Label",
            value=client.case.display_name,
            help="Cases are scoped with synthetic/demo identifiers.",
        )
        owner_alias = st.text_input("Analyst Alias", value=client.case.owner_alias)

        st.info("Purpose: **DEMO_RESEARCH** - synthetic and licensed demo footage only.")

        source_choice = st.radio(
            "Select Source Media:",
            ["Bundled Demo Clip (90-second sedan/plate clip, synthetic)", "Upload New MP4 (Experimental)"],
            index=0,
        )
        bundled = "Bundled" in source_choice
        upload_ok = False
        pending = st.session_state.get("pending_upload")

        if bundled:
            if client.uses_live_upload:
                client.restore_bundled_demo()
            st.success(
                f"✅ Bundled 90s test clip selected "
                f"(`fixtures/demo/source/{client.source_video.original_name}`)"
            )
            st.caption(f"License: {client.source_video.license_note}")
            upload_ok = True
        else:
            st.caption(
                f"Constraints: MP4 (H.264) only, up to {MAX_UPLOAD_MB} MB, synthetic/licensed footage."
            )
            if live_ok:
                st.success(f"Live API ready at `{live.base_url}`")
            else:
                st.error(
                    f"Live API is not reachable at `{live.base_url}`. "
                    "Start it with `make dev` (API + worker), then refresh."
                )

            uploaded = st.file_uploader("Upload MP4", type=["mp4"])
            if uploaded is None and pending is None:
                empty_state("No video yet", "Drop an MP4 above, or select the bundled demo clip.")
            elif uploaded is not None:
                if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                    st.error(
                        f"Refused: `{uploaded.name}` exceeds {MAX_UPLOAD_MB} MB. "
                        "No partial source was kept."
                    )
                    st.session_state.pop("pending_upload", None)
                else:
                    data = uploaded.getvalue()
                    if data[4:8] != b"ftyp":
                        st.error(
                            f"Refused: `{uploaded.name}` is not an MP4 container. "
                            "No partial source was kept."
                        )
                        st.session_state.pop("pending_upload", None)
                    else:
                        digest = hashlib.sha256(data).hexdigest()
                        st.code(digest, language=None)
                        st.session_state["pending_upload"] = {
                            "name": uploaded.name,
                            "data": data,
                            "sha256": digest,
                        }
                        pending = st.session_state["pending_upload"]
                        upload_ok = live_ok
                        if live_ok:
                            st.info("Upload validated. Confirm the case to send it to the live API.")
                        else:
                            st.warning(
                                "Upload hashed locally. Ingesting new uploads requires the live API."
                            )
            elif pending is not None:
                st.code(pending["sha256"], language=None)
                st.caption(f"Ready: `{pending['name']}` ({len(pending['data'])} bytes)")
                upload_ok = live_ok

    with col2:
        st.markdown("### Source Hash Lineage")
        if bundled:
            verified, observed = client.verify_source()
            status = (
                '<div style="color:#4ade80; font-weight:700; margin-top:8px;">'
                "✓ Re-hash matches manifest · immutable source</div>"
                if verified
                else '<div style="color:#f87171; font-weight:700; margin-top:8px;">'
                "✗ SOURCE_HASH_MISMATCH - case quarantined</div>"
            )
            st.markdown(
                f"""
                <div style="background:#1e293b; color:#f8fafc; padding:12px; border-radius:6px; font-size:12px; border:1px solid #334155;">
                  <div style="color:#94a3b8; font-weight:600; text-transform:uppercase; font-size:10px;">Canonical SHA-256</div>
                  <div style="font-family:monospace; margin-top:4px; word-break:break-all; color:#38bdf8;">
                    {client.source_video.sha256}
                  </div>
                  {status}
                  <div style="margin-top:10px; color:#94a3b8; font-size:11px;">
                    Resolution: <strong>{client.source_video.width_px}x{client.source_video.height_px}</strong><br>
                    Duration: <strong>{client.source_video.duration_us // 1000000}s</strong> · {client.source_video.frame_count} frames<br>
                    Container: <strong>{client.source_video.container.upper()} ({client.source_video.video_codec.upper()})</strong>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif pending is not None:
            st.markdown(
                f"""
                <div style="background:#1e293b; color:#f8fafc; padding:12px; border-radius:6px; font-size:12px; border:1px solid #334155;">
                  <div style="color:#94a3b8; font-weight:600; text-transform:uppercase; font-size:10px;">Local SHA-256</div>
                  <div style="font-family:monospace; margin-top:4px; word-break:break-all; color:#38bdf8;">
                    {pending["sha256"]}
                  </div>
                  <div style="margin-top:10px; color:#94a3b8; font-size:11px;">
                    Filename: <strong>{pending["name"]}</strong><br>
                    Bytes: <strong>{len(pending["data"])}</strong><br>
                    Status: <strong>{"ready for live ingest" if live_ok else "waiting for live API"}</strong>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            empty_state("No hash yet", "The hash card appears after a durable copy and re-hash.")

        st.markdown("")
        if st.button(
            "Confirm Case & Proceed to Ingest ➔",
            type="primary",
            width="stretch",
            disabled=not (bundled or upload_ok),
        ):
            if bundled:
                client.create_case(display_name=case_name, owner_alias=owner_alias)
                st.session_state.pop("pending_upload", None)
                st.session_state["case_confirmed"] = True
                st.session_state["step"] = "Ingest"
                st.rerun()
                return

            pending = st.session_state.get("pending_upload")
            if pending is None or not live_ok:
                st.error("Custom upload requires a validated file and a healthy live API.")
                return

            with st.spinner("Creating case and uploading to the live API..."):
                try:
                    case = live.create_case(display_name=case_name, owner_alias=owner_alias)
                    accepted = live.upload_video(
                        case.case_id,
                        pending["data"],
                        original_name=pending["name"],
                    )
                    video = live.get_video(accepted.video_id)
                    tmp = Path(tempfile.mkdtemp(prefix="probity-upload-")) / pending["name"]
                    tmp.write_bytes(pending["data"])
                    client.adopt_live_upload(
                        live_api=live,
                        case=case,
                        video=video,
                        job_id=accepted.job_id,
                        local_path=tmp,
                    )
                except (httpx.HTTPError, OSError, ValueError) as exc:
                    st.error(f"Live upload failed: {live.error_detail(exc)}")
                    return

            st.session_state.pop("pending_upload", None)
            st.session_state["case_confirmed"] = True
            st.session_state["ingest_state"] = "RUNNING"
            st.session_state["ingest_tick"] = 0
            st.session_state["ingest_cancel"] = False
            st.session_state.pop("ingest_job", None)
            st.session_state["step"] = "Ingest"
            st.rerun()
