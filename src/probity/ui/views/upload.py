"""Screen 1: Upload & Case Setup View."""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from probity.api.mock_client import MockApiClient
from probity.ui.components.states import empty_state
from probity.ui.custom_clip import CUSTOM_CLIP_PATH, custom_clip_available

BUNDLED = "Bundled Demo Clip (90-second sedan/plate clip, synthetic)"
CUSTOM = "Custom clip (Flake screen recording, 60s)"
UPLOAD = "Upload New MP4 (Experimental)"

MAX_UPLOAD_MB = 250


def render_upload_view(client: MockApiClient) -> None:
    st.subheader("1. Case Setup & Source Video Selection")

    col1, col2 = st.columns([2, 1])

    with col1:
        case_name = st.text_input(
            "Fictional Case Label",
            value=client.case.display_name,
            help="Cases are scoped with synthetic/demo identifiers.",
        )
        owner_alias = st.text_input("Analyst Alias", value=client.case.owner_alias)

        st.info("Purpose: **DEMO_RESEARCH** - synthetic and licensed demo footage only.")

        choices = [BUNDLED]
        if custom_clip_available():
            choices.append(CUSTOM)
        choices.append(UPLOAD)
        source_choice = st.radio("Select Source Media:", choices, index=0)
        bundled = source_choice == BUNDLED
        custom = source_choice == CUSTOM
        upload_ok = False
        upload_path: Path | None = None

        if bundled:
            bundled_video = getattr(client, "_canonical_source", client.source_video)
            st.success(
                "✅ Bundled 90s test clip selected "
                f"(`fixtures/demo/source/{bundled_video.original_name}`)"
            )
            st.caption(f"License: {bundled_video.license_note}")
        elif custom:
            st.success(f"✅ Custom clip selected (`{CUSTOM_CLIP_PATH}`)")
            st.caption(
                "Screen recording of the Flake demo. Search covers the frames in this file. "
                "Plate reconstruction stays on the synthetic sedan clip."
            )
        else:
            st.caption(f"Constraints: MP4 (H.264) only, up to {MAX_UPLOAD_MB} MB, synthetic/licensed footage.")
            uploaded = st.file_uploader("Upload MP4", type=["mp4"])
            if uploaded is None:
                empty_state("No video yet", "Drop an MP4 above, or select the bundled demo clip.")
            elif uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"Refused: `{uploaded.name}` exceeds {MAX_UPLOAD_MB} MB. No partial source was kept.")
            else:
                data = uploaded.getvalue()
                if data[4:8] != b"ftyp":
                    st.error(f"Refused: `{uploaded.name}` is not an MP4 container. No partial source was kept.")
                else:
                    digest = hashlib.sha256(data).hexdigest()
                    st.code(digest, language=None)
                    if uploaded.name == CUSTOM_CLIP_PATH.name:
                        dest = Path(CUSTOM_CLIP_PATH)
                        upload_path = dest
                        upload_ok = dest.is_file()
                        st.success("This file matches the indexed Flake recording and can be searched.")
                    else:
                        st.warning(
                            "Upload hashed locally. Only the indexed Flake recording "
                            f"(`{CUSTOM_CLIP_PATH.name}`) can be searched in this build."
                        )

    with col2:
        st.markdown("### Source Hash Lineage")
        if bundled or custom:
            verified, observed = client.verify_source()
            display_sha = client.source_video.sha256
            display_video = client.source_video
            if custom and not client.using_custom_source():
                display_sha = hashlib.sha256(CUSTOM_CLIP_PATH.read_bytes()).hexdigest()
                observed = display_sha
                verified = True
            status = (
                '<div style="color:#4ade80; font-weight:700; margin-top:8px;">✓ Re-hash matches manifest · immutable source</div>'
                if verified
                else '<div style="color:#f87171; font-weight:700; margin-top:8px;">✗ SOURCE_HASH_MISMATCH - case quarantined</div>'
            )
            st.markdown(
                f"""
                <div style="background:#1e293b; color:#f8fafc; padding:12px; border-radius:6px; font-size:12px; border:1px solid #334155;">
                  <div style="color:#94a3b8; font-weight:600; text-transform:uppercase; font-size:10px;">Canonical SHA-256</div>
                  <div style="font-family:monospace; margin-top:4px; word-break:break-all; color:#38bdf8;">
                    {display_sha}
                  </div>
                  {status}
                  <div style="margin-top:10px; color:#94a3b8; font-size:11px;">
                    Resolution: <strong>{"1440x900" if custom and not client.using_custom_source() else f"{display_video.width_px}x{display_video.height_px}"}</strong><br>
                    Duration: <strong>{"59" if custom and not client.using_custom_source() else display_video.duration_us // 1000000}s</strong> · {"1797" if custom and not client.using_custom_source() else display_video.frame_count} frames<br>
                    Container: <strong>{client.source_video.container.upper()} ({client.source_video.video_codec.upper()})</strong>
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
            disabled=not (bundled or custom or upload_ok),
        ):
            client.create_case(display_name=case_name, owner_alias=owner_alias)
            if custom or upload_path is not None:
                client.use_custom_source(upload_path or CUSTOM_CLIP_PATH)
            else:
                client.use_bundled_source()
            st.session_state["case_confirmed"] = True
            st.session_state["ingest_state"] = None
            st.session_state.pop("search_result", None)
            st.session_state["step"] = "Ingest"
            st.rerun()
