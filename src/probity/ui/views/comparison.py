"""Screen 5: TrueFrame Reconstruction & Provenance Comparison View."""

from __future__ import annotations

import time
from fractions import Fraction

import streamlit as st
from PIL import Image

from probity.api.mock_client import MockApiClient
from probity.domain.enums import TERMINAL_JOB_STATES, JobStage
from probity.domain.ids import parse_frame_id
from probity.domain.models import ReconstructionRun
from probity.ui.components.provenance_canvas import load_b64, render_provenance_canvas
from probity.ui.components.states import failure_card, loading_state, refusal_panel
from probity.ui.state import poll_interval


def _run_progress(client: MockApiClient, outcome: str) -> None:
    """Poll the reconstruction job with named stages until it is terminal."""
    ss = st.session_state
    seq = client.reconstruction_job_sequence(outcome)
    tick = min(ss.get("recon_tick", 0), len(seq) - 1)
    job = seq[tick]
    if job.state in TERMINAL_JOB_STATES:
        ss["run_outcome"] = outcome
        ss["recon_job"] = job
        ss["recon_done"] = True
        st.rerun()
    if job.stage is JobStage.ALIGN:
        label = f"Aligning donor {job.completed_units + 1} of {job.total_units}"
    else:
        label = "Fusing accepted tiles and validating provenance"
    loading_state(label, job.completed_units / max(job.total_units, 1), f"Target frame f{ss['target_frame']}")
    time.sleep(poll_interval(ss))
    ss["recon_tick"] = tick + 1
    st.rerun()


def _choose_another_target() -> None:
    ss = st.session_state
    if st.button("⬅ Choose a different target frame", type="primary"):
        ss["target_frame"] = None
        ss["run_outcome"] = None
        ss["recon_done"] = False
        ss["step"] = "Subject"
        st.rerun()


def _baseline_image(client: MockApiClient, run: ReconstructionRun) -> Image.Image:
    """Conventional 4x Lanczos upscale of the subject crop, a stand-in display baseline."""
    x1, y1, x2, y2 = run.target_bbox_px
    pad = 12
    with Image.open(client.get_asset_path("target_f417.png")) as img:
        crop = img.crop((x1 - pad, y1 - pad, x2 + pad, y2 + pad))
        return crop.resize((crop.width * 4, crop.height * 4), Image.Resampling.LANCZOS)


def _pixel_inspector(client: MockApiClient, run: ReconstructionRun) -> None:
    """Server-side inspector using the provenance endpoint semantics (exact NPZ lookup)."""
    ss = st.session_state
    st.markdown("#### Pixel provenance lookup (authoritative NPZ)")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        x = st.number_input("x", min_value=0, max_value=client.provenance_shape()[0] - 1, value=230, key="inspect_x")
    with c2:
        y = st.number_input("y", min_value=0, max_value=client.provenance_shape()[1] - 1, value=481, key="inspect_y")
    origin = client.resolve_pixel(int(x), int(y), run.run_id)
    if origin is None:
        st.warning("No provenance record for this pixel.")
        return
    codes = {d.decision_id: str(d.rule_code) for d in client.list_decisions(run.run_id)}
    _, frame_no = parse_frame_id(origin.source_frame_id)
    with c3:
        st.markdown(
            f"**{origin.provenance_class}** at ({origin.x}, {origin.y}) ← "
            f"**f{frame_no}** @ **{origin.source_pts_us / 1e6:.3f}s** "
            f"(`{origin.source_pts_us}` µs), source ({origin.source_x:.1f}, {origin.source_y:.1f}) · "
            f"{origin.alignment_method} `{origin.transform_id or 'identity'}` · reasons: "
            f"{', '.join(codes.get(i, i[:8]) for i in origin.decision_ids) or 'n/a'}"
        )
        if origin.source_index != 0 and st.button("🔍 Open source frame", key="open_source_py"):
            ss["open_source"] = {
                "frame": frame_no,
                "pts_us": origin.source_pts_us,
                "x": origin.source_x,
                "y": origin.source_y,
            }
    opened = ss.get("open_source")
    if opened:
        o1, o2 = st.columns(2)
        with o1:
            st.markdown(f"**Source donor f{opened['frame']} @ {opened['pts_us'] / 1e6:.3f}s (lossless still)**")
            name = client.frame_asset_name(opened["frame"])
            if name:
                with Image.open(client.get_asset_path(name)) as img:
                    sx, sy = int(opened["x"]), int(opened["y"])
                    region = img.crop((sx - 40, sy - 20, sx + 40, sy + 20)).resize((320, 160), Image.Resampling.NEAREST)
                st.image(region, caption=f"Source region around ({sx}, {sy})")
        with o2:
            st.markdown(f"**Original player seeked to {opened['pts_us'] / 1e6:.3f}s**")
            st.video(str(client.source_path()), start_time=int(opened["pts_us"] // 1_000_000))


def render_comparison_view(client: MockApiClient) -> None:
    ss = st.session_state
    st.subheader("5. TrueFrame Reconstruction & Provenance Comparison")

    target_frame = ss.get("target_frame", 417)
    outcome = client.outcome_for_target(int(target_frame))
    if not ss.get("recon_done"):
        _run_progress(client, outcome)
        return

    outcome = ss.get("run_outcome") or outcome
    run = client.get_reconstruction(outcome=outcome)

    if outcome == "REFUSED":
        refusal_panel([str(r) for r in run.refusal_reasons], client.list_decisions(run.run_id))
        st.markdown(f"**Original target frame f{parse_frame_id(run.target_frame_id)[1]} - unchanged**")
        st.video(str(client.source_path()), start_time=int(run.target_pts_us // 1_000_000))
        st.caption("; ".join(run.uncertainty))
        _choose_another_target()
        return

    if outcome == "FAILED":
        job = ss.get("recon_job")
        err = job.error if job is not None else None
        failure_card(
            title="Reconstruction failed",
            code=str(err.code) if err else "INTERNAL_ERROR",
            message=err.message if err else "Run failed before producing artifacts.",
            correlation_id=job.correlation_id if job is not None else client.correlation_id,
            retryable=bool(err and err.retryable),
        )
        if err and err.retryable and st.button("↻ Retry reconstruction"):
            ss["recon_done"] = False
            ss["recon_tick"] = 0
            st.rerun()
        _choose_another_target()
        return

    lut = list(run.provenance.source_lut) if run.provenance else []
    donor_uris = {}
    for entry in lut:
        if entry.role == "DONOR":
            name = client.frame_asset_name(entry.frame_number)
            if name:
                donor_uris[entry.index] = load_b64(client.get_asset_path(name))
    first_borrowed = next((r for r in client.provenance_exceptions() if r[2] == 1), None)
    initial = (int(first_borrowed[0]) + 8, int(first_borrowed[1]) + 3) if first_borrowed else (0, 0)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Provenance Inspector",
        "📍 Pixel Lookup & Open Source",
        "⚖️ Conventional Upscaler (Baseline)",
        "🖼️ Donor Gallery",
    ])

    with tab1:
        render_provenance_canvas(
            target_img_uri=load_b64(client.get_asset_path("target_f417.png")),
            result_img_uri=load_b64(client.get_asset_path("result.png")),
            donor_img_uris=donor_uris,
            source_video_uri=load_b64(client.source_path()),
            lut_entries=[e.model_dump(mode="json") for e in lut],
            exceptions=client.provenance_exceptions(),
            frame_size=client.provenance_shape(),
            subject_bbox=run.target_bbox_px,
            decision_codes={d.decision_id: str(d.rule_code) for d in client.list_decisions(run.run_id)},
            target_pts_us=run.target_pts_us,
            nominal_fps=float(Fraction(client.source_video.nominal_fps)),
            initial_pixel=initial,
        )
        st.caption(
            "Hover the result still to inspect any pixel; click to pin it; **Open source** seeks both "
            "players to the donor frame and outlines the source region."
        )

    with tab2:
        _pixel_inspector(client, run)

    with tab3:
        st.markdown(
            """
            <div style="background:#450a0a; color:#fecaca; border-left:4px solid #ef4444; padding:10px 14px; border-radius:4px; font-size:12px; font-weight:600; margin-bottom:14px;">
              ⚠️ BASELINE IS NON-EVIDENTIARY: conventional upscalers synthesize detail that cannot be traced to any recorded camera pixel. The baseline never enters the donor pool.
            </div>
            """,
            unsafe_allow_html=True,
        )
        b1, b2 = st.columns(2)
        with b1:
            x1, y1, x2, y2 = run.target_bbox_px
            with Image.open(client.get_asset_path("result.png")) as img:
                res_crop = img.crop((x1 - 12, y1 - 12, x2 + 12, y2 + 12))
                st.image(
                    res_crop.resize((res_crop.width * 4, res_crop.height * 4), Image.Resampling.NEAREST),
                    caption="Probity TrueFrame result (4x nearest-neighbour view, every pixel traced)",
                )
        with b2:
            st.image(
                _baseline_image(client, run),
                caption="Conventional 4x Lanczos upscale - NON-EVIDENTIARY display baseline "
                "(Real-ESRGAN asset not bundled in this fixture set)",
            )

    with tab4:
        st.markdown("**Accepted observations**")
        g_cols = st.columns(max(len(lut), 1))
        for i, entry in enumerate(lut):
            with g_cols[i]:
                name = client.frame_asset_name(entry.frame_number)
                if name:
                    st.image(str(client.get_asset_path(name)))
                st.caption(
                    f"f{entry.frame_number} · {entry.role} · {entry.pts_us / 1e6:.3f}s · {entry.alignment_method}"
                )
        rejected = [d for d in client.list_decisions(run.run_id) if str(d.outcome) == "REJECT"]
        st.markdown("**Rejected donors and why**")
        if rejected:
            for d in rejected:
                st.markdown(
                    f"- `{d.rule_code}` on `{d.subject_ref.split(':')[-1]}`: {d.reason} "
                    f"(observed {d.observed} {d.operator} {d.threshold} {d.units or ''})"
                )
        else:
            st.caption("No donors were rejected in this run.")

    st.markdown("---")
    _, c_btn2 = st.columns([3, 1])
    with c_btn2:
        if st.button("Proceed to Integrity & Review ➔", type="primary", width="stretch"):
            ss["step"] = "Review"
            st.rerun()
