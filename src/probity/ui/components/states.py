"""Reusable renderings of the required UI states (design section 11, "State language").

Loading, empty, failure, refusal, cached, partial, and cancellation states all go through
these helpers so every screen speaks the same language. None of them render stack traces,
and none rely on color alone: every state carries a text label.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

import streamlit as st

from probity.domain.models import PolicyDecision

AMBER_BG = "#78350f"
AMBER_FG = "#fef3c7"
AMBER_EDGE = "#f59e0b"


def _esc(value: object) -> str:
    return html.escape(str(value))


def loading_state(stage_label: str, fraction: float, detail: str = "") -> None:
    """Named stage plus progress and skeleton rows; never a bare spinner."""
    st.markdown(
        f"""
        <div style="background:#1e293b; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:12px;">
          <div style="font-size:11px; color:#94a3b8; font-weight:700; text-transform:uppercase;">In progress</div>
          <div style="font-size:15px; font-weight:700; margin-top:2px;">{_esc(stage_label)}</div>
          <div style="font-size:12px; color:#94a3b8;">{_esc(detail)}</div>
          <div style="margin-top:10px; height:10px; background:#334155; border-radius:4px; width:90%;"></div>
          <div style="margin-top:6px; height:10px; background:#334155; border-radius:4px; width:70%;"></div>
          <div style="margin-top:6px; height:10px; background:#334155; border-radius:4px; width:80%;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(fraction, 0.0), 1.0), text=stage_label)


def empty_state(title: str, message: str) -> None:
    st.markdown(
        f"""
        <div style="border:1px dashed #64748b; border-radius:6px; padding:16px; text-align:center; color:#64748b;">
          <div style="font-size:15px; font-weight:700;">{_esc(title)}</div>
          <div style="font-size:12px; margin-top:4px;">{_esc(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def failure_card(
    *,
    title: str,
    code: str,
    message: str,
    correlation_id: str,
    retryable: bool,
) -> None:
    """Plain-language failure with correlation ID and retryability; callers add the action."""
    retry_text = "Retryable: yes - safe to retry with the same input" if retryable else "Retryable: no"
    st.markdown(
        f"""
        <div style="background:#450a0a; color:#fecaca; border-left:6px solid #ef4444; padding:14px; border-radius:6px;">
          <div style="font-size:15px; font-weight:700;">FAILED - {_esc(title)}</div>
          <div style="font-size:13px; margin-top:6px;">{_esc(message)}</div>
          <div style="font-size:11px; margin-top:8px; font-family:monospace;">
            Error code: {_esc(code)}<br>
            Correlation ID: {_esc(correlation_id)}<br>
            {_esc(retry_text)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def refusal_panel(reasons: Sequence[str], decisions: Sequence[PolicyDecision] = ()) -> None:
    """Amber refusal panel: never looks like a crash; lists rule-coded reasons."""
    reason_items = "".join(f"<li><code>{_esc(r)}</code></li>" for r in reasons)
    rejected = [d for d in decisions if str(d.outcome) == "REJECT"]
    decision_items = "".join(
        f"<li><code>{_esc(d.rule_code)}</code> on {_esc(d.subject_ref.split(':')[-1])}: "
        f"{_esc(d.reason)} (observed {_esc(d.observed)} {_esc(d.operator)} {_esc(d.threshold)})</li>"
        for d in rejected
    )
    st.markdown(
        f"""
        <div style="background:{AMBER_BG}; color:{AMBER_FG}; border-left:6px solid {AMBER_EDGE}; padding:16px; border-radius:6px; margin-bottom:16px;">
          <h3 style="margin:0 0 8px 0; color:{AMBER_FG};">No defensible enhancement produced</h3>
          <div style="font-size:13px;">TrueFrame refused to alter the target frame. The original source frame is shown unchanged.</div>
          <div style="font-size:12px; margin-top:8px; font-weight:700;">Refusal reasons</div>
          <ul style="margin:4px 0 0 18px; font-size:12px;">{reason_items}</ul>
          {f'<div style="font-size:12px; margin-top:8px; font-weight:700;">Rejected evidence</div><ul style="margin:4px 0 0 18px; font-size:12px;">{decision_items}</ul>' if decision_items else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def partial_bar(covered_start_us: int, covered_end_us: int, total_us: int) -> None:
    """Striped coverage bar labeling the exact indexed time range."""
    pct = 100.0 * (covered_end_us - covered_start_us) / max(total_us, 1)
    left = 100.0 * covered_start_us / max(total_us, 1)
    st.markdown(
        f"""
        <div style="font-size:12px; font-weight:700; color:#b45309;">PARTIAL - indexed {covered_start_us / 1e6:.1f}s to {covered_end_us / 1e6:.1f}s of {total_us / 1e6:.1f}s ({pct:.0f}%)</div>
        <div style="position:relative; height:14px; background:#334155; border-radius:4px; overflow:hidden; margin:4px 0 8px 0;">
          <div style="position:absolute; left:{left:.2f}%; width:{pct:.2f}%; height:100%;
            background:repeating-linear-gradient(45deg,#f59e0b,#f59e0b 8px,#b45309 8px,#b45309 16px);"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def cancellation_notice(state: str) -> None:
    if state == "CANCELLING":
        st.warning("Stopping after current safe checkpoint...")
    else:
        st.warning(
            "CANCELLED - ingestion stopped at a safe checkpoint. Completed segments are retained "
            "as unapproved diagnostics."
        )
