"""Index a user-supplied MP4 for the analyst demo.

The bundled sedan clip stays the reconstruction fixture. This module only describes and
retrieves segments from one known screen recording, using notes taken from its frames.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from probity.domain.enums import (
    ExplanationSource,
    IndexState,
    InferenceMode,
)
from probity.domain.models import (
    ScoreComponents,
    SearchCandidate,
    SearchResult,
    VideoSegment,
)
from probity.search.embedding import EMBED_DIM, embedding_input_text, fixture_text_embedding

CUSTOM_CLIP_PATH = Path.home() / "Desktop" / "flake-demo-linkedin.mp4"
CUSTOM_NAME = "flake-demo-linkedin.mp4"

# 30 fps, 59.9 s, 1797 frames. Each row is one 8 s window except the tail.
_WINDOWS: tuple[dict[str, object], ...] = (
    {
        "start_us": 0,
        "end_us": 8_000_000,
        "description": (
            "Flake visual demo opens on Taco Council. The planning loop shows Memory, Agent, "
            "Gate, and Tools. The activity log says the search index was not callable, then the "
            "baseline is restored and the beat finishes as Reset baseline. Policy v1 is active."
        ),
        "terms": ("taco council", "planning loop", "baseline reset", "policy v1"),
    },
    {
        "start_us": 8_000_000,
        "end_us": 16_000_000,
        "description": (
            "Proposal for Taco Tuesday on 2026-09-29 with a $25 budget. RSVPs from Sam, Priya, "
            "Jordan, and Maya. The gate asks Alex to approve a non-refundable booking. Alex says "
            "yes, go ahead with book. The booking receipt lists a $125 deposit and a $0 premium."
        ),
        "terms": ("taco tuesday", "rsvp", "booking receipt", "alex approval"),
    },
    {
        "start_us": 16_000_000,
        "end_us": 24_000_000,
        "description": (
            "Seven days later Sam says he cannot make it. Priya paid 13 days late and Jordan paid "
            "8 days late. The chat shows request_money of $25 sent to Sam, Priya, Jordan, and Maya. "
            "The beat finishes as Advance 7 days."
        ),
        "terms": ("sam bailed", "late payment", "request money", "advance 7 days"),
    },
    {
        "start_us": 24_000_000,
        "end_us": 32_000_000,
        "description": (
            "Run Retro with policy v2 on trial. The pricing model sets a $150 exposure cap and a "
            "$60 auto non-refundable booking rule. A backtest over 6 plans shows incumbent $360 "
            "and a +$225 delta. The retro result is taco-council v2."
        ),
        "terms": ("run retro", "policy v2", "pricing model", "backtest"),
    },
    {
        "start_us": 32_000_000,
        "end_us": 40_000_000,
        "description": (
            "Policies are compared side by side. v1 and v2 rules, the auto backtest, and the "
            "canary column are on screen. The next beat queued is the beach weekend."
        ),
        "terms": ("compare policies", "policy v1", "policy v2", "canary"),
    },
    {
        "start_us": 40_000_000,
        "end_us": 48_000_000,
        "description": (
            "Beach weekend stage under policy v2. The gate asks Alex to approve booking Jordan "
            "with a $120 upfront deposit. Alex says go ahead with book. The receipt lists Maya, "
            "Sam, and Priya, a $360 deposit, and a $360 premium."
        ),
        "terms": ("beach weekend", "jordan deposit", "booking receipt", "premium"),
    },
    {
        "start_us": 48_000_000,
        "end_us": 56_000_000,
        "description": (
            "Reckless Retro is running and policy v2 is promoted. The risk table shows Sam bails "
            "at 72% upper and 64% lower. Priya paid 13 days late and Jordan paid 8 days late. "
            "Sam says he is sorry and has to bail tonight."
        ),
        "terms": ("reckless retro", "risk table", "sam bail", "policy promoted"),
    },
    {
        "start_us": 56_000_000,
        "end_us": 59_900_000,
        "description": (
            "The reckless retro beat continues to the end of the recording. The council log still "
            "shows promoted policy v2 and the resolved bail backtest."
        ),
        "terms": ("reckless retro", "policy v2", "council log", "end of clip"),
    },
)

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    {
        "find", "the", "a", "an", "when", "its", "and", "of", "to", "for", "in", "on", "with",
        "show", "over", "after", "before", "from", "this", "that", "into", "about",
    }
)


def custom_clip_available() -> bool:
    return CUSTOM_CLIP_PATH.is_file()


def _frame(pts_us: int) -> int:
    return pts_us * 30 // 1_000_000


def build_custom_segments(video_id: str, source_sha256: str) -> list[VideoSegment]:
    """Searchable segments for the Flake screen recording."""
    segments: list[VideoSegment] = []
    for ordinal, window in enumerate(_WINDOWS):
        start_us = int(window["start_us"])
        end_us = int(window["end_us"])
        start_frame = _frame(start_us)
        end_frame = max(start_frame, _frame(end_us) - 1)
        description = str(window["description"])
        raw_terms = window["terms"]
        terms = tuple(str(term) for term in raw_terms) if isinstance(raw_terms, tuple) else ()
        text = embedding_input_text(description, terms)
        segments.append(
            VideoSegment.create(
                segment_id=f"{video_id}:s{ordinal:04d}",
                video_id=video_id,
                source_sha256=source_sha256,
                ordinal=ordinal,
                start_pts_us=start_us,
                end_pts_us=end_us,
                start_frame=start_frame,
                end_frame=end_frame,
                thumbnail_frame=start_frame,
                description=description,
                description_model_id="local/frame-notes-v1",
                search_terms=terms,
                embedding_ref=f"custom-flake/segment-{ordinal:04d}",
                embedding_model_id="fixture/cosmos-embed-v1",
                embedding_dimension=EMBED_DIM,
                index_state=IndexState.INDEXED,
                mode=InferenceMode.FIXTURE,
            )
        )
        # Touch the embedding so a bad description fails here, not at search time.
        fixture_text_embedding(text)
    return segments


def _content_tokens(segment: VideoSegment) -> set[str]:
    text = f"{segment.description or ''} {' '.join(segment.search_terms)}"
    return {tok for tok in _TOKEN.findall(text.lower()) if tok not in _STOP and len(tok) > 2}


def rank_custom_segments(
    query: str, segments: list[VideoSegment], source_sha256: str
) -> tuple[tuple[SearchCandidate, ...], tuple[SearchResult, ...]]:
    """Rank indexed custom segments. A query with no shared words returns no results."""
    query_tokens = {tok for tok in _TOKEN.findall(query.lower()) if tok not in _STOP and len(tok) > 2}
    if not query_tokens or not segments:
        return (), ()

    query_vec = fixture_text_embedding(query)
    scored: list[tuple[float, float, float, VideoSegment]] = []
    for segment in segments:
        text = embedding_input_text(segment.description or "", segment.search_terms)
        cosine = float(np.dot(query_vec, fixture_text_embedding(text)))
        overlap_tokens = query_tokens & _content_tokens(segment)
        if not overlap_tokens:
            continue
        overlap = len(overlap_tokens) / len(query_tokens)
        cosine_01 = max(0.0, min(1.0, cosine))
        score = max(0.0, min(1.0, 0.45 * cosine_01 + 0.55 * overlap))
        scored.append((score, cosine, overlap, segment))

    scored.sort(key=lambda row: row[0], reverse=True)
    if not scored:
        return (), ()

    candidates: list[SearchCandidate] = []
    results: list[SearchResult] = []
    for index, (score, cosine, overlap, segment) in enumerate(scored, start=1):
        components = ScoreComponents(
            cosine=max(0.0, min(1.0, cosine)),
            lexical_overlap=overlap,
            detection_match=0.0,
            visibility_match=0.0,
            unavailable=("detection_match", "visibility_match"),
        )
        candidates.append(
            SearchCandidate(
                segment_id=segment.segment_id,
                vector_rank=index,
                similarity=max(-1.0, min(1.0, cosine)),
                components=components,
                score=score,
                final_rank=index if index <= 5 else None,
            )
        )
        if index > 5:
            continue
        explanation = (
            f"Segment {segment.ordinal} covers {segment.start_pts_us / 1e6:.1f}s to "
            f"{segment.end_pts_us / 1e6:.1f}s of {CUSTOM_NAME}. {segment.description}"
        )[:600]
        results.append(
            SearchResult(
                rank=index,
                segment_id=segment.segment_id,
                video_id=segment.video_id,
                source_sha256=source_sha256,
                score=score,
                components=components,
                start_pts_us=segment.start_pts_us,
                end_pts_us=segment.end_pts_us,
                start_frame=segment.start_frame,
                end_frame=segment.end_frame,
                thumbnail_uri=None,
                description=(segment.description or "")[:1000],
                explanation=explanation,
                explanation_source=ExplanationSource.TEMPLATE,
                explanation_input_hash=hashlib.sha256(
                    f"{query}:{segment.segment_id}".encode()
                ).hexdigest(),
            )
        )
    return tuple(candidates), tuple(results)
