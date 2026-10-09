"""Deterministic search rerank and time-IoU collapse (section 8)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from probity.domain.enums import VisibilityKind
from probity.domain.models import (
    ScoreComponents,
    SearchCandidate,
    SearchPlan,
    VideoSegment,
)
from probity.domain.policy import SearchPolicy, SearchWeights
from probity.ports import RetrievedSegment

UnavailableKey = Literal["lexical_overlap", "detection_match", "visibility_match"]
_NEGATIVE_VISIBILITY = frozenset(
    {
        VisibilityKind.OCCLUSION,
        VisibilityKind.DARKNESS,
        VisibilityKind.MOTION_BLUR,
        VisibilityKind.FOCUS_BLUR,
        VisibilityKind.GLARE,
    }
)


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def time_iou(left: VideoSegment, right: VideoSegment) -> float:
    start = max(left.start_pts_us, right.start_pts_us)
    end = min(left.end_pts_us, right.end_pts_us)
    inter = max(0, end - start)
    union = (left.end_pts_us - left.start_pts_us) + (right.end_pts_us - right.start_pts_us) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _blob(segment: VideoSegment) -> str:
    return f"{segment.description or ''} {' '.join(segment.search_terms)}".lower()


def lexical_overlap(plan: SearchPlan, segment: VideoSegment) -> tuple[float, bool]:
    terms = plan.objective_terms
    if not terms:
        return 0.0, True
    blob = _blob(segment)
    hits = sum(1 for term in terms if term.lower() in blob)
    return clamp01(hits / len(terms)), False


def detection_match(plan: SearchPlan, segment: VideoSegment) -> tuple[float, bool]:
    if not plan.subject_classes:
        return 0.0, True
    if not segment.detected_classes:
        return 0.0, True
    detected = {item.class_name: item.max_confidence for item in segment.detected_classes}
    matched = [detected[name] for name in plan.subject_classes if name in detected]
    overlap_frac = len(matched) / len(plan.subject_classes)
    mean_conf = sum(matched) / len(matched) if matched else 0.0
    return clamp01(overlap_frac * mean_conf), False


def _subject_hit(plan: SearchPlan, segment: VideoSegment) -> bool:
    detected = {item.class_name for item in segment.detected_classes}
    if plan.subject_classes and detected.intersection(plan.subject_classes):
        return True
    blob = _blob(segment)
    return any(term.lower() in blob for term in plan.objective_terms)


def visibility_match(plan: SearchPlan, segment: VideoSegment) -> tuple[float, bool]:
    """Precise rule: negative visibility tags score 0; otherwise 1 iff the subject is hit.

    OCCLUSION/DARKNESS/MOTION_BLUR/FOCUS_BLUR/GLARE are treated as not-visible. Missing
    tags are unavailable (score 0, listed). A subject hit is class overlap or any
    objective term appearing in the description/search_terms.
    """
    if not segment.visibility_tags:
        return 0.0, True
    tags = set(segment.visibility_tags)
    if tags & _NEGATIVE_VISIBILITY:
        return 0.0, False
    if _subject_hit(plan, segment):
        return 1.0, False
    return 0.0, False


def combine_score(components: ScoreComponents, weights: SearchWeights) -> float:
    return (
        weights.cosine * components.cosine
        + weights.lexical_overlap * components.lexical_overlap
        + weights.detection_match * components.detection_match
        + weights.visibility_match * components.visibility_match
    )


def score_retrieved(
    item: RetrievedSegment, plan: SearchPlan, weights: SearchWeights
) -> SearchCandidate:
    cosine = clamp01(item.similarity)
    lexical, lex_missing = lexical_overlap(plan, item.segment)
    detection, det_missing = detection_match(plan, item.segment)
    vis, vis_missing = visibility_match(plan, item.segment)
    unavailable: list[UnavailableKey] = []
    if lex_missing:
        unavailable.append("lexical_overlap")
    if det_missing:
        unavailable.append("detection_match")
    if vis_missing:
        unavailable.append("visibility_match")
    components = ScoreComponents(
        cosine=cosine,
        lexical_overlap=lexical,
        detection_match=detection,
        visibility_match=vis,
        unavailable=tuple(unavailable),
    )
    return SearchCandidate(
        segment_id=item.segment.segment_id,
        vector_rank=item.vector_rank,
        similarity=item.similarity,
        components=components,
        score=clamp01(combine_score(components, weights)),
        final_rank=None,
        collapsed_into=None,
    )


def rerank_candidates(
    retrieved: Sequence[RetrievedSegment],
    plan: SearchPlan,
    policy: SearchPolicy,
) -> list[tuple[SearchCandidate, RetrievedSegment]]:
    """Score, then order by score desc, similarity desc, ordinal asc."""
    paired = [(score_retrieved(item, plan, policy.weights), item) for item in retrieved]
    paired.sort(key=lambda pair: (-pair[0].score, -pair[0].similarity, pair[1].segment.ordinal))
    return paired


def collapse_overlaps(
    ranked: Sequence[tuple[SearchCandidate, RetrievedSegment]],
    iou_threshold: float,
) -> list[tuple[SearchCandidate, RetrievedSegment]]:
    """Collapse later segments whose time IoU with a kept segment is strictly > threshold.

    IoU equal to the threshold is retained. Collapsed candidates keep their scores and
    point at the higher-ranked survivor via ``collapsed_into``.
    """
    kept: list[tuple[SearchCandidate, RetrievedSegment]] = []
    collapsed: list[tuple[SearchCandidate, RetrievedSegment]] = []
    for candidate, item in ranked:
        winner: RetrievedSegment | None = None
        for _kept_candidate, kept_item in kept:
            if time_iou(item.segment, kept_item.segment) > iou_threshold:
                winner = kept_item
                break
        if winner is None:
            kept.append((candidate, item))
        else:
            collapsed.append(
                (
                    SearchCandidate(
                        segment_id=candidate.segment_id,
                        vector_rank=candidate.vector_rank,
                        similarity=candidate.similarity,
                        components=candidate.components,
                        score=candidate.score,
                        final_rank=None,
                        collapsed_into=winner.segment.segment_id,
                    ),
                    item,
                )
            )
    numbered: list[tuple[SearchCandidate, RetrievedSegment]] = []
    for rank, (candidate, item) in enumerate(kept, start=1):
        numbered.append(
            (
                SearchCandidate(
                    segment_id=candidate.segment_id,
                    vector_rank=candidate.vector_rank,
                    similarity=candidate.similarity,
                    components=candidate.components,
                    score=candidate.score,
                    final_rank=rank,
                    collapsed_into=None,
                ),
                item,
            )
        )
    return numbered + collapsed
