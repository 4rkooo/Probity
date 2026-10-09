"""Deterministic grounded explanations; uncited reasoner prose is replaced."""

from __future__ import annotations

import re
from collections.abc import Sequence

from probity.domain.enums import ExplanationSource
from probity.domain.ids import SEGMENT_ID_RE, UUID7_RE, canonical_sha256
from probity.domain.models import GroundedExplanation, VideoSegment
from probity.ports import RetrievedSegment

_KNOWN_CLASSES = frozenset(
    {
        "car",
        "truck",
        "license_plate",
        "sign",
        "bicycle",
        "bus",
        "motorcycle",
        "sedan",
        "van",
        "person",
    }
)
_CLOCK_RE = re.compile(r"\b\d{1,2}:\d{2}(?:\.\d+)?\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def explanation_input_hash(evidence: RetrievedSegment, query: str, plan_semantic: str) -> str:
    payload = {
        "evidence": evidence.model_dump(mode="json"),
        "query": query,
        "semantic_query": plan_semantic,
    }
    return canonical_sha256(payload)


def template_explanation(segment: VideoSegment) -> str:
    start_s = segment.start_pts_us / 1_000_000
    end_s = segment.end_pts_us / 1_000_000
    description = segment.description or ""
    text = f"Segment {start_s:.1f}-{end_s:.1f} s: {description}"
    if segment.evidence_detection_ids:
        names = [item.class_name for item in segment.detected_classes]
        if "license_plate" in names:
            class_name = "license_plate"
        else:
            class_name = names[0] if names else "detection"
        text += f" {len(segment.evidence_detection_ids)} {class_name} detection(s) cited."
    return text[:600]


def _allowed_numbers(segment: VideoSegment) -> set[str]:
    allowed: set[str] = set()
    for value in (
        segment.start_pts_us,
        segment.end_pts_us,
        segment.start_frame,
        segment.end_frame,
        segment.ordinal,
    ):
        allowed.add(str(value))
    allowed.add(f"{segment.start_pts_us / 1_000_000:.1f}")
    allowed.add(f"{segment.end_pts_us / 1_000_000:.1f}")
    allowed.add(str(len(segment.evidence_detection_ids)))
    blob = _blob(segment)
    allowed.update(_NUMBER_RE.findall(blob))
    allowed.update(_CLOCK_RE.findall(blob))
    for item in segment.detected_classes:
        allowed.add(str(item.detection_count))
        allowed.update(_NUMBER_RE.findall(f"{item.max_confidence}"))
        allowed.add(f"{item.max_confidence:.2f}")
        allowed.add(f"{item.max_confidence:.1f}")
    return allowed


def _blob(segment: VideoSegment) -> str:
    return f"{segment.description or ''} {' '.join(segment.search_terms)}"


def _allowed_ids(segment: VideoSegment) -> set[str]:
    return {segment.segment_id, segment.video_id, *segment.evidence_detection_ids}


def _allowed_classes(segment: VideoSegment) -> set[str]:
    names = {item.class_name.lower() for item in segment.detected_classes}
    blob = _blob(segment).lower()
    names.update(cls for cls in _KNOWN_CLASSES if re.search(rf"\b{re.escape(cls)}\b", blob))
    names.update(tag.value.lower() for tag in segment.visibility_tags)
    return names


def validate_explanation(text: str, segment: VideoSegment) -> bool:
    """Reject prose that introduces uncited timestamps, numbers, classes, or IDs."""
    allowed_ids = _allowed_ids(segment)
    for match in UUID7_RE.finditer(text):
        token = match.group(0)
        # Segment IDs contain a uuid prefix; accept if the full segment_id is cited.
        if token not in allowed_ids and not any(token in item for item in allowed_ids):
            return False
    for match in SEGMENT_ID_RE.finditer(text):
        if match.group(0) not in allowed_ids:
            return False
    allowed_nums = _allowed_numbers(segment)
    for match in (*_CLOCK_RE.findall(text), *_NUMBER_RE.findall(text)):
        if match not in allowed_nums:
            return False
    allowed_classes = _allowed_classes(segment)
    lowered = text.lower()
    for cls in _KNOWN_CLASSES:
        if re.search(rf"\b{re.escape(cls)}\b", lowered) and cls not in allowed_classes:
            return False
    return True


def explain_retrieved(
    query: str,
    plan_semantic: str,
    item: RetrievedSegment,
    reasoner_text: str | None = None,
    reasoner_model_id: str | None = None,
) -> GroundedExplanation:
    segment = item.segment
    input_hash = explanation_input_hash(item, query, plan_semantic)
    if reasoner_text is not None and validate_explanation(reasoner_text, segment):
        return GroundedExplanation(
            segment_id=segment.segment_id,
            text=reasoner_text[:600],
            cited_detection_ids=segment.evidence_detection_ids,
            cited_pts_us=(segment.start_pts_us, segment.end_pts_us),
            source=ExplanationSource.REASONER,
            model_id=reasoner_model_id,
            input_sha256=input_hash,
        )
    return GroundedExplanation(
        segment_id=segment.segment_id,
        text=template_explanation(segment),
        cited_detection_ids=segment.evidence_detection_ids,
        cited_pts_us=(segment.start_pts_us, segment.end_pts_us),
        source=ExplanationSource.TEMPLATE,
        model_id=None,
        input_sha256=input_hash,
    )


async def explain_results(
    query: str,
    plan_semantic: str,
    items: Sequence[RetrievedSegment],
    reasoner_explanations: Sequence[GroundedExplanation] | None = None,
) -> list[GroundedExplanation]:
    by_id = {item.segment_id: item for item in (reasoner_explanations or ())}
    out: list[GroundedExplanation] = []
    for retrieved in items:
        offered = by_id.get(retrieved.segment.segment_id)
        text = offered.text if offered is not None else None
        model_id = offered.model_id if offered is not None else None
        out.append(explain_retrieved(query, plan_semantic, retrieved, text, model_id))
    return out
