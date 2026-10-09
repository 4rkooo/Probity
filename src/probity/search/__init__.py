"""Search planning, retrieval composition, and grounded explanations."""

from probity.search.embedding import fixture_text_embedding
from probity.search.explain import template_explanation, validate_explanation
from probity.search.plan import (
    ALLOWED_SUBJECT_CLASSES,
    RULE_BASED_PLANNER_MODEL_ID,
    RuleBasedPlanner,
    sanitize_search_plan,
)
from probity.search.rerank import clamp01, collapse_overlaps, rerank_candidates, time_iou
from probity.search.service import LocalSearchService, SystemClock
from probity.search.sqlite_store import SqlEvidenceStore

__all__ = [
    "ALLOWED_SUBJECT_CLASSES",
    "RULE_BASED_PLANNER_MODEL_ID",
    "LocalSearchService",
    "RuleBasedPlanner",
    "SqlEvidenceStore",
    "SystemClock",
    "clamp01",
    "collapse_overlaps",
    "fixture_text_embedding",
    "rerank_candidates",
    "sanitize_search_plan",
    "template_explanation",
    "time_iou",
    "validate_explanation",
]
