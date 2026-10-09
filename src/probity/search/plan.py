"""Query-plan construction and policy sanitization (section 8)."""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import ValidationError

from probity.domain.enums import ReasonCode
from probity.domain.models import SearchPlan, TimeRangeUs
from probity.domain.policy import PolicyConfig, SearchPolicy
from probity.ports import EvidenceReasoner, QueryPlanRequest

ALLOWED_SUBJECT_CLASSES: tuple[str, ...] = (
    "car",
    "truck",
    "license_plate",
    "sign",
    "bicycle",
    "bus",
    "motorcycle",
)

CLASS_ALIASES: dict[str, str] = {
    "car": "car",
    "cars": "car",
    "sedan": "car",
    "sedans": "car",
    "vehicle": "car",
    "vehicles": "car",
    "auto": "car",
    "automobile": "car",
    "truck": "truck",
    "trucks": "truck",
    "van": "truck",
    "vans": "truck",
    "lorry": "truck",
    "license_plate": "license_plate",
    "license plate": "license_plate",
    "licence plate": "license_plate",
    "number plate": "license_plate",
    "rear plate": "license_plate",
    "front plate": "license_plate",
    "plate": "license_plate",
    "plates": "license_plate",
    "sign": "sign",
    "signs": "sign",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "bus": "bus",
    "motorcycle": "motorcycle",
}

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "every",
        "find",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "show",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "when",
        "where",
        "which",
        "with",
    }
)

# Remaining verbs/pronouns that are not searchable objectives on their own.
NON_OBJECTIVE: frozenset[str] = frozenset(
    {
        "flee",
        "fled",
        "flees",
        "fleeing",
        "they",
        "them",
        "someone",
        "anyone",
        "please",
        "tell",
        "me",
        "about",
        "happened",
    }
)

PHRASES: tuple[str, ...] = (
    "license plate",
    "licence plate",
    "number plate",
    "rear plate",
    "front plate",
    "most visible",
    "partly visible",
    "clearly visible",
    "blue sedan",
    "red van",
    "green sign",
)

_WORD_RE = re.compile(r"[a-z0-9]+")
RULE_BASED_PLANNER_MODEL_ID = "probity/rule-based-planner-v1"


def _all_disallowed(policy: SearchPolicy) -> tuple[str, ...]:
    terms = (
        *policy.disallowed_terms.identity,
        *policy.disallowed_terms.legal,
        *policy.disallowed_terms.intent,
    )
    return tuple(sorted(terms, key=lambda t: (-len(t), t)))


def _word_pattern(term: str) -> re.Pattern[str]:
    if " " in term:
        parts = [re.escape(p) for p in term.split()]
        return re.compile(r"\b" + r"\s+".join(parts) + r"\b", re.IGNORECASE)
    return re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)


def find_disallowed_terms(query: str, policy: SearchPolicy) -> tuple[str, ...]:
    """Return disallowed terms in first-appearance order, including multiword phrases."""
    found: list[str] = []
    seen: set[str] = set()
    matches: list[tuple[int, str]] = []
    for term in _all_disallowed(policy):
        for match in _word_pattern(term).finditer(query):
            key = term.lower()
            if key not in seen:
                seen.add(key)
                matches.append((match.start(), key))
    matches.sort(key=lambda item: item[0])
    for _, term in matches:
        found.append(term)
    return tuple(found)


def _strip_terms(query: str, terms: Iterable[str]) -> str:
    remaining = query
    for term in sorted(terms, key=len, reverse=True):
        remaining = _word_pattern(term).sub(" ", remaining)
    return re.sub(r"\s+", " ", remaining).strip()


def _extract_phrases(text: str) -> tuple[list[str], str]:
    lowered = text.lower()
    found: list[tuple[int, str]] = []
    consumed = lowered
    for phrase in sorted(PHRASES, key=len, reverse=True):
        pattern = _word_pattern(phrase)
        match = pattern.search(consumed)
        if match:
            found.append((match.start(), phrase))
            consumed = pattern.sub(" ", consumed, count=1)
    found.sort(key=lambda item: item[0])
    leftover = re.sub(r"\s+", " ", consumed).strip()
    return [phrase for _, phrase in found], leftover


def _objective_terms_and_classes(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    phrases, leftover = _extract_phrases(text)
    tokens = _WORD_RE.findall(leftover.lower())
    classes: list[str] = []
    seen_classes: set[str] = set()
    terms: list[str] = []
    seen_terms: set[str] = set()

    def add_class(name: str) -> None:
        if name not in seen_classes and name in ALLOWED_SUBJECT_CLASSES:
            seen_classes.add(name)
            classes.append(name)

    def add_term(term: str) -> None:
        cleaned = term.strip().lower()
        if not cleaned or cleaned in seen_terms or len(cleaned) > 40:
            return
        seen_terms.add(cleaned)
        terms.append(cleaned)

    for phrase in phrases:
        add_term(phrase)
        mapped = CLASS_ALIASES.get(phrase)
        if mapped:
            add_class(mapped)
        else:
            for token in phrase.split():
                aliased = CLASS_ALIASES.get(token)
                if aliased:
                    add_class(aliased)

    # Color + vehicle bigrams remaining in leftover (e.g. "white truck").
    colors = {"blue", "red", "green", "white", "black", "gray", "grey", "yellow"}
    i = 0
    leftover_tokens = [t for t in tokens if t not in STOPWORDS]
    while i < len(leftover_tokens):
        tok = leftover_tokens[i]
        nxt = leftover_tokens[i + 1] if i + 1 < len(leftover_tokens) else None
        if tok in colors and nxt in CLASS_ALIASES:
            add_term(f"{tok} {nxt}")
            add_class(CLASS_ALIASES[nxt])
            i += 2
            continue
        if tok in CLASS_ALIASES:
            add_class(CLASS_ALIASES[tok])
            add_term(tok)
            i += 1
            continue
        if tok not in NON_OBJECTIVE:
            add_term(tok)
        i += 1

    return tuple(terms), tuple(classes)


def _semantic_query(terms: tuple[str, ...], classes: tuple[str, ...]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for term in terms:
        expanded = term.replace("plate", "license plate") if term == "plate" else term
        if term in {"rear plate", "front plate"}:
            expanded = f"{term.split()[0]} license plate"
        for token in expanded.split():
            if token not in seen:
                seen.add(token)
                parts.append(token)
    for cls in classes:
        for token in cls.split("_"):
            if token not in seen:
                seen.add(token)
                parts.append(token)
    return " ".join(parts)[:300]


def rule_based_plan(
    query: str, policy: SearchPolicy, time_range: TimeRangeUs | None = None
) -> SearchPlan:
    """Stopword-aware planner that extracts objective terms and allowed classes."""
    filtered = find_disallowed_terms(query, policy)
    remaining = _strip_terms(query, filtered)
    terms, classes = _objective_terms_and_classes(remaining)
    objective = tuple(t for t in terms if t not in STOPWORDS and t not in NON_OBJECTIVE)
    if not objective and not classes:
        return SearchPlan(
            semantic_query="",
            objective_terms=(),
            subject_classes=(),
            time_range=time_range,
            needs_clarification=True,
            filtered_terms=filtered,
            policy_reason_codes=(ReasonCode.QUERY_POLICY_FILTERED,) if filtered else (),
        )
    semantic = _semantic_query(objective, classes) or remaining.lower()[:300]
    return SearchPlan(
        semantic_query=semantic,
        objective_terms=objective,
        subject_classes=classes,
        time_range=time_range,
        needs_clarification=False,
        filtered_terms=filtered,
        policy_reason_codes=(ReasonCode.QUERY_POLICY_FILTERED,) if filtered else (),
    )


def sanitize_search_plan(plan: SearchPlan, policy: SearchPolicy, original_query: str) -> SearchPlan:
    """Drop identity/legal/intent terms; require an objective remainder."""
    extra = find_disallowed_terms(
        " ".join((original_query, plan.semantic_query, *plan.objective_terms)), policy
    )
    filtered = tuple(dict.fromkeys((*plan.filtered_terms, *extra)))
    cleaned_semantic = _strip_terms(plan.semantic_query, extra)
    cleaned_terms = tuple(
        term
        for term in plan.objective_terms
        if not find_disallowed_terms(term, policy)
        and term.lower() not in STOPWORDS
        and term.lower() not in NON_OBJECTIVE
    )
    classes = tuple(cls for cls in plan.subject_classes if cls in ALLOWED_SUBJECT_CLASSES)
    reasons = tuple(plan.policy_reason_codes)
    if filtered and ReasonCode.QUERY_POLICY_FILTERED not in reasons:
        reasons = (*reasons, ReasonCode.QUERY_POLICY_FILTERED)
    leftover_tokens = [
        tok
        for tok in _WORD_RE.findall(cleaned_semantic.lower())
        if tok not in STOPWORDS and tok not in NON_OBJECTIVE
    ]
    needs = plan.needs_clarification
    if not leftover_tokens and not cleaned_terms and not classes:
        needs = True
        cleaned_semantic = ""
        cleaned_terms = ()
        classes = ()
    return SearchPlan(
        semantic_query=cleaned_semantic[:300],
        objective_terms=cleaned_terms,
        subject_classes=classes,
        time_range=plan.time_range,
        needs_clarification=needs,
        filtered_terms=filtered,
        policy_reason_codes=reasons,
    )


def _plan_is_invalid(plan: SearchPlan) -> bool:
    text = f"{plan.semantic_query} {' '.join(plan.objective_terms)}"
    if re.search(r":f\d+\b", text):
        return True
    if re.search(r"\bframe\s+\d+\b", text, re.IGNORECASE):
        return True
    try:
        SearchPlan.model_validate(plan.model_dump(mode="python"))
    except ValidationError:
        return True
    return False


class RuleBasedPlanner:
    """Deterministic planner used when no reasoner is configured or its plan is invalid."""

    model_id = RULE_BASED_PLANNER_MODEL_ID

    def __init__(self, policy: PolicyConfig) -> None:
        self._policy = policy

    def plan(self, request: QueryPlanRequest) -> SearchPlan:
        allowed = set(request.allowed_subject_classes) or set(ALLOWED_SUBJECT_CLASSES)
        raw = rule_based_plan(request.query, self._policy.search, time_range=None)
        classes = tuple(cls for cls in raw.subject_classes if cls in allowed)
        drafted = SearchPlan(
            semantic_query=raw.semantic_query,
            objective_terms=raw.objective_terms,
            subject_classes=classes,
            time_range=raw.time_range,
            needs_clarification=raw.needs_clarification
            or (not raw.objective_terms and not classes),
            filtered_terms=raw.filtered_terms,
            policy_reason_codes=raw.policy_reason_codes,
        )
        return sanitize_search_plan(drafted, self._policy.search, request.query)


async def plan_search_query(
    request: QueryPlanRequest,
    policy: PolicyConfig,
    reasoner: EvidenceReasoner | None = None,
) -> tuple[SearchPlan, str]:
    """Return (plan, planner_model_id). Invalid reasoner output falls back to rules."""
    rules = RuleBasedPlanner(policy)
    fallback = rules.plan(request)
    if reasoner is None:
        return fallback, RULE_BASED_PLANNER_MODEL_ID
    try:
        raw = await reasoner.plan_query(request)
        sanitized = sanitize_search_plan(raw, policy.search, request.query)
        if _plan_is_invalid(sanitized):
            return fallback, RULE_BASED_PLANNER_MODEL_ID
        return sanitized, reasoner.model_id or RULE_BASED_PLANNER_MODEL_ID
    except Exception:
        return fallback, RULE_BASED_PLANNER_MODEL_ID
