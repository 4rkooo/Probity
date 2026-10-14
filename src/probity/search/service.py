"""Local search orchestration: plan, retrieve, rerank, collapse, explain."""

from __future__ import annotations

import time
from typing import Protocol

from probity.domain.enums import AdapterMode, ExplanationSource, InferenceMode, SearchStatus
from probity.domain.ids import new_uuid7, utc_now
from probity.domain.models import (
    SearchCandidate,
    SearchEvidence,
    SearchPlan,
    SearchResult,
    SourceVideo,
    TimeRangeUs,
)
from probity.domain.policy import PolicyConfig
from probity.ports import (
    EvidenceReasoner,
    EvidenceStore,
    QueryPlanRequest,
    SearchQuery,
    SearchRequest,
    VideoUnderstanding,
)
from probity.search.explain import explain_retrieved
from probity.search.plan import ALLOWED_SUBJECT_CLASSES, plan_search_query
from probity.search.rerank import collapse_overlaps, rerank_candidates


class Clock(Protocol):
    def now(self) -> str: ...
    def monotonic_ns(self) -> int: ...


class SystemClock:
    def now(self) -> str:
        return utc_now()

    def monotonic_ns(self) -> int:
        return time.perf_counter_ns()


def _inference_mode(understanding: VideoUnderstanding, store: EvidenceStore) -> InferenceMode:
    """Mode that produced the evidence. Unresolved DEGRADED is not disclosed as live."""
    modes = {understanding.mode, store.mode}
    if AdapterMode.FIXTURE in modes or AdapterMode.DEGRADED in modes:
        return InferenceMode.FIXTURE
    return InferenceMode.LIVE


class LocalSearchService:
    """In-process SearchService. Persistence is the API's job."""

    def __init__(
        self,
        understanding: VideoUnderstanding,
        store: EvidenceStore,
        policy: PolicyConfig,
        reasoner: EvidenceReasoner | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._understanding = understanding
        self._store = store
        self._policy = policy
        self._reasoner = reasoner
        self._clock: Clock = clock if clock is not None else SystemClock()

    async def _disclosed_mode(self) -> InferenceMode:
        for adapter in (self._understanding, self._store):
            resolve = getattr(adapter, "resolve", None)
            if resolve is not None:
                await resolve()
        return _inference_mode(self._understanding, self._store)

    async def search(
        self, video: SourceVideo, query: SearchQuery, correlation_id: str
    ) -> SearchEvidence:
        started = self._clock.monotonic_ns()
        request = QueryPlanRequest(
            query=query.query,
            video_id=video.video_id,
            allowed_subject_classes=ALLOWED_SUBJECT_CLASSES,
            indexed_range=video.indexed_range,
        )
        plan, planner_model_id = await plan_search_query(request, self._policy, self._reasoner)
        if query.time_range is not None:
            plan = SearchPlan(
                semantic_query=plan.semantic_query,
                objective_terms=plan.objective_terms,
                subject_classes=plan.subject_classes,
                time_range=query.time_range,
                needs_clarification=plan.needs_clarification,
                filtered_terms=plan.filtered_terms,
                policy_reason_codes=plan.policy_reason_codes,
            )
        # Never search outside the video's indexed coverage (bad reasoner windows).
        plan = _clamp_plan_to_indexed(plan, video.indexed_range)

        def elapsed_ms() -> int:
            return max(0, int((self._clock.monotonic_ns() - started) / 1_000_000))

        if plan.needs_clarification:
            # SearchEvidence.embedding_model_id is a required ModelId. embed_query did
            # not run, and the frozen model rejects None. No sentinel for that absence exists.
            return SearchEvidence.create(
                created_at=self._clock.now(),
                search_id=new_uuid7(),
                case_id=video.case_id,
                video_id=video.video_id,
                query=query.query,
                query_plan=plan,
                planner_model_id=planner_model_id,
                embedding_model_id="fixture/cosmos-embed-v1",
                reasoner_model_id=self._reasoner.model_id if self._reasoner is not None else None,
                status=SearchStatus.NEEDS_CLARIFICATION,
                mode=await self._disclosed_mode(),
                indexed_range=video.indexed_range,
                candidates=(),
                results=(),
                latency_ms=elapsed_ms(),
                correlation_id=correlation_id,
            )

        embedding = await self._understanding.embed_query(plan.semantic_query)
        retrieved = await self._store.search(
            SearchRequest(
                video_id=video.video_id,
                source_sha256=video.sha256,
                query_embedding=embedding,
                top_k=self._policy.search.top_k,
                time_range=query.time_range or plan.time_range,
                subject_classes=plan.subject_classes,
            )
        )
        ranked = rerank_candidates(retrieved, plan, self._policy.search)
        collapsed = collapse_overlaps(ranked, self._policy.search.overlap_collapse_time_iou)
        candidates = tuple(candidate for candidate, _item in collapsed)
        kept = [(cand, item) for cand, item in collapsed if cand.collapsed_into is None]
        max_results = min(query.max_results, self._policy.search.max_results)
        selected = kept[:max_results]

        reasoner_rows = None
        if self._reasoner is not None and selected:
            try:
                reasoner_rows = await self._reasoner.explain_results(
                    query.query, [item for _cand, item in selected]
                )
            except Exception:
                reasoner_rows = None
        reasoner_by_id = {row.segment_id: row for row in (reasoner_rows or ())}

        results: list[SearchResult] = []
        for rank, (candidate, item) in enumerate(selected, start=1):
            offered = reasoner_by_id.get(item.segment.segment_id)
            grounded = explain_retrieved(
                query.query,
                plan.semantic_query,
                item,
                offered.text if offered is not None else None,
                offered.model_id if offered is not None else None,
            )
            segment = item.segment
            results.append(
                SearchResult(
                    rank=rank,
                    segment_id=segment.segment_id,
                    video_id=segment.video_id,
                    source_sha256=segment.source_sha256,
                    score=candidate.score,
                    components=candidate.components,
                    start_pts_us=segment.start_pts_us,
                    end_pts_us=segment.end_pts_us,
                    start_frame=segment.start_frame,
                    end_frame=segment.end_frame,
                    thumbnail_uri=segment.thumbnail_uri,
                    description=segment.description or "",
                    detected_classes=segment.detected_classes,
                    evidence_detection_ids=segment.evidence_detection_ids,
                    explanation=grounded.text,
                    explanation_source=grounded.source,
                    explanation_input_hash=grounded.input_sha256,
                    outside_indexed_range_warning=_outside_range(
                        video, segment.start_pts_us, segment.end_pts_us
                    ),
                )
            )

        status = SearchStatus.OK if results else SearchStatus.NO_RESULTS
        mode = await self._disclosed_mode()
        # Keep candidate final_rank aligned with result ranks for survivors.
        remapped: list[SearchCandidate] = []
        result_rank = {row.segment_id: row.rank for row in results}
        for candidate in candidates:
            if candidate.segment_id in result_rank:
                remapped.append(
                    SearchCandidate(
                        segment_id=candidate.segment_id,
                        vector_rank=candidate.vector_rank,
                        similarity=candidate.similarity,
                        components=candidate.components,
                        score=candidate.score,
                        final_rank=result_rank[candidate.segment_id],
                        collapsed_into=None,
                    )
                )
            else:
                remapped.append(candidate)

        reasoner_model = None
        if self._reasoner is not None and any(
            row.explanation_source is ExplanationSource.REASONER for row in results
        ):
            reasoner_model = self._reasoner.model_id

        return SearchEvidence.create(
            created_at=self._clock.now(),
            search_id=new_uuid7(),
            case_id=video.case_id,
            video_id=video.video_id,
            query=query.query,
            query_plan=plan,
            planner_model_id=planner_model_id,
            embedding_model_id=embedding.model_id,
            reasoner_model_id=reasoner_model,
            status=status,
            mode=mode,
            indexed_range=video.indexed_range,
            candidates=tuple(remapped),
            results=tuple(results),
            latency_ms=elapsed_ms(),
            correlation_id=correlation_id,
        )


def _clamp_plan_to_indexed(plan: SearchPlan, indexed) -> SearchPlan:
    """Drop or intersect a planner time window that falls outside indexed coverage."""
    if plan.time_range is None or indexed is None:
        return plan
    start = max(plan.time_range.start_pts_us, indexed.start_pts_us)
    end = min(plan.time_range.end_pts_us, indexed.end_pts_us)
    if end <= start:
        return SearchPlan(
            semantic_query=plan.semantic_query,
            objective_terms=plan.objective_terms,
            subject_classes=plan.subject_classes,
            time_range=None,
            needs_clarification=plan.needs_clarification,
            filtered_terms=plan.filtered_terms,
            policy_reason_codes=plan.policy_reason_codes,
        )
    if start == plan.time_range.start_pts_us and end == plan.time_range.end_pts_us:
        return plan
    return SearchPlan(
        semantic_query=plan.semantic_query,
        objective_terms=plan.objective_terms,
        subject_classes=plan.subject_classes,
        time_range=TimeRangeUs(start_pts_us=start, end_pts_us=end),
        needs_clarification=plan.needs_clarification,
        filtered_terms=plan.filtered_terms,
        policy_reason_codes=plan.policy_reason_codes,
    )


def _outside_range(video: SourceVideo, start_pts_us: int, end_pts_us: int) -> bool:
    indexed = video.indexed_range
    if indexed is None:
        return False
    return start_pts_us < indexed.start_pts_us or end_pts_us > indexed.end_pts_us
