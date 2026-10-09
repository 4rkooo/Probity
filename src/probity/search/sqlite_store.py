"""EvidenceStore that searches segments already written to SqlRepository.

Retrieval is FixtureEvidenceStore's normalized dot product. Rerank stays in
LocalSearchService. The API and the worker are separate processes, so search
reloads the repository at request time instead of trusting in-memory rows.
"""

from __future__ import annotations

from collections.abc import Sequence

from probity.adapters.fixture.vast import FixtureEvidenceStore
from probity.domain.enums import AdapterMode, IndexState
from probity.domain.models import (
    AdapterHealth,
    Detection,
    Embedding,
    EvidenceReport,
    LineageRecord,
    RetrievedSegment,
    SourceVideo,
    VideoSegment,
)
from probity.ports import Repository, SearchRequest

_SEARCHABLE = frozenset({IndexState.EMBEDDED, IndexState.INDEXED})


class SqlEvidenceStore:
    """Fixture index hydrated from the repository that Agent B already persists."""

    schema_version = "1.0"

    def __init__(self, repository: Repository, inner: FixtureEvidenceStore | None = None) -> None:
        self._repository = repository
        self._inner = inner if inner is not None else FixtureEvidenceStore()

    @property
    def inner(self) -> FixtureEvidenceStore:
        return self._inner

    @property
    def adapter_name(self) -> str:
        return self._inner.adapter_name

    @property
    def model_id(self) -> str | None:
        return self._inner.model_id

    @property
    def mode(self) -> AdapterMode:
        return self._inner.mode

    async def health(self) -> AdapterHealth:
        return await self._inner.health()

    async def hydrate(self, video_id: str) -> None:
        """Load the latest searchable segments and align embeddings by embedding_ref."""
        segments = [
            segment
            for segment in self._repository.list_segments(video_id)
            if segment.index_state in _SEARCHABLE and segment.embedding_ref
        ]
        if not segments:
            return
        by_ref = {
            embedding.embedding_ref: embedding
            for embedding in self._repository.list_embeddings(video_id)
        }
        paired_segments: list[VideoSegment] = []
        paired_embeddings: list[Embedding] = []
        for segment in segments:
            embedding = by_ref.get(segment.embedding_ref or "")
            if embedding is None:
                continue
            paired_segments.append(segment)
            paired_embeddings.append(embedding)
        if paired_segments:
            await self._inner.upsert_segments(paired_segments, paired_embeddings)

    async def put_source(self, video: SourceVideo, local_path: str) -> str:
        return await self._inner.put_source(video, local_path)

    async def upsert_segments(
        self, segments: Sequence[VideoSegment], embeddings: Sequence[Embedding]
    ) -> None:
        await self._inner.upsert_segments(segments, embeddings)

    async def upsert_detections(self, detections: Sequence[Detection]) -> None:
        await self._inner.upsert_detections(detections)

    async def search(self, request: SearchRequest) -> Sequence[RetrievedSegment]:
        await self.hydrate(request.video_id)
        return await self._inner.search(request)

    async def put_lineage(self, record: LineageRecord) -> None:
        await self._inner.put_lineage(record)

    async def put_report(self, report: EvidenceReport) -> None:
        await self._inner.put_report(report)
