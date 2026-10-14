"""In-memory fixture EvidenceStore: normalized-dot-product search over upserted segments."""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from probity.domain.enums import AdapterMode, HealthStatus
from probity.domain.errors import ValidationFailed
from probity.domain.ids import utc_now
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
from probity.ports import SearchRequest

STORE_MODEL_ID = "fixture/vast-index-v1"


def _copy_frozen(value: Any) -> Any:
    """Defensive copy so callers cannot observe internal aliasing of sequences."""
    if hasattr(value, "model_copy"):
        return value.model_copy()
    return copy.deepcopy(value)


class FixtureEvidenceStore:
    """In-memory VAST stand-in. Never mutates caller-owned inputs."""

    adapter_name = "vast-fixture"
    schema_version = "1.0"

    def __init__(self) -> None:
        self._segments: dict[str, VideoSegment] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._sources: dict[str, str] = {}
        self._detections: dict[str, Detection] = {}
        self._lineage: dict[str, LineageRecord] = {}
        self._reports: dict[str, EvidenceReport] = {}

    @property
    def model_id(self) -> str | None:
        return STORE_MODEL_ID

    @property
    def mode(self) -> AdapterMode:
        return AdapterMode.FIXTURE

    async def health(self) -> AdapterHealth:
        started = time.perf_counter()
        return AdapterHealth(
            adapter_name=self.adapter_name,
            model_id=self.model_id,
            mode=self.mode,
            status=HealthStatus.OK,
            checked_at=utc_now(),
            latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
            detail=f"indexed_segments={len(self._segments)}",
        )

    async def put_source(self, video: SourceVideo, local_path: str) -> str:
        _ = local_path
        self._sources[video.video_id] = video.storage_uri
        return video.storage_uri

    async def upsert_segments(
        self, segments: Sequence[VideoSegment], embeddings: Sequence[Embedding]
    ) -> None:
        segs = tuple(segments)
        embs = tuple(embeddings)
        if len(segs) != len(embs):
            raise ValidationFailed("segments and embeddings must be aligned 1:1")
        for segment, embedding in zip(segs, embs, strict=True):
            vector = np.asarray(embedding.vector, dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                raise ValidationFailed("zero embedding is not indexable")
            self._segments[segment.segment_id] = _copy_frozen(segment)
            self._vectors[segment.segment_id] = vector / norm

    async def upsert_detections(self, detections: Sequence[Detection]) -> None:
        for detection in tuple(detections):
            self._detections[detection.detection_id] = _copy_frozen(detection)

    async def search(self, request: SearchRequest) -> Sequence[RetrievedSegment]:
        query = np.asarray(request.query_embedding.vector, dtype=np.float64)
        qnorm = float(np.linalg.norm(query))
        if qnorm == 0.0:
            raise ValidationFailed("query embedding must be non-zero")
        query = query / qnorm
        wanted_classes = set(request.subject_classes)
        video_segments = [
            segment
            for segment in self._segments.values()
            if segment.video_id == request.video_id
            and segment.source_sha256 == request.source_sha256
        ]
        # Custom footage (games, indoor, etc.) often has empty YOLO class tags. If no
        # segment in this video carries any requested class, skip the class gate so
        # semantic/lexical retrieval still works.
        video_has_wanted = any(
            wanted_classes.intersection(item.class_name for item in segment.detected_classes)
            for segment in video_segments
        )
        apply_class_filter = bool(wanted_classes) and video_has_wanted
        ranked: list[tuple[float, int, RetrievedSegment]] = []
        for segment_id, segment in self._segments.items():
            if (
                segment.video_id != request.video_id
                or segment.source_sha256 != request.source_sha256
            ):
                continue
            if request.time_range is not None:
                if (
                    segment.end_pts_us <= request.time_range.start_pts_us
                    or segment.start_pts_us >= request.time_range.end_pts_us
                ):
                    continue
            if apply_class_filter:
                present = {item.class_name for item in segment.detected_classes}
                if not present.intersection(wanted_classes):
                    continue
            vector = self._vectors[segment_id]
            similarity = float(np.dot(vector, query))
            similarity = max(-1.0, min(1.0, similarity))
            ranked.append((similarity, segment.ordinal, segment))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        top_k = request.top_k
        out: list[RetrievedSegment] = []
        for rank, (similarity, _ordinal, segment) in enumerate(ranked[:top_k], start=1):
            out.append(
                RetrievedSegment(
                    segment=_copy_frozen(segment),
                    similarity=similarity,
                    vector_rank=rank,
                )
            )
        return tuple(out)

    async def put_lineage(self, record: LineageRecord) -> None:
        self._lineage[record.lineage_id] = _copy_frozen(record)

    async def put_report(self, report: EvidenceReport) -> None:
        self._reports[report.report_id] = _copy_frozen(report)

    def indexed_snapshot(self) -> Mapping[str, VideoSegment]:
        return dict(self._segments)
