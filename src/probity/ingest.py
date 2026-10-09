"""Fixture-first INGEST handler owned by the platform coordinator.

Describes each deterministic window with the exact Cosmos ingestion prompt, embeds it,
writes a non-canonical thumbnail, and commits the segment before moving on. A later
failure leaves those rows in place.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from probity.domain.enums import (
    AssetKind,
    ErrorCode,
    IndexState,
    IngestState,
    JobKind,
    JobStage,
    JobState,
    RigidSubjectKind,
)
from probity.domain.errors import (
    JobCancelled,
    ProbityError,
    SponsorMalformedResponse,
    SponsorTimeout,
    SponsorUnavailable,
    ValidationFailed,
)
from probity.domain.ids import sha256_hex, utc_now
from probity.domain.models import (
    AssetRef,
    ClassConfidence,
    Embedding,
    FrameManifest,
    JobError,
    PartialCoverage,
    SegmentDescription,
    SourceVideo,
    TimeRangeUs,
    VideoSegment,
)
from probity.domain.prompts import COSMOS_INGESTION_PROMPT
from probity.ports import (
    ClaimedJob,
    ClipReference,
    EvidenceStore,
    IngestPayload,
    JobContext,
    JobOutcome,
    MediaProber,
    Repository,
    Segmenter,
    SegmentWindow,
    SourceVerification,
    VideoUnderstanding,
)
from probity.search.plan import ALLOWED_SUBJECT_CLASSES, CLASS_ALIASES

_SPONSOR = (SponsorTimeout, SponsorUnavailable, SponsorMalformedResponse)
_KIND_CLASS = {
    RigidSubjectKind.LICENSE_PLATE: "license_plate",
    RigidSubjectKind.SIGN: "sign",
}


class DerivedSource(Protocol):
    """SourceStore plus the derived-path methods LocalSourceStore already provides."""

    @property
    def data_dir(self) -> Path: ...

    def verify(self, storage_uri: str, expected_sha256: str) -> SourceVerification: ...

    def resolve_read_path(self, storage_uri: str) -> str: ...

    def derived_path(self, video_id: str, kind: str, entity_id: str, filename: str) -> str: ...


class IngestHandler:
    """``JobHandler`` for ``JobKind.INGEST``. Live Cosmos is only reached via fallback."""

    kind = JobKind.INGEST

    def __init__(
        self,
        *,
        repository: Repository,
        source_store: DerivedSource,
        prober: MediaProber,
        segmenter: Segmenter,
        thumbnails: object,
        understanding: VideoUnderstanding,
        evidence_store: EvidenceStore,
        new_id: Callable[..., str],
    ) -> None:
        self._repository = repository
        self._source_store = source_store
        self._prober = prober
        self._segmenter = segmenter
        self._thumbnails = thumbnails
        self._understanding = understanding
        self._evidence = evidence_store
        self._new_id = new_id

    async def run(self, claimed: ClaimedJob, ctx: JobContext) -> JobOutcome:
        payload = claimed.payload
        if not isinstance(payload, IngestPayload):
            return _failed(ErrorCode.VALIDATION_FAILED, "INGEST payload required", retryable=False)
        video = self._repository.get_video(payload.video_id)
        try:
            path = self._source_store.resolve_read_path(video.storage_uri)
            verification = await asyncio.to_thread(
                self._source_store.verify, video.storage_uri, video.sha256
            )
            if not verification.verified:
                self._revise_video(video, IngestState.FAILED, None)
                return _failed(
                    ErrorCode.SOURCE_HASH_MISMATCH,
                    "Source bytes do not match the ingest hash.",
                    retryable=False,
                )
            manifest = await asyncio.to_thread(
                self._prober.frame_manifest, path, video.video_id, video.sha256
            )
            windows = tuple(
                await asyncio.to_thread(self._segmenter.plan, manifest, video.duration_us)
            )
        except JobCancelled:
            self._revise_video(video, IngestState.CANCELLED, None)
            raise
        except ProbityError as exc:
            self._revise_video(video, IngestState.FAILED, None)
            return _failed(exc.code, _message(exc), retryable=exc.retryable)

        if not windows:
            self._revise_video(video, IngestState.FAILED, None)
            return _failed(ErrorCode.UNSUPPORTED_MEDIA, "segment plan was empty", retryable=False)

        self._repository.put_manifest(manifest)
        video = self._revise_video(video, IngestState.INDEXING, None)
        total = len(windows)
        ctx.progress(JobStage.EXTRACT, 0, total)
        committed: list[VideoSegment] = []
        try:
            for window in windows:
                ctx.cancel.checkpoint()
                segment_id = f"{video.video_id}:s{window.ordinal:04d}"
                try:
                    segment, embedding = await self._commit_segment(
                        video, window, manifest, path, segment_id, ctx, len(committed), total
                    )
                    committed.append(segment)
                    await self._evidence.upsert_segments((segment,), (embedding,))
                except JobCancelled:
                    raise
                except _SPONSOR as exc:
                    return self._sponsor_failure(video, committed, exc)
                except Exception as exc:
                    return self._segment_failure(video, committed, segment_id, exc)
                ctx.progress(JobStage.INDEX, len(committed), total)
        except JobCancelled:
            self._cancelled(video, committed, ctx)

        indexed = _coverage(committed)
        self._revise_video(video, IngestState.SEARCHABLE, indexed)
        return JobOutcome(state=JobState.SUCCEEDED)

    async def _commit_segment(
        self,
        video: SourceVideo,
        window: SegmentWindow,
        manifest: FrameManifest,
        source_path: str,
        segment_id: str,
        ctx: JobContext,
        completed: int,
        total: int,
    ) -> tuple[VideoSegment, Embedding]:
        ctx.progress(JobStage.DESCRIBE, completed, total)
        clip = ClipReference(
            video_id=video.video_id,
            segment_id=segment_id,
            ordinal=window.ordinal,
            source_sha256=video.sha256,
            storage_uri=video.storage_uri,
            start_pts_us=window.start_pts_us,
            end_pts_us=window.end_pts_us,
        )
        description = await self._understanding.describe(clip, COSMOS_INGESTION_PROMPT)
        ctx.progress(JobStage.EMBED, completed, total)
        embeddings = tuple(await self._understanding.embed_segments((description,)))
        if len(embeddings) != 1:
            raise ValidationFailed("embed_segments must return one embedding for the segment")
        embedding = embeddings[0]
        out_path = self._source_store.derived_path(
            video.video_id, "thumbnails", segment_id, "thumb.jpg"
        )
        frame_number = await asyncio.to_thread(
            _extract_thumbnail, self._thumbnails, source_path, window, manifest, out_path
        )
        asset_id = self._new_id()
        data = Path(out_path).read_bytes()
        relative = Path(out_path).resolve().relative_to(self._source_store.data_dir.resolve())
        asset = AssetRef(
            asset_id=asset_id,
            case_id=video.case_id,
            kind=AssetKind.THUMBNAIL,
            storage_uri=relative.as_posix(),
            media_type="image/jpeg",
            byte_length=len(data),
            sha256=sha256_hex(data),
            non_evidentiary=True,
            created_at=utc_now(),
        )
        segment = _segment(video, window, description, embedding, asset_id, frame_number)
        self._repository.put_asset(asset)
        self._repository.put_segments((segment,))
        self._repository.put_embeddings(video.video_id, (embedding,))
        return segment, embedding

    def _sponsor_failure(
        self, video: SourceVideo, committed: Sequence[VideoSegment], exc: ProbityError
    ) -> JobOutcome:
        if committed:
            self._revise_video(video, IngestState.PARTIAL, _coverage(committed))
        else:
            self._revise_video(video, IngestState.FAILED, None)
        return _failed(exc.code, _message(exc), retryable=True)

    def _segment_failure(
        self,
        video: SourceVideo,
        committed: Sequence[VideoSegment],
        failed_id: str,
        exc: BaseException,
    ) -> JobOutcome:
        if committed:
            self._revise_video(video, IngestState.PARTIAL, _coverage(committed))
            return JobOutcome(
                state=JobState.PARTIAL,
                partial=_partial(committed, (failed_id,)),
            )
        self._revise_video(video, IngestState.FAILED, None)
        if isinstance(exc, ProbityError):
            return _failed(exc.code, _message(exc), retryable=exc.retryable)
        return _failed(ErrorCode.INTERNAL_ERROR, _message(exc), retryable=False)

    def _cancelled(
        self, video: SourceVideo, committed: Sequence[VideoSegment], ctx: JobContext
    ) -> None:
        if committed:
            partial = _partial(committed, ())
            self._revise_video(video, IngestState.PARTIAL, _coverage(committed))
            record = getattr(ctx, "record_partial", None)
            if record is not None:
                record(partial)
        else:
            self._revise_video(video, IngestState.CANCELLED, None)
        raise JobCancelled("job cancellation requested")

    def _revise_video(
        self, video: SourceVideo, state: IngestState, indexed: TimeRangeUs | None
    ) -> SourceVideo:
        revised = video.revise(ingest_state=state, indexed_range=indexed)
        self._repository.put_video(revised)
        return revised


def _extract_thumbnail(
    thumbnails: object,
    source_path: str,
    window: SegmentWindow,
    manifest: FrameManifest,
    out_path: str,
) -> int:
    extract = thumbnails.extract_thumbnail
    return int(extract(source_path, window, manifest, out_path))


def _alias_class(term: str) -> str | None:
    cleaned = term.strip().lower()
    mapped = CLASS_ALIASES.get(cleaned)
    if mapped in ALLOWED_SUBJECT_CLASSES:
        return mapped
    for token in cleaned.replace("-", " ").split():
        mapped = CLASS_ALIASES.get(token)
        if mapped in ALLOWED_SUBJECT_CLASSES:
            return mapped
    return None


def classes_from_description(description: SegmentDescription) -> tuple[ClassConfidence, ...]:
    """Objective class tags so retrieval can filter fixture descriptions. Not a detector."""
    scores: dict[str, float] = {}
    for subject in description.rigid_subjects:
        mapped = _KIND_CLASS.get(subject.kind)
        if mapped is not None:
            scores[mapped] = max(scores.get(mapped, 0.0), float(subject.confidence))
    for term in description.search_terms:
        mapped = _alias_class(term)
        if mapped is not None:
            scores.setdefault(mapped, float(description.confidence))
    return tuple(
        ClassConfidence(class_name=name, max_confidence=score, detection_count=1)
        for name, score in scores.items()
    )


def _segment(
    video: SourceVideo,
    window: SegmentWindow,
    description: SegmentDescription,
    embedding: Embedding,
    asset_id: str,
    thumbnail_frame: int,
) -> VideoSegment:
    return VideoSegment.create(
        video_id=video.video_id,
        segment_id=description.segment_id,
        source_sha256=video.sha256,
        ordinal=window.ordinal,
        start_pts_us=window.start_pts_us,
        end_pts_us=window.end_pts_us,
        start_frame=window.start_frame,
        end_frame=window.end_frame,
        thumbnail_uri=f"asset://{asset_id}",
        thumbnail_frame=thumbnail_frame,
        description=description.summary,
        description_model_id=description.model_id,
        search_terms=description.search_terms,
        detected_classes=classes_from_description(description),
        visibility_tags=tuple(item.kind for item in description.visibility),
        uncertainty=description.uncertainty,
        embedding_ref=embedding.embedding_ref,
        embedding_model_id=embedding.model_id,
        embedding_dimension=embedding.dimension,
        index_state=IndexState.INDEXED,
        mode=description.mode,
    )


def _coverage(segments: Sequence[VideoSegment]) -> TimeRangeUs:
    return TimeRangeUs(
        start_pts_us=min(segment.start_pts_us for segment in segments),
        end_pts_us=max(segment.end_pts_us for segment in segments),
    )


def _partial(committed: Sequence[VideoSegment], failed_ids: tuple[str, ...]) -> PartialCoverage:
    return PartialCoverage(
        committed_segment_ids=tuple(segment.segment_id for segment in committed),
        indexed_ranges=tuple(segment.time_range for segment in committed),
        failed_segment_ids=failed_ids,
    )


def _message(exc: BaseException) -> str:
    text = getattr(exc, "message", None) or str(exc)
    text = str(text).strip() or type(exc).__name__
    return text[:500]


def _failed(code: ErrorCode, message: str, *, retryable: bool) -> JobOutcome:
    return JobOutcome(
        state=JobState.FAILED,
        error=JobError(code=code, message=message[:500] or "ingest failed", retryable=retryable),
    )
