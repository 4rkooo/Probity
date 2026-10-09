"""Endpoint use-cases. Talks only to ports on AppServices."""

from __future__ import annotations

import logging
from pathlib import Path

import anyio
import numpy as np
from fastapi import Request
from fastapi.responses import Response

from probity import __version__
from probity.api.assets import build_asset_response, resolve_asset_file
from probity.api.deps import AppServices, optional_services
from probity.api.errors import correlation_id_of
from probity.api.idempotency import (
    fingerprint,
    lookup_replay,
    replay_response,
    request_route,
    save_success,
)
from probity.api.uploads import accept_upload
from probity.domain.api import (
    CreateCaseRequest,
    FixtureInfo,
    HealthResponse,
    ReconstructionCreateRequest,
    ReportAccepted,
    ReportCreateRequest,
    ReviewCreateRequest,
    RunAccepted,
    SearchCreateRequest,
    TrackAccepted,
    TrackCreateRequest,
    VideoAccepted,
)
from probity.domain.enums import (
    AssetKind,
    CaseStatus,
    ErrorCode,
    HealthStatus,
    IngestState,
    ReconstructionState,
    ReviewDecision,
    TrackState,
)
from probity.domain.errors import (
    ArtifactHashConflict,
    InvalidStateTransition,
    NotFound,
    ProbityError,
    SourceHashMismatch,
    SourcePathViolation,
    ValidationFailed,
)
from probity.domain.ids import parse_frame_id
from probity.domain.models import (
    AssetRef,
    CaseWorkspace,
    EvidenceReport,
    HumanReview,
    JobView,
    PixelOrigin,
    PixelProvenance,
    PolicyDecision,
    ReconstructionRun,
    SearchEvidence,
    SourceVideo,
    Track,
    VideoSegment,
)
from probity.media.hashing import sha256_file
from probity.ports import (
    IngestPayload,
    JobSpec,
    ReconstructionRequest,
    ReconstructPayload,
    ReportPayload,
    SearchQuery,
    TrackPayload,
    TrackRequest,
)

logger = logging.getLogger("probity.api")

SEARCHABLE_STATES = frozenset({IngestState.SEARCHABLE, IngestState.PARTIAL})


def health(request: Request) -> HealthResponse:
    services = optional_services(request)
    settings = getattr(request.app.state, "settings", None)
    catalog = getattr(request.app.state, "fixture_catalog", None)
    fixtures: tuple[FixtureInfo, ...] = ()
    if catalog is not None:
        fixtures = tuple(
            FixtureInfo(
                fixture_id=item.fixture_id,
                display_name=item.display_name,
                source_sha256=item.source.sha256,
                duration_us=item.source.duration_us,
                license_note=item.source.license_note,
                synthetic=item.source.synthetic,
            )
            for item in catalog.fixtures
        )
    adapters: tuple = ()
    database: str = "unavailable"
    status = "degraded"
    mode = settings.mode if settings is not None else None
    if services is not None:
        mode = services.settings.mode
        database = "ok"
        fixtures = tuple(
            FixtureInfo(
                fixture_id=item.fixture_id,
                display_name=item.display_name,
                source_sha256=item.source.sha256,
                duration_us=item.source.duration_us,
                license_note=item.source.license_note,
                synthetic=item.source.synthetic,
            )
            for item in services.fixture_catalog.fixtures
        )
    if mode is None:
        from probity.domain.enums import OperatingMode

        mode = OperatingMode.AUTO
    return HealthResponse(
        status=status if services is None else "ok",
        version=__version__,
        mode=mode,
        process="api",
        database=database,  # type: ignore[arg-type]
        adapters=adapters,
        fixtures=fixtures,
    )


async def health_with_adapters(request: Request) -> HealthResponse:
    base = health(request)
    services = optional_services(request)
    if services is None:
        return base
    adapters = tuple(await services.adapter_health())
    degraded = any(row.status is not HealthStatus.OK for row in adapters)
    status = "degraded" if degraded else "ok"
    return HealthResponse(
        status=status,
        version=base.version,
        mode=base.mode,
        process="api",
        database=base.database,
        adapters=adapters,
        fixtures=base.fixtures,
    )


def create_case(services: AppServices, body: CreateCaseRequest) -> CaseWorkspace:
    case = CaseWorkspace.create(
        case_id=services.new_id(),
        display_name=body.display_name,
        mode=services.settings.mode,
        owner_alias=body.owner_alias,
        status=CaseStatus.ACTIVE,
    )
    services.repository.put_case(case)
    return case


def get_case(services: AppServices, case_id: str) -> CaseWorkspace:
    return services.repository.get_case(case_id)


async def create_video(
    request: Request,
    services: AppServices,
    *,
    case_id: str,
    key: str,
    fixture_id: str | None,
    original_name: str | None,
) -> Response | VideoAccepted:
    services.repository.get_case(case_id)
    staged = None
    try:
        staged, manifest = await accept_upload(request, services, fixture_id=fixture_id)
        route = request_route(request)
        digest = fingerprint(
            route=route,
            case_id=case_id,
            upload_sha256=staged.sha256,
            query={"fixture_id": fixture_id, "original_name": original_name},
        )
        existing = lookup_replay(
            services, route=route, case_id=case_id, key=key, request_sha256=digest
        )
        if existing is not None:
            services.source_store.discard(staged)
            staged = None
            return replay_response(existing, correlation_id_of(request))

        probe = await anyio.to_thread.run_sync(services.prober.probe, staged.staging_path)
        stored = await anyio.to_thread.run_sync(services.source_store.commit, staged)
        staged = None

        asset_id = services.new_id()
        video_id = services.new_id()
        display_name = original_name or (
            Path(manifest.source.path).name if manifest is not None else "upload.mp4"
        )
        video = SourceVideo.create(
            video_id=video_id,
            case_id=case_id,
            original_name=display_name,
            sha256=stored.sha256,
            byte_length=stored.byte_length,
            mime="video/mp4",
            container=probe.container,
            video_codec=probe.video_codec,
            width_px=probe.width_px,
            height_px=probe.height_px,
            duration_us=probe.duration_us,
            time_base=probe.time_base,
            nominal_fps=probe.nominal_fps,
            frame_count=probe.frame_count,
            has_audio=probe.has_audio,
            storage_uri=stored.storage_uri,
            source_asset_uri=f"asset://{asset_id}",
            probe_sha256=probe.probe_sha256,
            ingest_state=IngestState.STORED,
            license_note=manifest.source.license_note if manifest is not None else None,
            fixture_id=manifest.fixture_id if manifest is not None else None,
        )
        asset = AssetRef(
            asset_id=asset_id,
            case_id=case_id,
            kind=AssetKind.SOURCE_VIDEO,
            storage_uri=stored.storage_uri,
            media_type="video/mp4",
            byte_length=stored.byte_length,
            sha256=stored.sha256,
            non_evidentiary=False,
            created_at=services.clock(),
        )
        services.repository.put_video(video)
        services.repository.put_asset(asset)
        job = services.jobs.create(
            JobSpec(
                job_id=services.new_id(),
                case_id=case_id,
                subject_id=video_id,
                payload=IngestPayload(video_id=video_id, fixture_id=video.fixture_id),
                correlation_id=correlation_id_of(request),
            )
        )
        accepted = VideoAccepted(
            video_id=video_id,
            job_id=job.job_id,
            sha256=stored.sha256,
            deduplicated=stored.deduplicated,
        )
        save_success(
            services,
            route=route,
            case_id=case_id,
            key=key,
            request_sha256=digest,
            status_code=202,
            body=accepted,
        )
        return accepted
    except Exception:
        if staged is not None:
            try:
                services.source_store.discard(staged)
            except Exception:
                logger.debug("staged upload discard failed")
        raise


def get_video(services: AppServices, video_id: str) -> SourceVideo:
    return services.repository.get_video(video_id)


def list_segments(services: AppServices, video_id: str) -> list[VideoSegment]:
    services.repository.get_video(video_id)
    segments = list(services.repository.list_segments(video_id))
    segments.sort(key=lambda item: item.ordinal)
    return segments


def get_job(services: AppServices, job_id: str) -> JobView:
    return services.jobs.get(job_id)


def cancel_job(services: AppServices, job_id: str) -> JobView:
    return services.jobs.request_cancel(job_id)


async def create_search(
    services: AppServices, video_id: str, body: SearchCreateRequest, correlation_id: str
) -> SearchEvidence:
    video = services.repository.get_video(video_id)
    if video.ingest_state not in SEARCHABLE_STATES:
        raise ProbityError(
            "Video is not searchable until ingestion reaches SEARCHABLE or PARTIAL.",
            code=ErrorCode.VIDEO_NOT_SEARCHABLE,
        )
    evidence = await services.search.search(
        video,
        SearchQuery(query=body.query, max_results=body.max_results, time_range=body.time_range),
        correlation_id,
    )
    services.repository.put_search(evidence)
    return evidence


def get_search(services: AppServices, search_id: str) -> SearchEvidence:
    return services.repository.get_search(search_id)


def create_track(
    request: Request, services: AppServices, video_id: str, body: TrackCreateRequest
) -> TrackAccepted:
    video = services.repository.get_video(video_id)
    seed_video, _ = parse_frame_id(body.seed_frame_id)
    if seed_video != video.video_id:
        raise ValidationFailed("seed_frame_id must belong to the target video.")
    track_id = services.new_id()
    payload = TrackPayload(
        request=TrackRequest(
            track_id=track_id,
            case_id=video.case_id,
            video_id=video.video_id,
            source_sha256=video.sha256,
            subject_type=body.subject_type,
            seed_frame_id=body.seed_frame_id,
            seed_bbox_px=body.seed_bbox_px,
            seed_detection_id=body.seed_detection_id,
            window_radius_us=int(services.policy.track.window_radius_s * 1_000_000),
        )
    )
    job = services.jobs.create(
        JobSpec(
            job_id=services.new_id(),
            case_id=video.case_id,
            subject_id=track_id,
            payload=payload,
            correlation_id=correlation_id_of(request),
        )
    )
    return TrackAccepted(track_id=track_id, job_id=job.job_id)


def get_track(services: AppServices, track_id: str) -> Track:
    return services.repository.get_track(track_id)


def create_reconstruction(
    request: Request, services: AppServices, body: ReconstructionCreateRequest
) -> RunAccepted:
    track = services.repository.get_track(body.track_id)
    if track.state is not TrackState.CONFIRMED or not track.confirmed:
        raise InvalidStateTransition("Reconstruction requires a CONFIRMED track.")
    target_video, _ = parse_frame_id(body.target_frame_id)
    if target_video != track.video_id:
        raise InvalidStateTransition("target_frame_id must belong to the track's video.")
    video = services.repository.get_video(track.video_id)
    run_id = services.new_id()
    payload = ReconstructPayload(
        request=ReconstructionRequest(
            run_id=run_id,
            case_id=track.case_id,
            video_id=track.video_id,
            source_sha256=video.sha256,
            source_storage_uri=video.storage_uri,
            track=track,
            target_frame_id=body.target_frame_id,
            target_bbox_px=body.target_bbox_px,
            policy_profile=body.policy_profile,
            config_sha256=services.policy.config_sha256,
        )
    )
    job = services.jobs.create(
        JobSpec(
            job_id=services.new_id(),
            case_id=track.case_id,
            subject_id=run_id,
            payload=payload,
            correlation_id=correlation_id_of(request),
        )
    )
    return RunAccepted(run_id=run_id, job_id=job.job_id)


def get_reconstruction(services: AppServices, run_id: str) -> ReconstructionRun:
    return services.repository.get_run(run_id)


def list_decisions(services: AppServices, run_id: str) -> list[PolicyDecision]:
    services.repository.get_run(run_id)
    return list(services.repository.list_decisions(run_id))


def _npz_path(services: AppServices, provenance: PixelProvenance) -> Path:
    try:
        path = resolve_asset_file(
            services,
            AssetRef(
                asset_id=services.new_id(),
                case_id="00000000-0000-7000-8000-000000000000",
                kind=AssetKind.PROVENANCE_NPZ,
                storage_uri=provenance.class_map_uri,
                media_type="application/octet-stream",
                byte_length=1,
                sha256="0" * 64,
                non_evidentiary=False,
                created_at=services.clock(),
            ),
        )
    except (SourcePathViolation, ValueError, NotFound) as exc:
        raise NotFound("Provenance artifact is not available.") from exc
    if not path.is_file():
        raise NotFound("Provenance artifact is not available.")
    digest, _byte_length = sha256_file(path)
    if digest != provenance.artifact_sha256:
        raise ArtifactHashConflict("Provenance artifact hash does not match the run.")
    return path


def get_pixel_origin(services: AppServices, run_id: str, x: int, y: int) -> PixelOrigin:
    services.repository.get_run(run_id)
    provenance = services.repository.get_provenance(run_id)
    if x >= provenance.width_px or y >= provenance.height_px:
        raise ValidationFailed(
            "Pixel coordinate is outside the provenance frame.",
            details={"x": x, "y": y, "width": provenance.width_px, "height": provenance.height_px},
        )
    arrays = np.load(_npz_path(services, provenance))
    try:
        classes = arrays["class"]
        indexes = arrays["source_index"]
        source_x = arrays["source_x"]
        source_y = arrays["source_y"]
        if y >= classes.shape[0] or x >= classes.shape[1]:
            raise ValidationFailed("Pixel coordinate is outside the provenance arrays.")
        cls_val = int(classes[y, x])
        idx = int(indexes[y, x])
        sx = float(source_x[y, x])
        sy = float(source_y[y, x])
    finally:
        arrays.close()
    if idx >= len(provenance.source_lut):
        raise ValidationFailed("Provenance source index is out of range.")
    entry = provenance.source_lut[idx]
    return PixelOrigin(
        run_id=run_id,
        x=x,
        y=y,
        provenance_class=PixelOrigin.class_name(cls_val),  # type: ignore[arg-type]
        source_index=idx,
        source_frame_id=entry.frame_id,
        source_pts_us=entry.pts_us,
        source_x=sx,
        source_y=sy,
        transform_id=entry.transform_id,
        alignment_method=entry.alignment_method,
        color_gain=entry.color_gain,
        color_bias=entry.color_bias,
        decision_ids=entry.decision_ids,
    )


def create_review(services: AppServices, run_id: str, body: ReviewCreateRequest) -> HumanReview:
    run = services.repository.get_run(run_id)
    if run.state is not ReconstructionState.SUCCEEDED:
        raise InvalidStateTransition("Review requires a SUCCEEDED reconstruction.")
    if run.integrity is None or run.integrity.score_0_100 < services.policy.integrity.min_score:
        raise InvalidStateTransition("Integrity score is below the review minimum.")
    if run.provenance is None or not run.provenance.coverage_complete:
        raise InvalidStateTransition("Review requires complete provenance.")
    if (
        run.provenance.coverage_pct.GENERATED_BLEND != 0.0
        or run.integrity.semantic_generated_pct != 0.0
    ):
        raise InvalidStateTransition("Generated semantic pixels cannot be reviewed.")
    if (
        body.reviewed_result_sha256 != run.result_png_sha256
        or body.reviewed_provenance_sha256 != run.provenance_sha256
    ):
        raise ArtifactHashConflict("Reviewed hashes do not match the current artifacts.")
    if body.decision is ReviewDecision.VETO and (
        body.reason_code is None or not body.comment.strip()
    ):
        raise ValidationFailed("VETO requires a reason_code and a comment.")
    review = HumanReview.create(
        review_id=services.new_id(),
        run_id=run_id,
        reviewer_alias=body.reviewer_alias,
        decision=body.decision,
        reason_code=body.reason_code,
        comment=body.comment,
        reviewed_result_sha256=body.reviewed_result_sha256,
        reviewed_provenance_sha256=body.reviewed_provenance_sha256,
    )
    services.repository.put_review(review)
    return review


def create_report(
    request: Request, services: AppServices, body: ReportCreateRequest
) -> ReportAccepted:
    run = services.repository.get_run(body.run_id)
    review = services.repository.latest_review(body.run_id)
    if (
        review is None
        or review.review_id != body.review_id
        or review.decision is not ReviewDecision.APPROVE
        or review.reviewed_result_sha256 != run.result_png_sha256
        or review.reviewed_provenance_sha256 != run.provenance_sha256
    ):
        raise ProbityError(
            "Export requires the latest APPROVE of the exact current artifact hashes.",
            code=ErrorCode.REVIEW_REQUIRED,
        )
    video = services.repository.get_video(run.video_id)
    verification = services.source_store.verify(video.storage_uri, video.sha256)
    if not verification.verified:
        raise SourceHashMismatch("Source bytes do not match the ingest hash.")
    report_id = services.new_id()
    job = services.jobs.create(
        JobSpec(
            job_id=services.new_id(),
            case_id=run.case_id,
            subject_id=report_id,
            payload=ReportPayload(
                report_id=report_id, run_id=run.run_id, review_id=review.review_id
            ),
            correlation_id=correlation_id_of(request),
        )
    )
    return ReportAccepted(report_id=report_id, job_id=job.job_id)


def get_report(services: AppServices, report_id: str) -> EvidenceReport:
    return services.repository.get_report(report_id)


def get_asset(
    request: Request, services: AppServices, asset_id: str, range_header: str | None
) -> Response:
    try:
        asset = services.repository.get_asset(asset_id)
    except NotFound:
        raise
    try:
        return build_asset_response(request, services, asset, range_header)
    except SourcePathViolation as exc:
        raise ProbityError(GENERIC_INTERNAL_FOR_ASSET, code=ErrorCode.INTERNAL_ERROR) from exc


GENERIC_INTERNAL_FOR_ASSET = "An internal error occurred."


def dump_body(model: object) -> dict:
    from pydantic import BaseModel

    if isinstance(model, BaseModel):
        return model.model_dump(mode="json")
    raise TypeError("expected a pydantic model")


async def mutate(
    request: Request,
    services: AppServices,
    *,
    key: str,
    case_id: str | None,
    body: object | None,
    status_code: int,
    produce,
) -> object:
    route = request_route(request)
    digest = fingerprint(
        route=route,
        case_id=case_id,
        body=dump_body(body) if body is not None else None,
    )
    existing = lookup_replay(services, route=route, case_id=case_id, key=key, request_sha256=digest)
    if existing is not None:
        return replay_response(existing, correlation_id_of(request))
    result = produce()
    if hasattr(result, "__await__"):
        result = await result
    save_success(
        services,
        route=route,
        case_id=case_id,
        key=key,
        request_sha256=digest,
        status_code=status_code,
        body=result,
    )
    return result
