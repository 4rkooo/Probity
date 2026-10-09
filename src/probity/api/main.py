"""FastAPI application: frozen route surface (section 7).

The route signatures, paths, status codes and response models below define the committed
OpenAPI snapshot (``contracts/openapi.json``). Implementations may change bodies, but not the
generated OpenAPI document, without coordinator review.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import FastAPI, Header, Path, Query, Request, status
from fastapi.responses import Response

from probity import __version__
from probity.api import services as svc
from probity.api.deps import (
    AppServices,
    ServicesNotWired,
    build_default_services,
    load_fixture_catalog,
    require_services,
)
from probity.api.errors import correlation_id_of, install_exception_handlers
from probity.api.middleware import CorrelationMiddleware
from probity.config import get_settings
from probity.domain.api import (
    CreateCaseRequest,
    ErrorEnvelope,
    HealthResponse,
    IdempotencyKey,
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
from probity.domain.models import (
    CaseWorkspace,
    EvidenceReport,
    HumanReview,
    JobView,
    PixelOrigin,
    PolicyDecision,
    ReconstructionRun,
    SearchEvidence,
    SourceVideo,
    Track,
    VideoSegment,
)
from probity.domain.policy import load_policy

UUID7_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
IdPath = Annotated[str, Path(pattern=UUID7_PATTERN)]
IdemHeader = Annotated[IdempotencyKey, Header(alias="Idempotency-Key")]


def _errors(*codes: int) -> dict[int | str, dict[str, Any]]:
    descriptions = {
        404: "Record not found",
        409: "Conflict (idempotency, state, or artifact hash)",
        413: "Payload too large",
        415: "Unsupported media",
        422: "Validation failed",
        500: "Internal error (sanitized)",
        503: "Sponsor retry budget exhausted (retryable)",
    }
    return {c: {"model": ErrorEnvelope, "description": descriptions[c]} for c in (*codes, 422, 500)}


UPLOAD_BODY: dict[str, Any] = {
    "requestBody": {
        "required": False,
        "description": (
            "Raw MP4 bytes (streamed, max 262144000). Omit the body and pass ?fixture_id= "
            "to select a verified demo fixture. Client filename and MIME are not trusted."
        ),
        "content": {
            "video/mp4": {"schema": {"type": "string", "format": "binary"}},
            "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
        },
    }
}

ASSET_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {"description": "Full asset", "content": {"application/octet-stream": {}}},
    206: {"description": "Byte range", "content": {"application/octet-stream": {}}},
    416: {"model": ErrorEnvelope, "description": "Range not satisfiable"},
    **_errors(404),
}


def register_routes(app: FastAPI) -> None:
    @app.get("/v1/health", response_model=HealthResponse, tags=["system"])
    async def health(request: Request) -> Any:
        return await svc.health_with_adapters(request)

    @app.post(
        "/v1/cases",
        status_code=201,
        response_model=CaseWorkspace,
        responses=_errors(409),
        tags=["cases"],
    )
    async def create_case(
        request: Request, body: CreateCaseRequest, idempotency_key: IdemHeader
    ) -> Any:
        services = require_services(request)
        return await svc.mutate(
            request,
            services,
            key=idempotency_key,
            case_id=None,
            body=body,
            status_code=status.HTTP_201_CREATED,
            produce=lambda: svc.create_case(services, body),
        )

    @app.get(
        "/v1/cases/{case_id}", response_model=CaseWorkspace, responses=_errors(404), tags=["cases"]
    )
    async def get_case(request: Request, case_id: IdPath) -> Any:
        return svc.get_case(require_services(request), case_id)

    @app.post(
        "/v1/cases/{case_id}/videos",
        status_code=202,
        response_model=VideoAccepted,
        responses=_errors(404, 409, 413, 415),
        openapi_extra=UPLOAD_BODY,
        tags=["videos"],
    )
    async def create_video(
        request: Request,
        case_id: IdPath,
        idempotency_key: IdemHeader,
        fixture_id: Annotated[str | None, Query(pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")] = None,
        original_name: Annotated[str | None, Query(max_length=255)] = None,
    ) -> Any:
        services = require_services(request)
        return await svc.create_video(
            request,
            services,
            case_id=case_id,
            key=idempotency_key,
            fixture_id=fixture_id,
            original_name=original_name,
        )

    @app.get(
        "/v1/videos/{video_id}", response_model=SourceVideo, responses=_errors(404), tags=["videos"]
    )
    async def get_video(request: Request, video_id: IdPath) -> Any:
        return svc.get_video(require_services(request), video_id)

    @app.get(
        "/v1/videos/{video_id}/segments",
        response_model=list[VideoSegment],
        responses=_errors(404),
        tags=["videos"],
    )
    async def list_segments(request: Request, video_id: IdPath) -> Any:
        return svc.list_segments(require_services(request), video_id)

    @app.get("/v1/jobs/{job_id}", response_model=JobView, responses=_errors(404), tags=["jobs"])
    async def get_job(request: Request, job_id: IdPath) -> Any:
        return svc.get_job(require_services(request), job_id)

    @app.post(
        "/v1/jobs/{job_id}/cancel",
        status_code=202,
        response_model=JobView,
        responses=_errors(404, 409),
        tags=["jobs"],
    )
    async def cancel_job(request: Request, job_id: IdPath, idempotency_key: IdemHeader) -> Any:
        services = require_services(request)
        return await svc.mutate(
            request,
            services,
            key=idempotency_key,
            case_id=None,
            body=None,
            status_code=status.HTTP_202_ACCEPTED,
            produce=lambda: svc.cancel_job(services, job_id),
        )

    @app.post(
        "/v1/videos/{video_id}/searches",
        status_code=200,
        response_model=SearchEvidence,
        responses=_errors(404, 409, 503),
        tags=["search"],
    )
    async def create_search(
        request: Request, video_id: IdPath, body: SearchCreateRequest, idempotency_key: IdemHeader
    ) -> Any:
        services = require_services(request)
        return await svc.mutate(
            request,
            services,
            key=idempotency_key,
            case_id=None,
            body=body,
            status_code=status.HTTP_200_OK,
            produce=lambda: svc.create_search(services, video_id, body, correlation_id_of(request)),
        )

    @app.get(
        "/v1/searches/{search_id}",
        response_model=SearchEvidence,
        responses=_errors(404),
        tags=["search"],
    )
    async def get_search(request: Request, search_id: IdPath) -> Any:
        return svc.get_search(require_services(request), search_id)

    @app.post(
        "/v1/videos/{video_id}/tracks",
        status_code=202,
        response_model=TrackAccepted,
        responses=_errors(404, 409, 503),
        tags=["tracks"],
    )
    async def create_track(
        request: Request, video_id: IdPath, body: TrackCreateRequest, idempotency_key: IdemHeader
    ) -> Any:
        services = require_services(request)
        return await svc.mutate(
            request,
            services,
            key=idempotency_key,
            case_id=None,
            body=body,
            status_code=status.HTTP_202_ACCEPTED,
            produce=lambda: svc.create_track(request, services, video_id, body),
        )

    @app.get("/v1/tracks/{track_id}", response_model=Track, responses=_errors(404), tags=["tracks"])
    async def get_track(request: Request, track_id: IdPath) -> Any:
        return svc.get_track(require_services(request), track_id)

    @app.post(
        "/v1/reconstructions",
        status_code=202,
        response_model=RunAccepted,
        responses=_errors(404, 409),
        tags=["reconstructions"],
    )
    async def create_reconstruction(
        request: Request, body: ReconstructionCreateRequest, idempotency_key: IdemHeader
    ) -> Any:
        services = require_services(request)
        return await svc.mutate(
            request,
            services,
            key=idempotency_key,
            case_id=None,
            body=body,
            status_code=status.HTTP_202_ACCEPTED,
            produce=lambda: svc.create_reconstruction(request, services, body),
        )

    @app.get(
        "/v1/reconstructions/{run_id}",
        response_model=ReconstructionRun,
        responses=_errors(404),
        tags=["reconstructions"],
    )
    async def get_reconstruction(request: Request, run_id: IdPath) -> Any:
        return svc.get_reconstruction(require_services(request), run_id)

    @app.get(
        "/v1/reconstructions/{run_id}/decisions",
        response_model=list[PolicyDecision],
        responses=_errors(404),
        tags=["reconstructions"],
    )
    async def list_decisions(request: Request, run_id: IdPath) -> Any:
        return svc.list_decisions(require_services(request), run_id)

    @app.get(
        "/v1/reconstructions/{run_id}/provenance",
        response_model=PixelOrigin,
        responses=_errors(404),
        tags=["reconstructions"],
    )
    async def get_pixel_origin(
        request: Request,
        run_id: IdPath,
        x: Annotated[int, Query(ge=0)],
        y: Annotated[int, Query(ge=0)],
    ) -> Any:
        return svc.get_pixel_origin(require_services(request), run_id, x, y)

    @app.post(
        "/v1/reconstructions/{run_id}/reviews",
        status_code=201,
        response_model=HumanReview,
        responses=_errors(404, 409),
        tags=["reviews"],
    )
    async def create_review(
        request: Request, run_id: IdPath, body: ReviewCreateRequest, idempotency_key: IdemHeader
    ) -> Any:
        services = require_services(request)
        return await svc.mutate(
            request,
            services,
            key=idempotency_key,
            case_id=None,
            body=body,
            status_code=status.HTTP_201_CREATED,
            produce=lambda: svc.create_review(services, run_id, body),
        )

    @app.post(
        "/v1/reports",
        status_code=202,
        response_model=ReportAccepted,
        responses=_errors(404, 409),
        tags=["reports"],
    )
    async def create_report(
        request: Request, body: ReportCreateRequest, idempotency_key: IdemHeader
    ) -> Any:
        services = require_services(request)
        return await svc.mutate(
            request,
            services,
            key=idempotency_key,
            case_id=None,
            body=body,
            status_code=status.HTTP_202_ACCEPTED,
            produce=lambda: svc.create_report(request, services, body),
        )

    @app.get(
        "/v1/reports/{report_id}",
        response_model=EvidenceReport,
        responses=_errors(404),
        tags=["reports"],
    )
    async def get_report(request: Request, report_id: IdPath) -> Any:
        return svc.get_report(require_services(request), report_id)

    @app.get(
        "/v1/assets/{asset_id}",
        response_class=Response,
        responses=ASSET_RESPONSES,
        tags=["assets"],
    )
    async def get_asset(
        request: Request,
        asset_id: IdPath,
        range_header: Annotated[str | None, Header(alias="Range")] = None,
    ) -> Any:
        return svc.get_asset(request, require_services(request), asset_id, range_header)


def create_app(services: AppServices | None = None) -> FastAPI:
    app = FastAPI(
        title="Probity API",
        version=__version__,
        description="Research/demo prototype - not for legal conclusions.",
    )
    settings = get_settings()
    app.state.settings = settings
    try:
        app.state.fixture_catalog = load_fixture_catalog(settings.fixture_manifest)
    except Exception:
        app.state.fixture_catalog = None
    if services is None:
        try:
            services = build_default_services()
        except ServicesNotWired:
            services = None
    app.state.services = services
    if services is not None:
        app.state.settings = services.settings
        app.state.fixture_catalog = services.fixture_catalog
        app.state.policy = services.policy
    else:
        try:
            app.state.policy = load_policy(settings.policy_path)
        except Exception:
            app.state.policy = None
    install_exception_handlers(app)
    register_routes(app)
    app.add_middleware(CorrelationMiddleware)
    return app


app = create_app()
