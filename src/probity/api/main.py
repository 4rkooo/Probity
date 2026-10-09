"""FastAPI application: frozen route surface (section 7).

The route signatures, paths, status codes and response models below define the committed
OpenAPI snapshot (``contracts/openapi.json``). Implementations may change bodies, but not the
generated OpenAPI document, without coordinator review.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Path, Query, Request, status
from fastapi.responses import Response

from probity import __version__
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


def _todo() -> Any:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="not implemented")


def register_routes(app: FastAPI) -> None:
    @app.get("/v1/health", response_model=HealthResponse, tags=["system"])
    async def health() -> Any:
        return _todo()

    @app.post(
        "/v1/cases",
        status_code=201,
        response_model=CaseWorkspace,
        responses=_errors(409),
        tags=["cases"],
    )
    async def create_case(body: CreateCaseRequest, idempotency_key: IdemHeader) -> Any:
        return _todo()

    @app.get(
        "/v1/cases/{case_id}", response_model=CaseWorkspace, responses=_errors(404), tags=["cases"]
    )
    async def get_case(case_id: IdPath) -> Any:
        return _todo()

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
        return _todo()

    @app.get(
        "/v1/videos/{video_id}", response_model=SourceVideo, responses=_errors(404), tags=["videos"]
    )
    async def get_video(video_id: IdPath) -> Any:
        return _todo()

    @app.get(
        "/v1/videos/{video_id}/segments",
        response_model=list[VideoSegment],
        responses=_errors(404),
        tags=["videos"],
    )
    async def list_segments(video_id: IdPath) -> Any:
        return _todo()

    @app.get("/v1/jobs/{job_id}", response_model=JobView, responses=_errors(404), tags=["jobs"])
    async def get_job(job_id: IdPath) -> Any:
        return _todo()

    @app.post(
        "/v1/jobs/{job_id}/cancel",
        status_code=202,
        response_model=JobView,
        responses=_errors(404, 409),
        tags=["jobs"],
    )
    async def cancel_job(job_id: IdPath, idempotency_key: IdemHeader) -> Any:
        return _todo()

    @app.post(
        "/v1/videos/{video_id}/searches",
        status_code=200,
        response_model=SearchEvidence,
        responses=_errors(404, 409, 503),
        tags=["search"],
    )
    async def create_search(
        video_id: IdPath, body: SearchCreateRequest, idempotency_key: IdemHeader
    ) -> Any:
        return _todo()

    @app.get(
        "/v1/searches/{search_id}",
        response_model=SearchEvidence,
        responses=_errors(404),
        tags=["search"],
    )
    async def get_search(search_id: IdPath) -> Any:
        return _todo()

    @app.post(
        "/v1/videos/{video_id}/tracks",
        status_code=202,
        response_model=TrackAccepted,
        responses=_errors(404, 409, 503),
        tags=["tracks"],
    )
    async def create_track(
        video_id: IdPath, body: TrackCreateRequest, idempotency_key: IdemHeader
    ) -> Any:
        return _todo()

    @app.get("/v1/tracks/{track_id}", response_model=Track, responses=_errors(404), tags=["tracks"])
    async def get_track(track_id: IdPath) -> Any:
        return _todo()

    @app.post(
        "/v1/reconstructions",
        status_code=202,
        response_model=RunAccepted,
        responses=_errors(404, 409),
        tags=["reconstructions"],
    )
    async def create_reconstruction(
        body: ReconstructionCreateRequest, idempotency_key: IdemHeader
    ) -> Any:
        return _todo()

    @app.get(
        "/v1/reconstructions/{run_id}",
        response_model=ReconstructionRun,
        responses=_errors(404),
        tags=["reconstructions"],
    )
    async def get_reconstruction(run_id: IdPath) -> Any:
        return _todo()

    @app.get(
        "/v1/reconstructions/{run_id}/decisions",
        response_model=list[PolicyDecision],
        responses=_errors(404),
        tags=["reconstructions"],
    )
    async def list_decisions(run_id: IdPath) -> Any:
        return _todo()

    @app.get(
        "/v1/reconstructions/{run_id}/provenance",
        response_model=PixelOrigin,
        responses=_errors(404),
        tags=["reconstructions"],
    )
    async def get_pixel_origin(
        run_id: IdPath,
        x: Annotated[int, Query(ge=0)],
        y: Annotated[int, Query(ge=0)],
    ) -> Any:
        return _todo()

    @app.post(
        "/v1/reconstructions/{run_id}/reviews",
        status_code=201,
        response_model=HumanReview,
        responses=_errors(404, 409),
        tags=["reviews"],
    )
    async def create_review(
        run_id: IdPath, body: ReviewCreateRequest, idempotency_key: IdemHeader
    ) -> Any:
        return _todo()

    @app.post(
        "/v1/reports",
        status_code=202,
        response_model=ReportAccepted,
        responses=_errors(404, 409),
        tags=["reports"],
    )
    async def create_report(body: ReportCreateRequest, idempotency_key: IdemHeader) -> Any:
        return _todo()

    @app.get(
        "/v1/reports/{report_id}",
        response_model=EvidenceReport,
        responses=_errors(404),
        tags=["reports"],
    )
    async def get_report(report_id: IdPath) -> Any:
        return _todo()

    @app.get(
        "/v1/assets/{asset_id}",
        response_class=Response,
        responses=ASSET_RESPONSES,
        tags=["assets"],
    )
    async def get_asset(
        asset_id: IdPath,
        range_header: Annotated[str | None, Header(alias="Range")] = None,
    ) -> Any:
        return _todo()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Probity API",
        version=__version__,
        description="Research/demo prototype - not for legal conclusions.",
    )
    register_routes(app)
    return app


app = create_app()
