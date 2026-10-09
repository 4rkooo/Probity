"""Thin sync HTTP client for the live Probity API (used by the Streamlit UI)."""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx

from probity.domain.api import SearchCreateRequest, VideoAccepted
from probity.domain.models import CaseWorkspace, JobView, SearchEvidence, SourceVideo, VideoSegment


def default_api_base() -> str:
    return os.environ.get("PROBITY_API_BASE", "http://127.0.0.1:8000").rstrip("/")


def _idem_key(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


class LiveApiClient:
    """Minimal client for case creation, upload, job polling, and search."""

    def __init__(self, base_url: str | None = None, timeout_s: float = 120.0) -> None:
        self.base_url = (base_url or default_api_base()).rstrip("/")
        self.timeout_s = timeout_s

    def health_ok(self) -> bool:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{self.base_url}/v1/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def create_case(self, display_name: str, owner_alias: str = "analyst") -> CaseWorkspace:
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(
                f"{self.base_url}/v1/cases",
                json={"display_name": display_name, "owner_alias": owner_alias},
                headers={"Idempotency-Key": _idem_key("case")},
            )
        response.raise_for_status()
        return CaseWorkspace.model_validate(response.json())

    def get_case(self, case_id: str) -> CaseWorkspace:
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.get(f"{self.base_url}/v1/cases/{case_id}")
        response.raise_for_status()
        return CaseWorkspace.model_validate(response.json())

    def upload_video(
        self,
        case_id: str,
        data: bytes,
        *,
        original_name: str,
        content_type: str = "video/mp4",
    ) -> VideoAccepted:
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(
                f"{self.base_url}/v1/cases/{case_id}/videos",
                params={"original_name": original_name},
                content=data,
                headers={
                    "Idempotency-Key": _idem_key("upload"),
                    "Content-Type": content_type,
                    "Content-Length": str(len(data)),
                },
            )
        response.raise_for_status()
        return VideoAccepted.model_validate(response.json())

    def get_video(self, video_id: str) -> SourceVideo:
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.get(f"{self.base_url}/v1/videos/{video_id}")
        response.raise_for_status()
        return SourceVideo.model_validate(response.json())

    def get_job(self, job_id: str) -> JobView:
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.get(f"{self.base_url}/v1/jobs/{job_id}")
        response.raise_for_status()
        return JobView.model_validate(response.json())

    def cancel_job(self, job_id: str) -> JobView:
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(
                f"{self.base_url}/v1/jobs/{job_id}/cancel",
                headers={"Idempotency-Key": _idem_key("cancel")},
            )
        response.raise_for_status()
        return JobView.model_validate(response.json())

    def list_segments(self, video_id: str) -> list[VideoSegment]:
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.get(f"{self.base_url}/v1/videos/{video_id}/segments")
        response.raise_for_status()
        return [VideoSegment.model_validate(item) for item in response.json()]

    def search(self, video_id: str, query: str, max_results: int = 5) -> SearchEvidence:
        body = SearchCreateRequest(query=query, max_results=max_results)
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(
                f"{self.base_url}/v1/videos/{video_id}/searches",
                json=body.model_dump(mode="json"),
                headers={"Idempotency-Key": _idem_key("search")},
            )
        response.raise_for_status()
        return SearchEvidence.model_validate(response.json())

    def error_detail(self, exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            try:
                payload: dict[str, Any] = exc.response.json()
                err = payload.get("error") or {}
                code = err.get("code", exc.response.status_code)
                message = err.get("message", exc.response.text)
                return f"{code}: {message}"
            except Exception:
                return f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
        return str(exc)
