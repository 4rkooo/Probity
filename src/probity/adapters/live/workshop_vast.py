"""Builders Challenge VAST transport.

Health-checks the team's VSS/VAST ingress and mirrors evidence into the local
SqlEvidenceStore used by Probity search. Source bytes remain content-addressed
on disk (LocalSourceStore); this transport attributes live VAST availability.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from probity.domain.errors import SponsorMalformedResponse, SponsorUnavailable
from probity.domain.models import (
    Detection,
    Embedding,
    EvidenceReport,
    LineageRecord,
    SourceVideo,
    VideoSegment,
)
from probity.ports import SearchRequest
from probity.search.sqlite_store import SqlEvidenceStore


class WorkshopVastTransport:
    """``VastTransport`` bound to VSS ingress health + local evidence mirror."""

    def __init__(
        self,
        *,
        endpoint: str,
        token: str | None,
        store: SqlEvidenceStore,
        timeout_s: float = 30.0,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._token = token
        self._store = store
        self._timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        if not self._token:
            return {}
        return {"Authorization": f"Bearer {self._token}"}

    async def execute(self, operation: str, payload: Mapping[str, Any]) -> Any:
        if operation == "health":
            return await self._health()
        if operation == "put_source":
            video = SourceVideo.model_validate(payload["video"])
            return await self._store.put_source(video, str(payload["local_path"]))
        if operation == "upsert_segments":
            segments = [VideoSegment.model_validate(item) for item in payload.get("segments") or []]
            embeddings = [
                Embedding.model_validate(item) for item in payload.get("embeddings") or []
            ]
            await self._store.upsert_segments(segments, embeddings)
            return {"ok": True, "count": len(segments)}
        if operation == "upsert_detections":
            detections = [
                Detection.model_validate(item) for item in payload.get("detections") or []
            ]
            await self._store.upsert_detections(detections)
            return {"ok": True, "count": len(detections)}
        if operation == "search":
            request = SearchRequest.model_validate(payload)
            rows = await self._store.search(request)
            return [item.model_dump(mode="json") for item in rows]
        if operation == "put_lineage":
            await self._store.put_lineage(LineageRecord.model_validate(payload))
            return {"ok": True}
        if operation == "put_report":
            await self._store.put_report(EvidenceReport.model_validate(payload))
            return {"ok": True}
        raise SponsorUnavailable(f"unsupported vast operation {operation!r}")

    async def _health(self) -> dict[str, str]:
        # Prefer the VSS frontend/ingress /health documented for the workshop stack.
        urls = (f"{self._endpoint}/health", f"{self._endpoint}/api/v1/health")
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=2.0) as client:
            for url in urls:
                try:
                    response = await client.get(url, headers=self._headers())
                except httpx.HTTPError as exc:
                    last_error = exc
                    continue
                if response.status_code < 400:
                    return {"status": "ok", "url": url}
                last_error = SponsorUnavailable(f"vast health HTTP {response.status_code}")
        if last_error is not None:
            raise SponsorUnavailable("vast ingress health failed") from last_error
        raise SponsorMalformedResponse("vast health returned no usable endpoint")
