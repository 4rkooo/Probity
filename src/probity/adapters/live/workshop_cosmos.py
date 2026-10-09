"""Builders Challenge Cosmos transport: Cosmos3-Reason + Embed1 over CoreWeave NIMs.

Endpoints and tokens come from environment variables. No host or credential is hard-coded.
"""

from __future__ import annotations

import base64
import json
import math
import re
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from probity.domain.enums import InferenceMode
from probity.domain.errors import SponsorMalformedResponse, SponsorTimeout, SponsorUnavailable
from probity.domain.ids import sha256_hex
from probity.domain.models import Embedding, SegmentDescription
from probity.domain.prompts import COSMOS_INGESTION_PROMPT, COSMOS_INGESTION_PROMPT_SHA256
from probity.search.embedding import embedding_input_text

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _auth_headers(token: str | None) -> dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _normalize(vector: list[float]) -> tuple[float, ...]:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= 0:
        raise SponsorMalformedResponse("embedding vector has non-finite norm")
    return tuple(float(x) for x in (arr / norm).tolist())


class WorkshopCosmosTransport:
    """``CosmosTransport`` bound to Cosmos3-Reason (describe) and Embed1 (embeddings)."""

    def __init__(
        self,
        *,
        reason_url: str,
        embed_url: str,
        token: str | None,
        data_dir: Path,
        reason_model_id: str | None = None,
        embed_model_id: str | None = None,
        timeout_s: float = 90.0,
    ) -> None:
        self._reason_url = reason_url.rstrip("/")
        self._embed_url = embed_url.rstrip("/")
        self._token = token
        self._data_dir = Path(data_dir)
        self._reason_model_id = reason_model_id
        self._embed_model_id = embed_model_id
        self._timeout_s = timeout_s

    async def execute(self, operation: str, payload: Mapping[str, Any]) -> Any:
        if operation == "health":
            return await self._health()
        if operation == "describe":
            return await self._describe(payload)
        if operation == "embed_segments":
            return await self._embed_segments(payload)
        if operation == "embed_query":
            return await self._embed_query(payload)
        raise SponsorUnavailable(f"unsupported cosmos operation {operation!r}")

    async def _health(self) -> dict[str, str]:
        headers = _auth_headers(self._token)
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                reason = await client.get(f"{self._reason_url}/v1/models", headers=headers)
                embed = await client.get(f"{self._embed_url}/v1/models", headers=headers)
        except httpx.HTTPError as exc:
            raise SponsorUnavailable("cosmos health request failed") from exc
        if reason.status_code >= 400 or embed.status_code >= 400:
            raise SponsorUnavailable(
                f"cosmos health failed reason={reason.status_code} embed={embed.status_code}"
            )
        return {"status": "ok"}

    async def _resolve_model(self, base: str, configured: str | None) -> str:
        if configured:
            return configured
        headers = _auth_headers(self._token)
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.get(f"{base}/v1/models", headers=headers)
        if response.status_code >= 400:
            raise SponsorUnavailable(f"could not list models at {base}: {response.status_code}")
        data = response.json().get("data") or []
        if not data:
            raise SponsorUnavailable(f"no models listed at {base}")
        return str(data[0]["id"])

    def _source_path(self, storage_uri: str) -> Path:
        path = self._data_dir / storage_uri
        if not path.is_file():
            raise SponsorUnavailable(f"source bytes missing for {storage_uri}")
        return path

    def _extract_clip(self, source: Path, start_pts_us: int, end_pts_us: int) -> bytes:
        start_s = max(0.0, start_pts_us / 1_000_000)
        duration_s = max(0.1, (end_pts_us - start_pts_us) / 1_000_000)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            out = Path(handle.name)
        try:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start_s:.3f}",
                "-i",
                str(source),
                "-t",
                f"{duration_s:.3f}",
                "-c:v",
                "libx264",
                "-an",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                str(out),
            ]
            completed = subprocess.run(cmd, check=False, capture_output=True, timeout=60)
            if completed.returncode != 0 or not out.is_file() or out.stat().st_size <= 0:
                detail = completed.stderr.decode("utf-8", errors="replace")[:300]
                raise SponsorUnavailable(f"ffmpeg segment extract failed: {detail}")
            return out.read_bytes()
        finally:
            out.unlink(missing_ok=True)

    async def _describe(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        clip = payload.get("clip") or {}
        prompt = str(payload.get("prompt") or "")
        if prompt != COSMOS_INGESTION_PROMPT:
            raise SponsorMalformedResponse("describe requires the exact Cosmos ingestion prompt")
        storage_uri = str(clip.get("storage_uri") or "")
        start_pts_us = int(clip["start_pts_us"])
        end_pts_us = int(clip["end_pts_us"])
        video_id = str(clip["video_id"])
        segment_id = str(clip["segment_id"])
        source_sha256 = str(clip["source_sha256"])
        source = self._source_path(storage_uri)
        media = self._extract_clip(source, start_pts_us, end_pts_us)
        b64 = base64.b64encode(media).decode("ascii")
        model = await self._resolve_model(self._reason_url, self._reason_model_id)
        instruction = (
            f"{COSMOS_INGESTION_PROMPT}\n\n"
            "Return ONLY a JSON object with keys:\n"
            "- summary (string)\n"
            "- rigid_subjects (array of objects with kind in "
            "[LICENSE_PLATE, SIGN, OTHER_RIGID], location, legibility in "
            "[CLEAR, PARTIAL, ILLEGIBLE], confidence 0..1; optional color, "
            "approximate_size, orientation, quoted_text)\n"
            "- actions (array of {description, direction, confidence})\n"
            "- visibility (array of {kind in [CLEAR, OCCLUSION, MOTION_BLUR, "
            "FOCUS_BLUR, COMPRESSION, LOW_RESOLUTION, GLARE, DARKNESS, UNKNOWN], "
            "confidence})\n"
            "- scene_changes (string array)\n"
            "- search_terms (short string array)\n"
            "- uncertainty (string array)\n"
            "- confidence (0..1 number)\n"
            "Omit rigid_subjects that are not plates/signs/other rigid text-bearing objects. "
            "Do not invent identity or legal conclusions."
        )
        body = {
            "model": model,
            "temperature": 0,
            "max_tokens": 1200,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{b64}"},
                        },
                    ],
                }
            ],
        }
        headers = {**_auth_headers(self._token), "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(
                    f"{self._reason_url}/v1/chat/completions", headers=headers, json=body
                )
        except httpx.TimeoutException as exc:
            raise SponsorTimeout("cosmos describe timed out") from exc
        except httpx.HTTPError as exc:
            raise SponsorUnavailable("cosmos describe request failed") from exc
        if response.status_code >= 400:
            raise SponsorUnavailable(f"cosmos describe HTTP {response.status_code}")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise SponsorMalformedResponse("cosmos describe response missing content") from exc
        parsed = self._parse_description_json(str(content))
        raw_bytes = json.dumps(parsed, sort_keys=True).encode("utf-8")
        time_range = {"start_pts_us": start_pts_us, "end_pts_us": end_pts_us}
        description = SegmentDescription.model_validate(
            self._coerce_description(
                parsed,
                segment_id=segment_id,
                video_id=video_id,
                source_sha256=source_sha256,
                time_range=time_range,
                model_id=model,
                raw_output_sha256=sha256_hex(raw_bytes),
            )
        )
        return description.model_dump(mode="json")

    def _parse_description_json(self, content: str) -> dict[str, Any]:
        text = content.strip()
        fence = _JSON_FENCE.search(text)
        if fence:
            text = fence.group(1).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise SponsorMalformedResponse("cosmos describe did not return JSON") from None
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise SponsorMalformedResponse("cosmos describe JSON was invalid") from exc
        if not isinstance(data, dict):
            raise SponsorMalformedResponse("cosmos describe JSON must be an object")
        return data

    def _coerce_description(
        self,
        parsed: Mapping[str, Any],
        *,
        segment_id: str,
        video_id: str,
        source_sha256: str,
        time_range: dict[str, int],
        model_id: str,
        raw_output_sha256: str,
    ) -> dict[str, Any]:
        """Normalize free-form Cosmos JSON into Probity SegmentDescription fields."""

        def clip_str(value: Any, max_len: int, default: str = "") -> str:
            text = str(value if value is not None else default).strip()
            return text[:max_len] if text else default[:max_len]

        def conf(value: Any, default: float = 0.5) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return default
            return max(0.0, min(1.0, number))

        def map_kind(raw: Any) -> str | None:
            text = str(raw or "").strip().upper().replace(" ", "_").replace("-", "_")
            aliases = {
                "LICENSE_PLATE": "LICENSE_PLATE",
                "PLATE": "LICENSE_PLATE",
                "NUMBER_PLATE": "LICENSE_PLATE",
                "SIGN": "SIGN",
                "TRAFFIC_SIGN": "SIGN",
                "STREET_SIGN": "SIGN",
                "OTHER_RIGID": "OTHER_RIGID",
                "OTHER": "OTHER_RIGID",
            }
            if text in aliases:
                return aliases[text]
            lower = str(raw or "").lower()
            if "plate" in lower:
                return "LICENSE_PLATE"
            if "sign" in lower:
                return "SIGN"
            return None

        subjects: list[dict[str, Any]] = []
        for item in parsed.get("rigid_subjects") or []:
            if not isinstance(item, dict):
                continue
            kind = map_kind(item.get("kind") or item.get("type") or item.get("class"))
            if kind is None:
                continue
            legibility = str(item.get("legibility") or "ILLEGIBLE").upper()
            if legibility not in {"CLEAR", "PARTIAL", "ILLEGIBLE"}:
                legibility = "ILLEGIBLE"
            quoted = item.get("quoted_text")
            if quoted is not None and legibility != "CLEAR":
                quoted = None
            subjects.append(
                {
                    "kind": kind,
                    "color": clip_str(item.get("color"), 40) or None,
                    "location": clip_str(
                        item.get("location") or item.get("position") or "in frame", 200, "in frame"
                    ),
                    "approximate_size": clip_str(item.get("approximate_size"), 100) or None,
                    "orientation": clip_str(item.get("orientation"), 100) or None,
                    "legibility": legibility,
                    "quoted_text": clip_str(quoted, 32) or None if quoted is not None else None,
                    "uncertainty": clip_str(item.get("uncertainty"), 300) or None,
                    "has_multiple_clearer_observations": item.get(
                        "has_multiple_clearer_observations"
                    ),
                    "confidence": conf(item.get("confidence"), 0.5),
                    "time_range": time_range,
                }
            )

        actions: list[dict[str, Any]] = []
        for item in parsed.get("actions") or []:
            if not isinstance(item, dict):
                continue
            desc = clip_str(item.get("description") or item.get("action"), 200)
            if not desc:
                continue
            actions.append(
                {
                    "description": desc,
                    "direction": clip_str(item.get("direction"), 60) or None,
                    "confidence": conf(item.get("confidence"), 0.5),
                    "time_range": time_range,
                }
            )

        visibility: list[dict[str, Any]] = []
        for item in parsed.get("visibility") or []:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "UNKNOWN").upper().replace(" ", "_")
            allowed = {
                "CLEAR",
                "OCCLUSION",
                "MOTION_BLUR",
                "FOCUS_BLUR",
                "COMPRESSION",
                "LOW_RESOLUTION",
                "GLARE",
                "DARKNESS",
                "UNKNOWN",
            }
            if kind not in allowed:
                kind = "UNKNOWN"
            visibility.append(
                {
                    "kind": kind,
                    "observable_source": clip_str(item.get("observable_source"), 200) or None,
                    "confidence": conf(item.get("confidence"), 0.5),
                    "time_range": None,
                }
            )
        if not visibility:
            visibility = [
                {
                    "kind": "CLEAR",
                    "observable_source": None,
                    "confidence": 0.5,
                    "time_range": None,
                }
            ]

        terms = [
            clip_str(term, 40)
            for term in (parsed.get("search_terms") or ["video", "segment"])
            if clip_str(term, 40)
        ][:20] or ["video", "segment"]
        uncertainty = [
            clip_str(item, 300)
            for item in (parsed.get("uncertainty") or [])
            if clip_str(item, 300)
        ][:20]
        scene_changes = [
            clip_str(item, 200)
            for item in (parsed.get("scene_changes") or [])
            if clip_str(item, 200)
        ][:20]

        return {
            "segment_id": segment_id,
            "video_id": video_id,
            "source_sha256": source_sha256,
            "time_range": time_range,
            "model_id": model_id,
            "prompt_sha256": COSMOS_INGESTION_PROMPT_SHA256,
            "raw_output_sha256": raw_output_sha256,
            "summary": clip_str(parsed.get("summary"), 1000, "Observable scene content.")
            or "Observable scene content.",
            "rigid_subjects": subjects,
            "actions": actions,
            "visibility": visibility,
            "scene_changes": scene_changes,
            "search_terms": terms,
            "uncertainty": uncertainty,
            "confidence": conf(parsed.get("confidence"), 0.5),
            "mode": InferenceMode.LIVE.value,
        }

    async def _embed_text(self, text: str, ref: str) -> dict[str, Any]:
        model = await self._resolve_model(self._embed_url, self._embed_model_id)
        body = {
            "input": text,
            "model": model,
            "request_type": "query",
            "encoding_format": "float",
        }
        headers = {**_auth_headers(self._token), "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(
                    f"{self._embed_url}/v1/embeddings", headers=headers, json=body
                )
        except httpx.TimeoutException as exc:
            raise SponsorTimeout("cosmos embed timed out") from exc
        except httpx.HTTPError as exc:
            raise SponsorUnavailable("cosmos embed request failed") from exc
        if response.status_code >= 400:
            raise SponsorUnavailable(f"cosmos embed HTTP {response.status_code}")
        try:
            vector = list(response.json()["data"][0]["embedding"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise SponsorMalformedResponse("cosmos embed response missing vector") from exc
        unit = _normalize(vector)
        embedding = Embedding(
            embedding_ref=ref,
            model_id=model,
            dimension=len(unit),
            vector=unit,
            normalized=True,
            input_sha256=sha256_hex(text.encode("utf-8")),
            mode=InferenceMode.LIVE,
        )
        return embedding.model_dump(mode="json")

    async def _embed_segments(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        items = payload.get("items") or []
        if not items:
            raise SponsorMalformedResponse("embed_segments requires items")
        item = items[0]
        summary = str(item.get("summary") or "")
        terms = tuple(str(t) for t in (item.get("search_terms") or ()))
        text = embedding_input_text(summary, terms)
        ref = f"live/{item.get('segment_id', 'segment')}"
        return await self._embed_text(text, ref)

    async def _embed_query(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        query = str(payload.get("query") or "").strip()
        if not query:
            raise SponsorMalformedResponse("embed_query requires query")
        return await self._embed_text(query, "query")
