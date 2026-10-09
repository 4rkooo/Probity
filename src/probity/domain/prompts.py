"""Exact Cosmos ingestion prompt (section 8). Never interpolate user text into this string."""

from __future__ import annotations

from probity.domain.ids import sha256_hex

COSMOS_INGESTION_PROMPT = """You are creating a factual search index for a research/demo video-evidence tool.
Describe only directly observable content in this clip. Do not infer identity, intent,
motive, guilt, legality, relationships, or events outside the clip.

Return strict JSON matching the supplied schema. Include:
1. Rigid text-bearing subjects such as license plates and signs: type, color, location,
   approximate size, orientation, and whether text is clearly legible, partly legible,
   or illegible. Quote text only when every quoted character is visibly supported;
   otherwise use null and explain the uncertainty without guessing.
2. Objective actions and motion: subject/camera direction and visible change over time.
3. Visibility limits: occlusion or obstruction and its observable source; blur caused by
   motion, focus, compression, low resolution, glare, darkness, or unknown cause.
4. Lighting and appearance changes, camera motion, subject motion, scale change, pose,
   perspective, and whether a planar rigid subject has multiple clearer observations.
5. Timestamp ranges in microseconds relative to the source clip and concise search terms.

Use calibrated confidence values from 0 to 1. If evidence is ambiguous, say so. Never
complete a word, plate, object, or action from context. Never give legal conclusions."""

COSMOS_INGESTION_PROMPT_SHA256 = sha256_hex(COSMOS_INGESTION_PROMPT.encode("utf-8"))
