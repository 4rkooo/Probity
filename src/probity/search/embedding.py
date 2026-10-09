"""Deterministic hashed unigram+bigram embeddings used by fixture Cosmos and tests."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from probity.domain.ids import sha256_hex

EMBED_DIM = 64
TOKEN_RE = re.compile(r"[a-z0-9]+")
_HASH_PREFIX = "probity-fixture-emb:"


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(text.lower()))


def embedding_input_text(summary: str, search_terms: tuple[str, ...] = ()) -> str:
    """Text actually hashed into a fixture embedding (summary plus search terms)."""
    extra = " ".join(search_terms)
    return f"{summary} {extra}".strip() if extra else summary


def input_sha256(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))


def fixture_text_embedding(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """L2-normalized float32 vector from hashed unigrams and bigrams.

    Feature-hashing is deterministic and does not call any sponsor SDK.
    """
    if dim <= 0:
        raise ValueError("dim must be positive")
    tokens = tokenize(text)
    grams: list[str] = list(tokens)
    grams.extend(f"{tokens[i]} {tokens[i + 1]}" for i in range(max(0, len(tokens) - 1)))
    vec = np.zeros(dim, dtype=np.float32)
    for gram in grams:
        digest = hashlib.sha256(f"{_HASH_PREFIX}{gram}".encode()).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = np.float32(1.0) if digest[4] < 128 else np.float32(-1.0)
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        vec[0] = np.float32(1.0)
    else:
        vec /= np.float32(norm)
    return vec.astype(np.float32, copy=False)


def as_unit_tuple(vector: np.ndarray) -> tuple[float, ...]:
    return tuple(float(x) for x in np.asarray(vector, dtype=np.float32).tolist())
