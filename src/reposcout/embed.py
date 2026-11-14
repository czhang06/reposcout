"""Embedding models behind one small interface, so evals can swap them."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local embeddings with sentence-transformers. Vectors are L2-normalised."""

    # BGE models expect this prefix on queries (not on documents).
    _QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
                     "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: "}

    def __init__(self, model_name: str, batch_size: int = 64, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 500
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        prefix = self._QUERY_PREFIX.get(self.name, "")
        return self.embed_documents([prefix + text])[0]


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. For tests and offline smoke runs only."""

    def __init__(self, dim: int = 256):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9_]+", text.lower()):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._one(text)


def get_embedder(name: str, batch_size: int = 64) -> Embedder:
    if name.startswith("hashing"):
        return HashingEmbedder()
    return SentenceTransformerEmbedder(name, batch_size=batch_size)
