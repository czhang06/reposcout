"""Exact cosine search over a NumPy matrix, persisted as vectors.npy + chunks.jsonl.

Fine up to a few hundred thousand chunks and needs no database, which makes it the
baseline for tests and for comparing against pgvector later.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reposcout.chunk import Chunk
from reposcout.store import Hit


class MemoryStore:
    def __init__(self, dim: int, path: Path | None = None):
        self.dim = dim
        self.path = path
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self._bm25 = None  # built lazily, reset on add()

    @classmethod
    def load_or_create(cls, path: Path, dim: int) -> "MemoryStore":
        store = cls(dim, path)
        vec_file, meta_file = path / "vectors.npy", path / "chunks.jsonl"
        if vec_file.exists() and meta_file.exists():
            vectors = np.load(vec_file)
            if vectors.shape[1] != dim:
                raise ValueError(
                    f"Index at {path} has dim {vectors.shape[1]} but the embedder has dim {dim}; "
                    "re-ingest with --rebuild."
                )
            store.vectors = vectors
            with meta_file.open(encoding="utf-8") as f:
                store.chunks = [Chunk.from_dict(json.loads(line)) for line in f]
        return store

    def existing_ids(self) -> set[str]:
        return {c.id for c in self.chunks}

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        self.chunks.extend(chunks)
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        self._bm25 = None

    def search(self, query_vec: np.ndarray, k: int, source_types: list[str] | None = None) -> list[Hit]:
        if not self.chunks:
            return []
        scores = self.vectors @ query_vec.astype(np.float32)
        mask = self._mask(source_types)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(self.chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [Hit(self.chunks[i], float(scores[i])) for i in top if np.isfinite(scores[i])]

    def _mask(self, source_types: list[str] | None) -> np.ndarray | None:
        if not source_types:
            return None
        return np.array([c.source_type in source_types for c in self.chunks])

    def keyword_search(self, query: str, k: int, source_types: list[str] | None = None) -> list[Hit]:
        from reposcout.retrieval.bm25 import BM25

        if not self.chunks:
            return []
        if self._bm25 is None:
            self._bm25 = BM25([c.embed_text() for c in self.chunks])
        return [Hit(self.chunks[i], s) for i, s in self._bm25.top_k(query, k, self._mask(source_types))]

    def count(self) -> int:
        return len(self.chunks)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.path / "vectors.npy", self.vectors)
        with (self.path / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
