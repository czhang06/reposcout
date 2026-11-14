"""Vector stores. MemoryStore needs nothing installed; PgVectorStore needs Postgres + pgvector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from reposcout.chunk import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def existing_ids(self) -> set[str]: ...
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, query_vec: np.ndarray, k: int, source_types: list[str] | None = None) -> list[Hit]: ...
    def keyword_search(self, query: str, k: int, source_types: list[str] | None = None) -> list[Hit]: ...
    def count(self) -> int: ...
    def save(self) -> None: ...


def get_store(settings, dim: int) -> VectorStore:
    if settings.store == "pg":
        from reposcout.store.pg import PgVectorStore

        return PgVectorStore(settings.database_url, dim)
    from reposcout.store.memory import MemoryStore

    return MemoryStore.load_or_create(settings.index_dir, dim)
