"""Chunk -> skip already-indexed -> embed -> store, with timing for the README numbers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from reposcout.chunk import Chunk
from reposcout.embed import Embedder
from reposcout.ingest.repo import chunk_repo
from reposcout.store import VectorStore


@dataclass
class IngestStats:
    chunks_total: int
    chunks_new: int
    chunks_skipped: int
    chunk_seconds: float
    embed_seconds: float


def index_chunks(chunks: list[Chunk], embedder: Embedder, store: VectorStore, batch: int = 256) -> tuple[int, float]:
    """Embed and store chunks the store does not have yet. Returns (new_count, embed_seconds)."""
    seen = store.existing_ids()
    new, ids = [], set()
    for c in chunks:
        if c.id not in seen and c.id not in ids:
            new.append(c)
            ids.add(c.id)
    t0 = time.perf_counter()
    for i in range(0, len(new), batch):
        part = new[i : i + batch]
        store.add(part, embedder.embed_documents([c.embed_text() for c in part]))
    store.save()
    return len(new), time.perf_counter() - t0


def ingest_repo(root: Path, embedder: Embedder, store: VectorStore, include: list[str] | None = None) -> IngestStats:
    t0 = time.perf_counter()
    chunks = chunk_repo(root, include)
    chunk_seconds = time.perf_counter() - t0
    new, embed_seconds = index_chunks(chunks, embedder, store)
    return IngestStats(len(chunks), new, len(chunks) - new, chunk_seconds, embed_seconds)
