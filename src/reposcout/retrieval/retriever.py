"""Retrieval pipeline: vector and/or keyword search -> RRF fusion -> optional rerank.

Every stage can be turned off, so the eval can measure what each one adds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from reposcout.chunk import Chunk
from reposcout.embed import Embedder
from reposcout.store import Hit, VectorStore

RRF_K = 60  # standard constant from Cormack et al. (2009)


def rrf(rankings: list[list[Chunk]], k: int = RRF_K) -> list[tuple[Chunk, float]]:
    """Reciprocal rank fusion: score = sum over rankings of 1 / (k + rank).

    Uses only ranks, so it needs no tuning to combine cosine and BM25 scores,
    which are on different scales.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, 1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            by_id[chunk.id] = chunk
    return sorted(((by_id[i], s) for i, s in scores.items()), key=lambda x: -x[1])


class CrossEncoderReranker:
    """Scores (query, chunk) pairs jointly. Slower than embeddings but more accurate,
    so it only reorders the top candidates."""

    def __init__(self, model_name: str, device: str | None = None, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self._model = CrossEncoder(model_name, device=device, max_length=max_length)

    def rerank(self, query: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        scores = self._model.predict([(query, c.embed_text()) for c in chunks], batch_size=32)
        return sorted(zip(chunks, (float(s) for s in scores)), key=lambda x: -x[1])


@dataclass
class RetrievalConfig:
    vector: bool = True
    keyword: bool = True
    rerank: bool = False
    candidates: int = 30  # per retriever, before fusion / rerank
    source_types: list[str] | None = None
    # Hide PRs merged on or after this date (YYYY-MM-DD). Evals set it per question.
    before_date: str | None = None

    @property
    def label(self) -> str:
        parts = [p for p, on in (("vector", self.vector), ("bm25", self.keyword)) if on]
        label = " + ".join(parts) + (" + rerank" if self.rerank else "")
        if self.source_types:
            label += f" [{', '.join(self.source_types)}]"
        return label


@dataclass
class RetrievalResult:
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, reranker: CrossEncoderReranker | None = None):
        self.embedder = embedder
        self.store = store
        self.reranker = reranker

    def retrieve(self, query: str, k: int, cfg: RetrievalConfig) -> RetrievalResult:
        if not (cfg.vector or cfg.keyword):
            raise ValueError("enable at least one of vector or keyword search")
        timings: dict[str, float] = {}
        rankings: list[list[Chunk]] = []
        n = max(cfg.candidates, k)

        def visible(c: Chunk) -> bool:
            return not (cfg.before_date and c.source_type == "pr" and c.extra.get("date", "") >= cfg.before_date)

        if cfg.vector:
            t = time.perf_counter()
            hits = self.store.search(self.embedder.embed_query(query), n, cfg.source_types)
            rankings.append([h.chunk for h in hits if visible(h.chunk)])
            timings["vector_ms"] = (time.perf_counter() - t) * 1000
        if cfg.keyword:
            t = time.perf_counter()
            hits = self.store.keyword_search(query, n, cfg.source_types)
            rankings.append([h.chunk for h in hits if visible(h.chunk)])
            timings["keyword_ms"] = (time.perf_counter() - t) * 1000

        fused = rrf(rankings) if len(rankings) > 1 else [(c, 1.0 / (RRF_K + r)) for r, c in enumerate(rankings[0], 1)]

        if cfg.rerank:
            if self.reranker is None:
                raise ValueError("rerank requested but no reranker configured")
            t = time.perf_counter()
            fused = self.reranker.rerank(query, [c for c, _ in fused[:n]])
            timings["rerank_ms"] = (time.perf_counter() - t) * 1000

        return RetrievalResult([Hit(c, s) for c, s in fused[:k]], timings)


def build_retriever(settings, embedder: Embedder, store: VectorStore) -> tuple[Retriever, RetrievalConfig]:
    """Retriever and config from Settings, shared by the CLI and the API."""
    reranker = CrossEncoderReranker(settings.rerank_model) if settings.rerank_model else None
    cfg = RetrievalConfig(
        keyword=settings.keyword_search, rerank=reranker is not None, candidates=settings.rerank_candidates
    )
    return Retriever(embedder, store, reranker), cfg
