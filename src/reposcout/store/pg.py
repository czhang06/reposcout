"""Postgres + pgvector store. Run `docker compose up -d db` first.

The `tsv` column is a generated full-text index; week 2 adds hybrid search over it.
"""

from __future__ import annotations

import json

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column, Computed, Index, Integer, MetaData, String, Table, Text, create_engine, func, select, text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, insert

from reposcout.chunk import Chunk
from reposcout.store import Hit


def _table(metadata: MetaData, dim: int) -> Table:
    return Table(
        "chunks",
        metadata,
        Column("id", String(20), primary_key=True),
        Column("source_type", String(10), nullable=False),
        Column("path", Text, nullable=False),
        Column("start_line", Integer, nullable=False),
        Column("end_line", Integer, nullable=False),
        Column("symbol", Text),
        Column("commit", String(40)),
        Column("content_hash", String(64), nullable=False),
        Column("text", Text, nullable=False),
        Column("extra", JSONB, nullable=False, server_default="{}"),
        Column("embedding", Vector(dim), nullable=False),
        Column("tsv", TSVECTOR, Computed("to_tsvector('english', coalesce(symbol,'') || ' ' || text)", persisted=True)),
        Index("chunks_embedding_hnsw", "embedding", postgresql_using="hnsw",
              postgresql_ops={"embedding": "vector_cosine_ops"}),
        Index("chunks_tsv_gin", "tsv", postgresql_using="gin"),
        Index("chunks_content_hash", "content_hash"),
    )


class PgVectorStore:
    def __init__(self, url: str, dim: int):
        self.engine = create_engine(url)
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        self.metadata = MetaData()
        self.t = _table(self.metadata, dim)
        self.metadata.create_all(self.engine)

    def existing_ids(self) -> set[str]:
        with self.engine.connect() as conn:
            return set(conn.execute(select(self.t.c.id)).scalars())

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        rows = [
            {
                "id": c.id, "source_type": c.source_type, "path": c.path, "start_line": c.start_line,
                "end_line": c.end_line, "symbol": c.symbol, "commit": c.commit,
                "content_hash": c.content_hash, "text": c.text, "extra": c.extra,
                "embedding": v.tolist(),
            }
            for c, v in zip(chunks, vectors)
        ]
        stmt = insert(self.t).on_conflict_do_nothing(index_elements=["id"])
        with self.engine.begin() as conn:
            for i in range(0, len(rows), 500):
                conn.execute(stmt, rows[i : i + 500])

    def search(self, query_vec: np.ndarray, k: int, source_types: list[str] | None = None) -> list[Hit]:
        distance = self.t.c.embedding.cosine_distance(query_vec.tolist())
        q = select(self.t, distance.label("distance")).order_by(distance).limit(k)
        if source_types:
            q = q.where(self.t.c.source_type.in_(source_types))
        with self.engine.connect() as conn:
            rows = conn.execute(q).mappings().all()
        return [Hit(self._row_chunk(r), 1.0 - float(r["distance"])) for r in rows]

    def keyword_search(self, query: str, k: int, source_types: list[str] | None = None) -> list[Hit]:
        """Postgres full-text search, OR-ing the query terms and ranking with ts_rank_cd.

        Uses the same code-aware tokenizer as the in-memory BM25 so identifiers like
        clear_task_instances also match their parts.
        """
        from reposcout.retrieval.bm25 import tokenize

        terms = sorted({t for t in tokenize(query) if t.isidentifier()})
        if not terms:
            return []
        tsq = func.to_tsquery("english", " | ".join(terms))
        rank = func.ts_rank_cd(self.t.c.tsv, tsq)
        q = select(self.t, rank.label("rank")).where(self.t.c.tsv.op("@@")(tsq)).order_by(rank.desc()).limit(k)
        if source_types:
            q = q.where(self.t.c.source_type.in_(source_types))
        with self.engine.connect() as conn:
            rows = conn.execute(q).mappings().all()
        return [Hit(self._row_chunk(r), float(r["rank"])) for r in rows]

    @staticmethod
    def _row_chunk(r) -> Chunk:
        extra = r["extra"] if isinstance(r["extra"], dict) else json.loads(r["extra"])
        return Chunk(r["source_type"], r["path"], r["start_line"], r["end_line"], r["text"], r["symbol"], r["commit"], extra)

    def count(self) -> int:
        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(self.t)).scalar_one()

    def save(self) -> None:  # writes are committed in add()
        pass
