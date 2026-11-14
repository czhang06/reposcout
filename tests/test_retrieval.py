import pytest

from reposcout.chunk import Chunk
from reposcout.embed import HashingEmbedder
from reposcout.ingest.pipeline import index_chunks
from reposcout.retrieval.bm25 import BM25, tokenize
from reposcout.retrieval.retriever import RetrievalConfig, Retriever, rrf
from reposcout.store.memory import MemoryStore


def test_tokenize_splits_identifiers():
    toks = tokenize("DagRun.clear_task_instances(session)")
    assert {"dagrun", "dag", "run", "clear_task_instances", "clear", "task", "instanc", "session"} <= set(toks)
    assert tokenize("clearing tasks") == tokenize("cleared task")
    assert "the" not in tokenize("the task")


def test_bm25_prefers_rare_matching_terms():
    docs = ["task scheduler pool executor trigger", "xcom push value", "task xcom"]
    bm = BM25(docs)
    assert bm.top_k("xcom push", 3)[0][0] == 1
    assert bm.top_k("nothing matches here", 3) == []
    # same term frequency: the shorter document ranks higher (length normalisation)
    assert bm.top_k("task", 3)[0][0] == 2


def _c(name: str) -> Chunk:
    return Chunk("code", f"{name}.py", 1, 2, name)


def test_rrf_rewards_agreement():
    a, b, c = _c("a"), _c("b"), _c("c")
    fused = [x.path for x, _ in rrf([[a, b, c], [b, c, a]])]
    assert fused[0] == "b.py"  # rank 2 + rank 1 beats rank 1 + rank 3
    assert set(fused) == {"a.py", "b.py", "c.py"}



def _store():
    emb = HashingEmbedder()
    store = MemoryStore(emb.dim)
    index_chunks(
        [
            Chunk("code", "models/taskinstance.py", 1, 9, "def clear_task_instances(tis): reset state", "clear_task_instances"),
            Chunk("doc", "docs/clearing.md", 1, 3, "Clearing a task resets it so the scheduler runs it again", "Clearing"),
            Chunk("code", "models/xcom.py", 1, 5, "def xcom_push(key, value): store", "xcom_push"),
        ],
        emb,
        store,
    )
    return emb, store


def test_keyword_search_finds_identifier_parts():
    _, store = _store()
    hits = store.keyword_search("clear task instances", 3)
    assert hits[0].chunk.path == "models/taskinstance.py"
    assert store.keyword_search("clear", 3, source_types=["doc"])[0].chunk.source_type == "doc"


@pytest.mark.parametrize(
    "cfg",
    [RetrievalConfig(keyword=False), RetrievalConfig(vector=False), RetrievalConfig()],
    ids=lambda c: c.label,
)
def test_retriever_modes(cfg):
    emb, store = _store()
    res = Retriever(emb, store).retrieve("how does clear_task_instances reset state", 2, cfg)
    assert len(res.hits) <= 2
    assert res.hits[0].chunk.path == "models/taskinstance.py"


def test_retriever_rerank_reorders():
    emb, store = _store()

    class LastFirst:
        def rerank(self, query, chunks):
            return [(c, float(i)) for i, c in enumerate(chunks)][::-1]

    plain = Retriever(emb, store).retrieve("clear task", 3, RetrievalConfig())
    reranked = Retriever(emb, store, LastFirst()).retrieve("clear task", 3, RetrievalConfig(rerank=True))
    assert [h.chunk.id for h in reranked.hits] == [h.chunk.id for h in plain.hits][::-1]
    assert "rerank_ms" in reranked.timings_ms


def test_retriever_requires_a_mode():
    emb, store = _store()
    with pytest.raises(ValueError):
        Retriever(emb, store).retrieve("q", 1, RetrievalConfig(vector=False, keyword=False))


def test_prs_hidden_from_questions_dated_before_their_merge():
    emb = HashingEmbedder()
    store = MemoryStore(emb.dim)
    old = Chunk("pr", "pr/1", 1, 1, "Fix clearing of task instances", "Fix clearing", "a", {"date": "2025-01-01", "pr": 1})
    new = Chunk("pr", "pr/2", 1, 1, "Fix clearing of task instances again", "Fix clearing", "b", {"date": "2026-01-01", "pr": 2})
    index_chunks([old, new], emb, store)
    r = Retriever(emb, store)
    assert {h.chunk.path for h in r.retrieve("clearing task instances", 5, RetrievalConfig()).hits} == {"pr/1", "pr/2"}
    seen = {h.chunk.path for h in r.retrieve("clearing task instances", 5, RetrievalConfig(before_date="2026-01-01")).hits}
    assert seen == {"pr/1"}  # the PR merged on the question date is hidden too
