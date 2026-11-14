from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from reposcout.chunk import Chunk
from reposcout.embed import HashingEmbedder
from reposcout.ingest.pipeline import index_chunks, ingest_repo
from reposcout.rag import answer_question, parse_citations
from reposcout.retrieval.retriever import Retriever
from reposcout.store.memory import MemoryStore

FIXTURE = Path(__file__).parent / "fixtures" / "minirepo"


def _chunks():
    return [
        Chunk("code", "models/taskinstance.py", 10, 20, "def clear_task_instances(tis): load notes eagerly", "clear_task_instances", "sha1"),
        Chunk("doc", "docs/scheduler.md", 1, 5, "The scheduler parses DAG files and queues tasks", "Scheduler", "sha1"),
        Chunk("code", "utils/xcom.py", 3, 9, "def push_xcom(key, value): store a value", "push_xcom", "sha1"),
    ]


def test_memory_store_search_and_persistence(tmp_path):
    emb = HashingEmbedder()
    store = MemoryStore(emb.dim, tmp_path)
    new, _ = index_chunks(_chunks(), emb, store)
    assert new == 3
    hits = store.search(emb.embed_query("how does the scheduler queue tasks"), 2)
    assert hits[0].chunk.path == "docs/scheduler.md"
    assert store.search(emb.embed_query("scheduler"), 3, source_types=["code"])[0].chunk.source_type == "code"

    reloaded = MemoryStore.load_or_create(tmp_path, emb.dim)
    assert reloaded.count() == 3
    # re-indexing the same chunks embeds nothing new
    assert index_chunks(_chunks(), emb, reloaded)[0] == 0


def test_parse_citations_flags_out_of_range():
    emb = HashingEmbedder()
    store = MemoryStore(emb.dim)
    index_chunks(_chunks(), emb, store)
    hits = store.search(emb.embed_query("xcom"), 2)
    cites, invalid = parse_citations("A [1] and B [2] and C [7].", hits, "apache/airflow")
    assert [c.n for c in cites] == [1, 2] and invalid == [7]
    assert cites[0].url.startswith("https://github.com/apache/airflow/blob/sha1/")


def test_answer_question_with_fake_llm():
    emb = HashingEmbedder()
    store = MemoryStore(emb.dim)
    index_chunks(_chunks(), emb, store)
    llm = FakeListChatModel(responses=["Notes are loaded eagerly in clear_task_instances [1]."])
    a = answer_question("How are notes loaded when clearing task instances?", Retriever(emb, store), llm, "claude-haiku-5-5", k=2)
    assert a.citations[0].location == "models/taskinstance.py:10-20"
    assert a.invalid_citations == []


def test_ingest_fixture_repo(tmp_path):
    emb = HashingEmbedder()
    store = MemoryStore(emb.dim, tmp_path)
    st = ingest_repo(FIXTURE, emb, store)
    assert st.chunks_new == st.chunks_total > 0
    paths = {c.path for c in store.chunks}
    assert {"pkg/dag.py", "docs/intro.md"} <= paths
    again = ingest_repo(FIXTURE, emb, MemoryStore.load_or_create(tmp_path, emb.dim))
    assert again.chunks_new == 0 and again.chunks_skipped == st.chunks_total
