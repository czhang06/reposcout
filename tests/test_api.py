from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from reposcout import api
from reposcout.config import get_settings
from reposcout.embed import HashingEmbedder
from reposcout.ingest.pipeline import ingest_repo
from reposcout.store.memory import MemoryStore

FIXTURE = Path(__file__).parent / "fixtures" / "minirepo"


def test_ask_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOSCOUT_EMBED_MODEL", "hashing")
    monkeypatch.setenv("REPOSCOUT_INDEX_DIR", str(tmp_path))
    get_settings.cache_clear()
    ingest_repo(FIXTURE, HashingEmbedder(), MemoryStore(HashingEmbedder().dim, tmp_path))
    monkeypatch.setattr(
        api, "get_chat_model", lambda spec, max_tokens: FakeListChatModel(responses=["Use DAG.add_task [1]."])
    )
    try:
        with TestClient(api.app) as client:
            assert client.get("/health").json()["chunks"] > 0
            r = client.post("/ask", json={"question": "How do I add a task to a DAG?", "k": 3})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["answer"] == "Use DAG.add_task [1]."
            assert len(body["citations"]) == 1 and body["invalid_citations"] == []
            assert client.post("/ask", json={"question": "x"}).status_code == 422
    finally:
        get_settings.cache_clear()
