import json

import pytest
from fastapi.testclient import TestClient

from reposcout.evals.review import LabelStore, create_app, summarize

ITEMS = [
    {"id": f"v1-00{i}", "question": f"Q{i}?", "title": "Fix x (#1)", "commit": "abc", "pr": 1, "date": "2026-01-01",
     "gold_files": ["a.py"], "body": "", "stat": "a.py | 1 +", "diff": "+x", "agent_answer": "ans",
     "judge_correctness": g}
    for i, g in enumerate([2, 1, 0])
]


def test_label_store_validates_and_persists(tmp_path):
    path = tmp_path / "labels.json"
    store = LabelStore(path)
    store.put("v1-000", {"verdict": "keep", "grade": 2, "note": "ok"})
    with pytest.raises(ValueError):
        store.put("v1-001", {"verdict": "maybe"})
    with pytest.raises(ValueError):
        store.put("v1-001", {"grade": 3})
    assert LabelStore(path).labels["v1-000"]["grade"] == 2  # reloads from disk


def test_summary_counts_and_agreement(tmp_path):
    store = LabelStore(tmp_path / "l.json")
    store.put("v1-000", {"verdict": "keep", "grade": 2})
    store.put("v1-001", {"verdict": "reword", "grade": 2, "rewrite": "Better Q?"})
    store.put("v1-002", {"verdict": "drop"})
    s = summarize(ITEMS, store.labels)
    assert s["reviewed"] == 3 and s["verdicts"] == {"keep": 1, "reword": 1, "fix_gold": 0, "drop": 1}
    assert s["judge_exact_agreement"] == "1/2" and s["judge_within_one"] == "2/2"
    assert [c["id"] for c in s["changes"]] == ["v1-001", "v1-002"]
    assert s["disagreements"] == [{"id": "v1-001", "yours": 2, "judge": 1}]


def test_app_hides_judge_grades_and_saves(tmp_path):
    path = tmp_path / "labels.json"
    client = TestClient(create_app(ITEMS, LabelStore(path)))
    assert "Eval Spot-Check" in client.get("/").text
    data = client.get("/api/items").json()
    assert len(data["items"]) == 3 and all("judge_correctness" not in i for i in data["items"])
    assert client.put("/api/labels/v1-000", json={"verdict": "keep", "grade": 1}).status_code == 200
    assert json.loads(path.read_text())["v1-000"]["grade"] == 1
    assert client.put("/api/labels/nope", json={"verdict": "keep"}).status_code == 404
    assert client.put("/api/labels/v1-000", json={"verdict": "bad"}).status_code == 422
