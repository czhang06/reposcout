"""Hand review of the eval set: a random sample, a small web page, and saved labels.

For each sampled question the reviewer answers two separate things:
1. Is the question itself fair? keep / reword / fix_gold / drop, with a note and an
   optional rewording or corrected gold files.
2. How good is the agent's answer? A 0/1/2 grade made without seeing the judge's
   grade, so judge agreement can be measured honestly.

Labels are saved to a JSON file in the repo (evals/spot_check_v1.json) as the
reviewer works, so the review survives restarts and is versioned with the eval set.
"""

from __future__ import annotations

import json
import random
import subprocess
import threading
from pathlib import Path

from pydantic import BaseModel

VERDICTS = ("keep", "reword", "fix_gold", "drop")
MAX_DIFF_LINES = 300


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True).stdout


def build_sample(dataset: Path, answers_file: Path, repo: Path, n: int = 20, seed: int = 7) -> list[dict]:
    """A seeded random sample (not the first n: those are the newest commits)."""
    questions = [json.loads(l) for l in dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
    answers = {q["id"]: q for q in json.loads(answers_file.read_text(encoding="utf-8"))["result"]["per_question"]}
    picked = random.Random(seed).sample(questions, min(n, len(questions)))
    items = []
    for q in sorted(picked, key=lambda q: q["id"]):
        diff = _git(repo, "show", "--format=", q["commit"], "--", *q["gold_files"]).splitlines()
        a = answers.get(q["id"], {})
        items.append({
            "id": q["id"],
            "question": q["question"],
            "title": q["title"],
            "commit": q["commit"],
            "pr": q.get("pr"),
            "date": q["date"],
            "gold_files": q["gold_files"],
            "body": _git(repo, "show", "-s", "--format=%b", q["commit"]).strip()[:3000],
            "stat": _git(repo, "show", "--stat", "--format=", q["commit"]).strip(),
            "diff": "\n".join(diff[:MAX_DIFF_LINES]) + (f"\n... ({len(diff) - MAX_DIFF_LINES} more lines)" if len(diff) > MAX_DIFF_LINES else ""),
            "agent_answer": a.get("answer", ""),
            # Kept out of the page; used only for the agreement summary.
            "judge_correctness": a.get("correctness"),
        })
    return items


class LabelStore:
    """Labels keyed by question id, written to disk on every change."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.labels: dict[str, dict] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def put(self, qid: str, label: dict) -> dict:
        verdict = label.get("verdict")
        grade = label.get("grade")
        if verdict is not None and verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        if grade is not None and grade not in (0, 1, 2):
            raise ValueError("grade must be 0, 1 or 2")
        clean = {
            "verdict": verdict,
            "grade": grade,
            "note": str(label.get("note") or "")[:2000],
            "rewrite": str(label.get("rewrite") or "")[:500],
            "gold_files": [str(f) for f in (label.get("gold_files") or [])][:10],
        }
        with self._lock:
            self.labels[qid] = clean
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.labels, indent=1, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)
        return clean


def summarize(items: list[dict], labels: dict[str, dict]) -> dict:
    done = [i for i in items if labels.get(i["id"], {}).get("verdict")]
    counts = {v: sum(labels[i["id"]]["verdict"] == v for i in done) for v in VERDICTS}
    graded = [i for i in items if labels.get(i["id"], {}).get("grade") is not None and i["judge_correctness"] is not None]
    agree = sum(labels[i["id"]]["grade"] == i["judge_correctness"] for i in graded)
    within_one = sum(abs(labels[i["id"]]["grade"] - i["judge_correctness"]) <= 1 for i in graded)
    return {
        "sampled": len(items),
        "reviewed": len(done),
        "verdicts": counts,
        "graded": len(graded),
        "judge_exact_agreement": f"{agree}/{len(graded)}" if graded else "n/a",
        "judge_within_one": f"{within_one}/{len(graded)}" if graded else "n/a",
        "changes": [
            {"id": i["id"], **labels[i["id"]]} for i in done if labels[i["id"]]["verdict"] != "keep"
        ],
        "disagreements": [
            {"id": i["id"], "yours": labels[i["id"]]["grade"], "judge": i["judge_correctness"]}
            for i in graded if labels[i["id"]]["grade"] != i["judge_correctness"]
        ],
    }


class Label(BaseModel):
    # Module level: with postponed annotations, FastAPI cannot resolve a model
    # defined inside create_app and would treat the body as query parameters.
    verdict: str | None = None
    grade: int | None = None
    note: str = ""
    rewrite: str = ""
    gold_files: list[str] = []


def create_app(items: list[dict], store: LabelStore):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="RepoScout eval review")
    public = [{k: v for k, v in i.items() if k != "judge_correctness"} for i in items]
    by_id = {i["id"] for i in items}
    page = (Path(__file__).parent / "review.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page

    @app.get("/api/items")
    def get_items() -> dict:
        return {"items": public, "labels": store.labels}

    @app.put("/api/labels/{qid}")
    def put_label(qid: str, label: Label) -> dict:
        if qid not in by_id:
            raise HTTPException(404, "not in the sample")
        try:
            return store.put(qid, label.model_dump())
        except ValueError as e:
            raise HTTPException(422, str(e)) from None

    @app.get("/api/summary")
    def get_summary() -> dict:
        return summarize(items, store.labels)

    return app
