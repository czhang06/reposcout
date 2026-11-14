"""Retrieval eval: does search surface the files that the real fix changed?

Metrics are file-level, because gold evidence is the set of files a merged fix touched:
- recall@k: share of gold files found among the files of the top-k chunks
- hit@k: share of questions with at least one gold file in the top k
- MRR@10: mean of 1 / rank of the first chunk from a gold file (0 if none in top 10)
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from reposcout.retrieval.retriever import RetrievalConfig, Retriever


@dataclass
class EvalQuestion:
    id: str
    question: str
    gold_files: list[str]
    mentions_symbol: bool = False
    date: str | None = None  # merge date of the fix; PRs from this date on are hidden

    @classmethod
    def load(cls, path: Path) -> list["EvalQuestion"]:
        out = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                out.append(cls(d["id"], d["question"], d["gold_files"], d.get("mentions_symbol", False), d.get("date")))
        return out


def score_ranking(files: list[str], gold: list[str], ks=(1, 5, 10)) -> dict[str, float]:
    """files = file path of each retrieved chunk, in rank order (may repeat)."""
    gold_set = set(gold)
    out: dict[str, float] = {}
    for k in ks:
        found = gold_set & set(files[:k])
        out[f"recall@{k}"] = len(found) / len(gold_set)
        out[f"hit@{k}"] = 1.0 if found else 0.0
    first = next((i for i, f in enumerate(files[:10], 1) if f in gold_set), None)
    out["mrr@10"] = 1.0 / first if first else 0.0
    return out


def _mean(rows: list[dict], key: str) -> float:
    return statistics.fmean(r[key] for r in rows) if rows else 0.0


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def evaluate(retriever: Retriever, questions: list[EvalQuestion], cfg: RetrievalConfig, k: int = 10) -> dict:
    per_q = []
    for q in questions:
        t = time.perf_counter()
        res = retriever.retrieve(q.question, k, replace(cfg, before_date=q.date))
        ms = (time.perf_counter() - t) * 1000
        files = [h.chunk.path for h in res.hits]
        per_q.append({"id": q.id, "mentions_symbol": q.mentions_symbol, "latency_ms": ms,
                      "top_files": files[:5], **score_ranking(files, q.gold_files)})

    def summary(rows: list[dict]) -> dict:
        keys = ["recall@1", "recall@5", "recall@10", "hit@5", "mrr@10"]
        return {"n": len(rows), **{key: round(_mean(rows, key), 4) for key in keys}}

    lat = [r["latency_ms"] for r in per_q]
    return {
        "config": cfg.label,
        "all": summary(per_q),
        "with_symbol": summary([r for r in per_q if r["mentions_symbol"]]),
        "no_symbol": summary([r for r in per_q if not r["mentions_symbol"]]),
        "latency_ms": {"p50": round(_pct(lat, 50), 1), "p95": round(_pct(lat, 95), 1)},
        "per_question": per_q,
    }


def markdown_table(results: list[dict]) -> str:
    head = ("| Retrieval | Recall@5 | Recall@10 | Hit@5 | MRR@10 | Recall@5, names a symbol | "
            "Recall@5, no symbol | p50 ms | p95 ms |")
    rows = [head, "|" + "---|" * 9]
    for r in results:
        a, s, n = r["all"], r["with_symbol"], r["no_symbol"]
        rows.append(
            f"| {r['config']} | {a['recall@5']:.1%} | {a['recall@10']:.1%} | {a['hit@5']:.1%} | {a['mrr@10']:.3f} "
            f"| {s['recall@5']:.1%} (n={s['n']}) | {n['recall@5']:.1%} (n={n['n']}) "
            f"| {r['latency_ms']['p50']:.0f} | {r['latency_ms']['p95']:.0f} |"
        )
    return "\n".join(rows)


def save(results: list[dict], out_dir: Path, meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out_dir / f"retrieval-{stamp}.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    md = out_dir / f"retrieval-{stamp}.md"
    lines = [f"# Retrieval eval {stamp}", ""] + [f"- **{k}**: {v}" for k, v in meta.items()] + ["", markdown_table(results), ""]
    md.write_text("\n".join(lines), encoding="utf-8")
    return md


__all__ = ["EvalQuestion", "evaluate", "markdown_table", "save", "score_ranking", "asdict"]
