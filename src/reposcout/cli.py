"""Command line: `reposcout ingest <repo>`, `reposcout search "<q>"`, `reposcout ask "<q>"`."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import typer

# Windows consoles default to cp1252; Airflow docs contain emoji and other non-ASCII text.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from reposcout.config import get_settings
from reposcout.embed import get_embedder
from reposcout.store import get_store

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _setup():
    s = get_settings()
    embedder = get_embedder(s.embed_model, s.embed_batch_size)
    return s, embedder, get_store(s, embedder.dim)


@app.command()
def ingest(
    repo: Path = typer.Argument(..., exists=True, file_okay=False, help="Path to a git checkout"),
    include: list[str] = typer.Option(None, "--include", "-i", help="Subpaths to index (repeatable)"),
    rebuild: bool = typer.Option(False, help="Delete the local index first"),
):
    """Chunk, embed and index a repository."""
    from reposcout.ingest.pipeline import ingest_repo

    s = get_settings()
    if rebuild and s.store == "memory" and s.index_dir.exists():
        shutil.rmtree(s.index_dir)
    s, embedder, store = _setup()
    st = ingest_repo(repo, embedder, store, include or None)
    typer.echo(
        f"{st.chunks_total} chunks ({st.chunks_new} new, {st.chunks_skipped} already indexed); "
        f"chunking {st.chunk_seconds:.1f}s, embedding {st.embed_seconds:.1f}s; index size {store.count()}"
    )


@app.command("ingest-prs")
def ingest_prs(
    repo: Path = typer.Argument(..., exists=True, file_okay=False),
    since: str = typer.Option("2025-06-01", help="Only PRs merged on or after this date"),
    path: list[str] = typer.Option(None, "--path", "-p", help="Only PRs touching these paths (repeatable)"),
    rev: str = typer.Option("origin/main"),
):
    """Index merged PRs (title, description, files changed) from git history."""
    from reposcout.ingest.pipeline import index_chunks
    from reposcout.ingest.prs import read_prs

    _, embedder, store = _setup()
    chunks = read_prs(repo, since, path or None, rev)
    new, secs = index_chunks(chunks, embedder, store)
    typer.echo(f"{len(chunks)} PRs ({new} new), embedding {secs:.1f}s; index size {store.count()}")


@app.command()
def search(question: str, k: int = 5):
    """Show the top-k retrieved chunks without calling an LLM."""
    from reposcout.retrieval.retriever import build_retriever

    s, embedder, store = _setup()
    retriever, cfg = build_retriever(s, embedder, store)
    for i, h in enumerate(retriever.retrieve(question, k, cfg).hits, 1):
        typer.echo(f"[{i}] {h.score:.3f}  {h.chunk.citation}  {h.chunk.symbol or ''}")


@app.command()
def ask(question: str, k: int = typer.Option(None)):
    """Answer a question with citations."""
    from reposcout.llm import get_chat_model, model_name
    from reposcout.rag import answer_question

    from reposcout.retrieval.retriever import build_retriever

    s, embedder, store = _setup()
    retriever, cfg = build_retriever(s, embedder, store)
    llm = get_chat_model(s.chat_model, s.max_answer_tokens)
    a = answer_question(question, retriever, llm, model_name(s.chat_model), k or s.top_k, s.github_repo, cfg)
    typer.echo(a.answer)
    typer.echo("\nSources:")
    for c in a.citations:
        typer.echo(f"  [{c.n}] {c.location}  {c.url or ''}")
    if a.invalid_citations:
        typer.echo(f"  (invalid citation markers: {a.invalid_citations})")
    typer.echo(
        f"\n{a.input_tokens} in / {a.output_tokens} out tokens, ${a.cost_usd:.5f}, "
        f"retrieve {a.retrieve_ms:.0f} ms, generate {a.generate_ms:.0f} ms"
    )


@app.command()
def agent(
    question: str,
    repo: Path = typer.Option(Path("data/airflow"), exists=True, file_okay=False, help="Indexed checkout, for read_file"),
    trace: bool = typer.Option(False, help="Print each step and save the trace to results/traces"),
):
    """Answer with the LangGraph agent: it searches and reads the code, then cites verified sources."""
    from reposcout.agent.run import ToolboxFactory, make_agent, save_trace

    s, embedder, store = _setup()
    r = make_agent(s).run(question, ToolboxFactory(s, embedder, store, repo)())
    if trace:
        for e in r.trace:
            detail = {k: v for k, v in e.items() if k not in ("t_ms", "node")}
            typer.echo(f"  {e['t_ms']:>8.0f} ms  {e['node']:<8} {detail}")
        typer.echo(f"  trace saved to {save_trace(r, Path('results/traces'))}\n")
    typer.echo(r.answer)
    typer.echo("\nSources:")
    for c in r.citations:
        typer.echo(f"  [{c['sid']}] {c['location']}  {c['url'] or ''}")
    if r.issues:
        typer.echo(f"\nUnresolved citation issues: {r.issues}")
    typer.echo(
        f"\n{r.rounds} tool rounds, {r.tool_calls} tool calls ({', '.join(r.tools_used) or 'none'}); "
        f"{r.input_tokens} in / {r.output_tokens} out tokens, ${r.cost_usd:.5f}, {r.latency_ms / 1000:.1f} s; "
        f"stop: {r.stop_reason}"
    )


@app.command("eval-retrieval")
def eval_retrieval(
    dataset: Path = typer.Option(Path("evals/airflow_v1.jsonl"), exists=True),
    reranker: list[str] = typer.Option(
        ["cross-encoder/ms-marco-MiniLM-L-6-v2"], "--reranker", "-r", help="Cross-encoder models to try (repeatable)"
    ),
    candidates: int = typer.Option(30, help="Candidates per retriever before fusion/rerank"),
    out: Path = typer.Option(Path("results")),
):
    """Run the retrieval ablation: vector, BM25, hybrid, hybrid + each reranker."""
    from reposcout.evals.retrieval import EvalQuestion, evaluate, markdown_table, save
    from reposcout.retrieval.retriever import CrossEncoderReranker, RetrievalConfig, Retriever

    s, embedder, store = _setup()
    questions = EvalQuestion.load(dataset)
    base = Retriever(embedder, store)
    runs = [
        (base, RetrievalConfig(keyword=False, candidates=candidates)),
        (base, RetrievalConfig(vector=False, candidates=candidates)),
        (base, RetrievalConfig(candidates=candidates)),
    ]
    for name in reranker:
        runs.append((Retriever(embedder, store, CrossEncoderReranker(name)), RetrievalConfig(rerank=True, candidates=candidates)))

    base.retrieve("warm up", 1, RetrievalConfig())  # builds the BM25 index outside the timings
    results = []
    for retriever, cfg in runs:
        r = evaluate(retriever, questions, cfg)
        if cfg.rerank:
            r["config"] += f" ({retriever.reranker.name.split('/')[-1]})"
        results.append(r)
        typer.echo(f"done: {r['config']}  recall@5={r['all']['recall@5']:.1%}")
    meta = {
        "dataset": f"{dataset} ({len(questions)} questions)",
        "index": f"{store.count()} chunks",
        "embedder": s.embed_model,
        "candidates": candidates,
        "commit": store.chunks[0].commit if getattr(store, "chunks", None) else None,
    }
    path = save(results, out, meta)
    typer.echo("\n" + markdown_table(results) + f"\n\nSaved {path}")


@app.command("eval-answers")
def eval_answers(
    repo: Path = typer.Argument(..., exists=True, file_okay=False, help="Airflow checkout with history"),
    dataset: Path = typer.Option(Path("evals/airflow_v1.jsonl"), exists=True),
    limit: int = typer.Option(None, help="Only the first N questions (cheap smoke run)"),
    judge_model: str = typer.Option(None, "--judge", help="claude-code:<model>, anthropic:<model> or ollama:<model>"),
    mode: str = typer.Option("agent", help="agent (LangGraph tool loop) or rag (single-shot baseline)"),
    out: Path = typer.Option(Path("results")),
):
    """Answer each eval question, then grade it with an LLM judge.

    Answers use REPOSCOUT_CHAT_MODEL (API key, or ollama:<model> for free). The judge
    defaults to Claude Code on your Claude subscription; see REPOSCOUT_JUDGE.
    """
    import json

    from reposcout.evals.answers import agent_answerer, make_judge, rag_answerer, run_answer_eval, save
    from reposcout.llm import get_chat_model, model_name

    s, embedder, store = _setup()
    judge = make_judge(judge_model or s.judge, s.claude_bin)
    questions = [json.loads(l) for l in dataset.read_text(encoding="utf-8").splitlines() if l.strip()][:limit]
    if mode == "rag":
        from reposcout.retrieval.retriever import build_retriever

        retriever, cfg = build_retriever(s, embedder, store)
        answerer = rag_answerer(retriever, cfg, get_chat_model(s.chat_model, s.max_answer_tokens),
                                model_name(s.chat_model), s.top_k)
        label = f"rag ({cfg.label}) -> {model_name(s.chat_model)}"
    elif mode == "agent":
        from reposcout.agent.run import ToolboxFactory, make_agent

        answerer = agent_answerer(make_agent(s), ToolboxFactory(s, embedder, store, repo), out / "traces")
        label = f"agent -> {model_name(s.chat_model)}"
    else:
        raise typer.BadParameter("mode must be 'agent' or 'rag'")
    # One checkpoint per (mode, answer model, judge, dataset): re-running resumes it.
    slug = "-".join(x.replace(":", "_").replace("/", "_") for x in (mode, s.chat_model, judge.name, dataset.stem))
    checkpoint = out / "checkpoints" / f"{slug}.jsonl"
    result = run_answer_eval(questions, repo, answerer, judge, label, checkpoint)
    path = save(result, out, {"dataset": str(dataset), "embedder": s.embed_model, "judge": judge.name, "mode": mode,
                              "checkpoint": str(checkpoint)})
    summary = {k: v for k, v in result.items() if k != "per_question"}
    typer.echo(json.dumps(summary, indent=1) + f"\nSaved {path}")


def _latest_agent_answers(results: Path) -> Path:
    import json

    runs = sorted(results.glob("answers-*.json"), reverse=True)
    for p in runs:
        meta = json.loads(p.read_text(encoding="utf-8")).get("meta", {})
        if meta.get("mode") == "agent":
            return p
    raise typer.BadParameter(f"no agent answer run in {results}; run `reposcout eval-answers` first")


def _review_items(dataset: Path, answers: Path | None, repo: Path, n: int, seed: int) -> list[dict]:
    """Build the sample once and cache it, so the page shows the same questions every time."""
    import json

    from reposcout.evals.review import build_sample

    answers = answers or _latest_agent_answers(Path("results"))
    cache = Path("data/review") / f"sample-{dataset.stem}-n{n}-seed{seed}-{answers.stem}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    items = build_sample(dataset, answers, repo, n, seed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(items, indent=1), encoding="utf-8")
    return items


@app.command()
def review(
    repo: Path = typer.Option(Path("data/airflow"), exists=True, file_okay=False),
    dataset: Path = typer.Option(Path("evals/airflow_v1.jsonl"), exists=True),
    answers: Path = typer.Option(None, help="answers-*.json to grade (default: latest agent run)"),
    labels: Path = typer.Option(Path("evals/spot_check_v1.json"), help="Where your labels are saved"),
    n: int = typer.Option(20),
    seed: int = typer.Option(7),
    port: int = typer.Option(8765),
):
    """Hand-check a random sample of eval questions and grade the agent's answers in a browser."""
    import uvicorn

    from reposcout.evals.review import LabelStore, create_app

    items = _review_items(dataset, answers, repo, n, seed)
    typer.echo(f"Reviewing {len(items)} questions; labels save to {labels}. Open http://127.0.0.1:{port}")
    uvicorn.run(create_app(items, LabelStore(labels)), host="127.0.0.1", port=port, log_level="warning")


@app.command("review-summary")
def review_summary(
    repo: Path = typer.Option(Path("data/airflow"), exists=True, file_okay=False),
    dataset: Path = typer.Option(Path("evals/airflow_v1.jsonl"), exists=True),
    answers: Path = typer.Option(None),
    labels: Path = typer.Option(Path("evals/spot_check_v1.json"), exists=True),
    n: int = typer.Option(20),
    seed: int = typer.Option(7),
):
    """Your verdicts on the sample and how often the LLM judge agreed with your grades."""
    import json

    from reposcout.evals.review import LabelStore, summarize

    items = _review_items(dataset, answers, repo, n, seed)
    typer.echo(json.dumps(summarize(items, LabelStore(labels).labels), indent=1))


@app.command("compare-answers")
def compare_answers(
    baseline: Path = typer.Argument(..., exists=True, help="answers-*.json for the baseline (e.g. rag)"),
    candidate: Path = typer.Argument(..., exists=True, help="answers-*.json to compare (e.g. agent)"),
):
    """Paired bootstrap over the questions both runs answered: which differences are real?"""
    import json

    from reposcout.evals.compare import paired_bootstrap

    def rows(p: Path) -> tuple[str, dict]:
        r = json.loads(p.read_text(encoding="utf-8"))["result"]
        return r["config"], {q["id"]: q for q in r["per_question"]}

    (name_a, a), (name_b, b) = rows(baseline), rows(candidate)
    ids = sorted(a.keys() & b.keys())
    typer.echo(f"{name_b}  vs  {name_a}   ({len(ids)} paired questions)\n")
    metrics = {
        "fully correct (2)": lambda q: float(q["correctness"] == 2),
        "mean correctness (0-2)": lambda q: float(q["correctness"]),
        "faithful": lambda q: float(q["faithful"]),
        "cited a gold file": lambda q: float(q["cited_gold_file"]),
        "abstained": lambda q: float(q["abstained"]),
    }
    typer.echo(f"| Metric | {name_a[:28]} | {name_b[:28]} | Difference | 95% CI | Wins/losses |")
    typer.echo("|---|---|---|---|---|---|")
    for label, f in metrics.items():
        xa, xb = [f(a[i]) for i in ids], [f(b[i]) for i in ids]
        r = paired_bootstrap(xa, xb)
        lo, hi = r["ci95"]
        mark = " *" if lo > 0 or hi < 0 else ""
        typer.echo(f"| {label} | {sum(xa) / len(ids):.2f} | {sum(xb) / len(ids):.2f} | {r['mean_diff']:+.2f}{mark} "
                   f"| [{lo:+.2f}, {hi:+.2f}] | {r['wins']}/{r['losses']} |")
    typer.echo("\n* = 95% interval excludes 0")


if __name__ == "__main__":
    app()
