# How RepoScout works, end to end

This guide follows the whole loop: building the index, answering questions, measuring
quality, and checking the measurements by hand. Commands run from the repo root with
the virtualenv active (`.venv/Scripts/reposcout ...` on Windows).

```
            ┌────────────── 1. INDEX (once, then incremental) ──────────────┐
 Airflow    │ code (.py) ─ AST chunker ─┐                                    │
 checkout ──┤ docs (.md/.rst) ─ heading ┼─► embed (bge-small) ─► index      │
            │ git log (merged PRs) ─────┘    + BM25 keyword index          │
            └───────────────────────────────────────────────────────────────┘
                                         │
            ┌──────────── 2. ANSWER (each question) ─────────────────────────┐
 question ─►│ ask:   hybrid search ─► top 8 chunks ─► Claude ─► cited answer │
            │ agent: Claude ⇄ tools (search / read_file / find_symbol)       │
            │        ─► verify citations ─► cited answer                     │
            └───────────────────────────────────────────────────────────────┘
                                         │
            ┌──────────── 3. MEASURE ────────────────────────────────────────┐
 82 real   ─┤ eval-retrieval: did search find the files the fix changed?     │
 Airflow    │ eval-answers:   answer each question, LLM judge grades it      │
 fixes      │ compare-*:      paired bootstrap, which differences are real   │
            └───────────────────────────────────────────────────────────────┘
                                         │
            ┌──────────── 4. CHECK THE MEASUREMENTS ─────────────────────────┐
            │ review: you hand-check 20 questions and grade 20 answers;      │
            │ the summary shows how often the LLM judge agreed with you      │
            └───────────────────────────────────────────────────────────────┘
```

---

## 0. One-time setup

| Step | Command / action |
|---|---|
| Python env | `py -3.12 -m venv .venv` then `.venv/Scripts/pip install -e ".[local,dev]"` |
| Secrets | Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` (never committed) |
| Judge | Install Claude Code (`npm install -g @anthropic-ai/claude-code`) and sign in once with `claude auth login` |
| Local model (optional) | Install Ollama, then `ollama pull qwen3:8b` |
| Postgres (optional) | `docker compose up -d db` and set `REPOSCOUT_STORE=pg`. Written but not yet tested; the default in-memory index needs nothing |

---

## 1. Build the index

### 1a. Code and docs

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/apache/airflow.git data/airflow
git -C data/airflow sparse-checkout set airflow-core/src/airflow/models airflow-core/docs/core-concepts
reposcout ingest data/airflow
```

What happens:
1. **Walk the repo** (`ingest/repo.py`) for `.py`, `.md` and `.rst` files, recording the
   commit SHA so every citation can link to the exact version.
2. **Chunk** each file:
   - **Python** (`ingest/code.py`): the stdlib `ast` module gives one chunk per
     function or class. Classes over 120 lines are split into methods. Functions over
     80 lines are cut into overlapping 60-line windows, because the embedder only reads
     about 512 tokens (~50 lines). Module-level code becomes `<module>` chunks.
   - **Docs** (`ingest/docs.py`): one chunk per section under a heading, keeping the
     heading path (e.g. `Dags > Default Arguments`). Long sections are split at blank
     lines, or at line breaks for long lists.
   - Every chunk stores its exact start and end line, which is what makes `file:line`
     citations possible.
3. **Skip what's already indexed.** A chunk's id hashes its location and text, so
   re-running `ingest` only embeds chunks that changed.
4. **Embed** each chunk with `BAAI/bge-small-en-v1.5`, a local model that turns text
   into 384 numbers, then **store** the vectors and text in `data/index/`.

### 1b. Merged pull requests

```bash
git -C data/airflow fetch --filter=blob:none --shallow-since=2025-06-01 origin main
reposcout ingest-prs data/airflow -p airflow-core/src/airflow/models -p airflow-core/docs/core-concepts
```

Airflow squash-merges PRs, so each commit on `main` holds the PR title, number and
description. `ingest/prs.py` reads them from `git log`, so there's no GitHub API rate
limit. Each PR chunk keeps its **merge date**, which the eval uses to prevent leakage
(see 3a).

### 1c. The keyword index

There's no separate step. The first search builds a BM25 index over the same chunks
(`retrieval/bm25.py`). Its tokenizer splits identifiers, so `clear_task_instances`
also matches "clear task instances", and it stems plain words, so "clearing" matches
"clear".

---

## 2. Answer questions

### 2a. Search only (no LLM, free)

```bash
reposcout search "How are task instance notes loaded when clearing task instances?"
```

**Hybrid search** (`retrieval/retriever.py`):
1. **Vector search** embeds the question and finds the chunks with the closest
   meaning (cosine similarity).
2. **BM25 keyword search** finds the chunks sharing rare words and identifiers.
3. **Reciprocal rank fusion (RRF)** merges the two ranked lists. Each chunk scores
   `1/(60 + rank)` per list. It uses only ranks, so the two very different score
   scales never have to be calibrated against each other.
4. **Rerank** (optional, off by default): a cross-encoder rereads the top 30. It was
   measured as not worth its latency (see the README).

### 2b. Single-shot RAG

```bash
reposcout ask "What happens when a DagRun state is updated?"
```

Hybrid search → the top 8 chunks go to Claude Haiku 5.5 as numbered sources `[1]…[8]`
→ Claude answers using only those, citing `[n]` → each `[n]` becomes a GitHub link, and
any citation pointing at nothing is flagged. Tokens, cost and latency are printed.

### 2c. The agent

```bash
reposcout agent "What happens to a Dag run's state and queued_at when I clear its tasks?" --trace
```

A LangGraph state machine (`agent/graph.py`):

```
START → agent ──tool calls──► tools ──► agent ──… ──final text──► verify ──ok──► END
          │                                                    │
          └──budget used up──► finalize ──► verify             └─problems─► agent (once)
```

| Step | What it does |
|---|---|
| **agent** | Claude, with 5 tools bound, decides whether to call tools or answer |
| **tools** | Runs each call with validated arguments (`agent/tools.py`): `search_code`, `search_docs`, `search_prs`, `read_file` (exact lines, up to 150), `find_symbol` (definitions and uses). A bad call returns an error to the model instead of crashing |
| **source registry** | Every tool result gets an id like `[S3]` tied to an exact location. The agent may only cite these ids |
| **verify** | Rejects citations no tool returned, uncited claims (unless it says it couldn't find the answer), and `backticked` names that aren't in the cited source. A failed answer goes back once with the reasons |
| **finalize** | When a budget runs out (6 tool rounds, 15 tool calls, 120K input tokens, 120 s), asks for an answer with no further tool use |
| **trace** | Every LLM and tool call is logged with timing and tokens. `--trace` prints it and saves JSON to `results/traces/` |

Both modes are also an HTTP API: `uvicorn reposcout.api:app`, then `POST /ask` or
`POST /agent` with `{"question": "..."}`.

---

## 3. Measure quality

### 3a. The eval set

`evals/airflow_v1.jsonl` has **82 questions**, built by `evals/build_v1.py`:
- One per merged Airflow **bug fix** since June 2025 that changed 1–3 files in
  `airflow/models/`. Commits that only changed types, typos or comments are skipped.
- Each question describes the behavior a developer would ask about, without naming
  the file.
- **Gold evidence** = the files the fix changed, read from `git show`. It comes from
  the repo's history, not from anyone's opinion.
- **No leakage:** each question carries the fix's date, and retrieval hides any PR
  merged on or after it, so a question can never find its own fix.

### 3b. Retrieval eval (free, a few minutes on CPU, mostly the reranker)

```bash
reposcout eval-retrieval
```

Runs every question through vector-only, BM25-only, hybrid and hybrid + reranker, and
reports **recall@k** (share of gold files in the top k), **hit@5** and **MRR@10** (how
high the first gold file ranks), split by whether the question names a code symbol.
Writes `results/retrieval-<time>.md` and `.json`.

### 3c. Answer eval (paid answers, judge on your Claude plan)

```bash
reposcout eval-answers data/airflow --limit 10            # agent, quick check
reposcout eval-answers data/airflow --mode rag            # single-shot baseline, all 82
reposcout eval-answers data/airflow                       # agent, all 82
```

For each question:
1. **Answer** with the chosen system (`--mode agent` or `rag`), using `REPOSCOUT_CHAT_MODEL`.
2. **Judge:** Claude Sonnet (through `claude -p` on your subscription by default)
   sees the question, the **real fix** (PR title, description and files changed), the
   exact sources the answer saw, and the answer. It returns:
   - `correctness` 0/1/2
   - `faithful`: is every claim supported by its cited sources?
   - `abstained`: did it say "the sources don't say"?
3. **Also recorded:** did it cite a gold file, invalid citations, latency, cost, and
   for the agent, tool calls and stop reason.
4. **Checkpoint:** each graded question is appended to `results/checkpoints/`. If the
   run stops (an API error, or your plan's usage limit), the same command resumes
   where it left off.

Writes `results/answers-<time>.json`, plus per-question traces for the agent.

### 3d. Are the differences real?

```bash
reposcout compare-answers results/answers-<rag run>.json results/answers-<agent run>.json
```

**Paired bootstrap:** resample the 82 questions with replacement 10,000 times and
recompute the difference each time. If the 95% interval of those differences excludes
0, the difference is marked `*` and counts as real; otherwise it's noise at this
sample size. The retrieval ablation uses the same method.

---

## 4. Check the measurements by hand

An LLM wrote the questions and an LLM grades the answers, so both need a human check.

```bash
reposcout review          # opens http://127.0.0.1:8765
```

The page shows a **seeded random sample of 20** questions. For each one you see the
real fix (diff, commit message, files changed) and the agent's answer, and you record:

| You decide | Options | Key |
|---|---|---|
| Is the question fair? | Keep · Reword · Fix gold files · Drop, plus a note | `1`–`4` |
| How good is the agent's answer, judged against the diff? | 0 · 1 · 2 | `q` `w` `e` |

You don't need to know Airflow already. **The diff is the answer key:** `+` lines were
added and `-` lines removed by the fix. The judge's grade stays hidden while you grade.
Labels save on every click to `evals/spot_check_v1.json`.

```bash
reposcout review-summary
```

Prints verdict counts, the questions you changed, and **judge agreement** (exact and
within one point). That agreement is the evidence behind every LLM-judged number.

**Applying your verdicts** (not automated yet): rewordings and drops go into
`QUESTIONS` in `evals/build_v1.py`, then rebuild the dataset as v1.1 and rerun 3b–3d.

---

## Where things live

| Path | What | In git? |
|---|---|---|
| `src/reposcout/ingest/` | Chunkers, PR reader, indexing pipeline | yes |
| `src/reposcout/retrieval/` | BM25, RRF, reranker, retriever | yes |
| `src/reposcout/agent/` | Tools, source registry, LangGraph graph | yes |
| `src/reposcout/evals/` | Retrieval eval, answer eval and judges, bootstrap, review tool | yes |
| `evals/` | Eval set, its builder, your review labels | yes |
| `results/*.json` / `.md` | Eval results behind the README numbers | yes |
| `results/checkpoints/` | Per-question grades (resume and audit) | yes |
| `results/traces/` | Per-question agent traces | no (large) |
| `data/` | Airflow checkout, vector index, review sample cache | no (rebuildable) |
| `.env` | API key | no |

## Settings (`.env` or environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Needed for Haiku answers |
| `REPOSCOUT_CHAT_MODEL` | `anthropic:claude-haiku-5-5` | Model that answers; `ollama:qwen3:8b` for local |
| `REPOSCOUT_JUDGE` | `claude-code:sonnet` | Judge; `anthropic:claude-sonnet-5-5` uses the API instead |
| `REPOSCOUT_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (`hashing` = offline test embedder) |
| `REPOSCOUT_KEYWORD_SEARCH` | `true` | BM25 on/off (off = vector only) |
| `REPOSCOUT_RERANK_MODEL` | none | Cross-encoder name to turn reranking on |
| `REPOSCOUT_TOP_K` | `8` | Chunks given to single-shot RAG |
| `REPOSCOUT_STORE` | `memory` | `pg` for Postgres + pgvector |
| `REPOSCOUT_REPO_DIR` | `data/airflow` | Checkout used by `read_file` |
| `REPOSCOUT_OLLAMA_NUM_CTX` | `12288` | Ollama context; its 4,096 default silently cuts prompts |
| `REPOSCOUT_CLAUDE_BIN` | `claude` | Path to the Claude Code CLI if it isn't on PATH |

## Typical loops

| You changed… | Run | Cost |
|---|---|---|
| Chunking or retrieval | `ingest --rebuild` → `eval-retrieval` → compare with the last results | Free |
| The agent's prompt or tools | `eval-answers --limit 10` to sanity-check, then a full run → `compare-answers` | ~$0.40 API + 82 judge calls |
| The answer model | `REPOSCOUT_CHAT_MODEL=...` → `eval-answers` (new checkpoint automatically) → `compare-answers` | Free with Ollama |
| The eval set | `review` → apply verdicts → rebuild v1.1 → rerun everything | Your time |

Tests: `pytest` (51 tests; no network or API calls; the LLM is replaced by a scripted
model).
