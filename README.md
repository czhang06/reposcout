# RepoScout

Agentic RAG over the Apache Airflow codebase, docs and issues. It answers developer
questions with `file:line` citations that link to GitHub at the indexed commit.

**Status: week 3 of 5.** Ingestion of code, docs and merged PRs; hybrid search (vector +
BM25 with reciprocal rank fusion); an 82-question eval set built from real Airflow
fixes, with a retrieval ablation and paired-bootstrap significance tests; a LangGraph
agent that searches, reads and cites the code, with a citation-checking step and
guardrails. On all 82 questions, the agent beats single-shot RAG on correctness and on
finding the right file, with significance tested by paired bootstrap (results below).

## Quick start

```bash
py -3.12 -m venv .venv
.venv/Scripts/pip install -e ".[local,dev]"
cp .env.example .env            # add ANTHROPIC_API_KEY to use `ask`

# Index a sparse checkout of Airflow (models package + core docs)
git clone --depth 1 --filter=blob:none --sparse https://github.com/apache/airflow.git data/airflow
git -C data/airflow sparse-checkout set airflow-core/src/airflow/models airflow-core/docs/core-concepts
reposcout ingest data/airflow

git -C data/airflow fetch --filter=blob:none --shallow-since=2025-06-01 origin main
reposcout ingest-prs data/airflow -p airflow-core/src/airflow/models -p airflow-core/docs/core-concepts

reposcout search "How are task instance notes loaded when clearing task instances?"
reposcout ask "What happens when a DagRun state is updated?"          # single-shot RAG
reposcout agent "What happens when a DagRun state is updated?" --trace  # LangGraph agent
uvicorn reposcout.api:app --reload      # POST /ask or POST /agent {"question": "..."}

reposcout eval-retrieval                 # retrieval ablation -> results/
reposcout eval-answers data/airflow --limit 10             # agent, LLM-judged (see "Judging")
reposcout eval-answers data/airflow --limit 10 --mode rag  # single-shot baseline
pytest
```

Postgres + pgvector instead of the local index: `docker compose up -d db`, then set
`REPOSCOUT_STORE=pg`.

## How it works

- **Chunking** (`ingest/code.py`, `ingest/docs.py`): Python is split with the stdlib `ast`
  module into one chunk per function or class. Large classes are split into methods,
  and functions over 80 lines are cut into overlapping 60-line windows, because the
  embedder only reads ~512 tokens. Markdown and reST are split by heading, with the
  heading path kept as context. Every chunk records an exact line range, and an
  ingest-time check against the source files confirms each range.
- **Incremental indexing**: chunk ids hash the location and the text, so re-running
  ingest embeds only chunks that changed.
- **PRs** (`ingest/prs.py`): Airflow squash-merges PRs, so each commit on `main` holds
  the PR title, number and description. Indexing reads them from git, not the GitHub
  API. Each PR chunk keeps its merge date.
- **Retrieval** (`retrieval/`): vector search (`BAAI/bge-small-en-v1.5`; exact cosine
  or pgvector HNSW) and BM25 keyword search, fused with reciprocal rank fusion. The BM25
  tokenizer splits identifiers (`clear_task_instances` -> clear, task, instances;
  `DagRun` -> dag, run) and stems plain words. An optional cross-encoder reranks the
  fused candidates. Each stage can be switched off for ablations.
- **Answering** (`rag.py`): the top-k chunks go to the model as numbered sources. Each
  `[n]` the model cites is mapped back to a GitHub link, and markers that point at no
  source are reported. Token usage and cost are recorded per query.
- **Agent** (`agent/`): a LangGraph state machine.

  ```
  START -> agent --tool calls--> tools -> agent ... --answer--> verify --ok--> END
                  \--budget spent--> finalize -> verify   verify --citation problems--> agent (once)
  ```

  - **Tools** (`agent/tools.py`): `search_code`, `search_docs`, `search_prs` (hybrid
    search limited to one source type), `read_file` (exact lines from the indexed
    checkout) and `find_symbol` (definitions and uses). Arguments are Pydantic-validated:
    a bad call, such as a `..` path or an empty query, returns an error to the model
    instead of crashing.
  - **Checkable citations**: every tool result is registered as a source with an id
    (`[S3]`) and an exact location. The `verify` node rejects an answer that cites an id
    no tool returned, cites nothing without saying it couldn't find the answer, or names
    a `backticked` identifier that isn't in the sources cited in that sentence. That
    last check catches invented function names. A failed answer goes back to the agent
    once with the reasons.
  - **Guardrails** (`Budget`): at most 6 tool rounds, 15 tool calls, 120K input tokens and
    120 s per question. When one runs out, `finalize` asks for an answer with no tools,
    and any tool calls left unrun get results so the conversation stays valid.
  - **Tracing**: every LLM call and tool call is logged with its timing, tokens and
    arguments. `--trace` prints the log and saves it as JSON; eval runs save one trace
    per question in `results/traces/`.
- **Models** (`llm.py`): `anthropic:claude-haiku-5-5` by default; `ollama:<model>` for
  local comparison runs.

## Evaluation

**Dataset** (`evals/airflow_v1.jsonl`, built by `evals/build_v1.py`): 82 questions,
one per merged Airflow bug fix since June 2025 that changed 1–3 files in
`airflow/models/`. Commits that only changed types, typos or comments are left out.
Each question describes the behavior without naming the file; 24 of the 82 mention a
code symbol. The gold evidence is the set of files the fix changed, read from
`git show`, so it isn't anyone's judgment. Questions were drafted by Claude from commit
titles and are meant to be spot-checked by hand.

**No leakage**: retrieval hides any PR merged on or after the question's date, so a
question can never retrieve its own fix. A check over all 82 questions finds 0 leaked PRs.

**Retrieval ablation** (file-level; index of 2,559 code and doc chunks; CPU):

| Retrieval | Recall@5 | Recall@10 | Hit@5 | MRR@10 | p50 latency |
|---|---|---|---|---|---|
| vector only | 63.0% | 77.6% | 68.3% | 0.501 | 11 ms |
| BM25 only | 70.5% | 85.4% | 76.8% | 0.617 | <1 ms |
| **vector + BM25 (RRF)** | **70.9%** | 82.5% | 74.4% | 0.613 | 13 ms |
| + rerank, MiniLM-L6 | 73.6% | 84.8% | 79.3% | 0.566 | 1,045 ms |
| + rerank, bge-reranker-base | 73.0% | 82.7% | 79.3% | 0.554 | 5,755 ms |

**Which differences are real** (paired bootstrap over questions, 10,000 resamples,
`src/reposcout/evals/compare.py`):
- Hybrid vs vector only: recall@5 +7.9 pts, 95% CI [+0.0, +15.9]; MRR +0.112, CI
  [+0.062, +0.166]. A real improvement.
- Hybrid vs BM25 only, and either reranker vs hybrid: every interval contains 0. With
  82 questions these gaps are noise, and the rerankers cost 80–450× the latency.
- **Decision:** default to hybrid with no reranker. Revisit with a larger eval set
  and a GPU.

Adding the 701 PRs to the index changes hybrid recall@5 from 71.5% to 69.3% (within
noise). On 39 of the 82 questions, PRs take 1–5 of the top 8 results. The agent
searches PRs with a separate tool (`search_prs`), so they don't compete with code.

**Judging answers** (`src/reposcout/evals/answers.py`): an LLM judge grades each answer
against the real fix (PR title, description and files changed) for correctness (0–2),
faithfulness to its cited sources, and abstention. There are two backends:
- `claude-code:<model>` (default): runs the Claude Code CLI headlessly (`claude -p`,
  with no tools, no MCP servers and a JSON schema for the verdict) on a Claude
  subscription, so judging needs no API key. It uses the plan's usage limits, so it is
  for local runs, not CI.
- `anthropic:<model>` or `ollama:<model>`: through LangChain.

Generating the answers is the system under test, so it always uses
`REPOSCOUT_CHAT_MODEL`: Haiku through the API, or `ollama:<model>` for free.

**Agent vs single-shot RAG** (82 questions; both answer with Claude Haiku 5.5 over
the same hybrid index; Claude Sonnet judge through Claude Code; paired bootstrap with
10,000 resamples via `reposcout compare-answers`; files in `results/answers-*.json`):

| Metric | RAG | Agent | Difference [95% CI] |
|---|---|---|---|
| Mean correctness (0–2) | 1.24 | 1.40 | **+0.16 [+0.01, +0.32]** |
| Cited a file the real fix changed | 82% | 95% | **+13 pts [+6, +22]** |
| Abstained ("sources don't say") | 27% | 11% | **−16 pts [−26, −7]** |
| Fully correct (score 2) | 48% | 50% | +2 pts [−9, +13] |
| Faithful to cited sources | 95% | 88% | −7 pts [−16, +1] |
| p50 latency | 5.0 s | 9.9 s | |
| Cost per answer | $0.0010 | $0.0045 | |

Agent behavior: 4.1 tool calls per question on average; 2% of questions hit the tool
budget; 12% ended with a citation issue the verify step could not get fixed.

What this shows:
- **The agent finds the right code more often and answers more questions.** That
  costs 4.5× the money and 2× the latency.
- **It does not yet produce more fully correct answers.** The extra answers are mostly
  partly right.
- **Faithfulness may drop.** The −7 pts is not significant, but it points the wrong way.
  Of the 10 answers the judge marked unfaithful, the verify step flagged only 1. It
  catches invented identifiers and bad citation ids, not wrong claims about what the
  code does. Checking claims against their cited sources is the next thing to add.

**Bugs found while running this eval, all fixed:**
- `read_file` output was cut to 1,800 characters, so the agent saw about 45 of the
  lines it asked for and filled the gap by guessing. On a 10-question run, faithfulness
  went from 6/10 to 9/10 after the fix.
- The judge now grades against exactly the text the agent was shown.
- The forced final answer called Claude without tools. Claude ties its thinking
  blocks to the tools list, so that request failed with a 400. It now keeps the
  same tools and sets `tool_choice: none`.
- Ollama silently cut prompts to its 4,096-token default context. RepoScout now sets 12K.

## Indexing measurements

Sample: Airflow `d099acc`, models package and core-concepts docs. 2,559 chunks
(1,081 code, 1,478 doc).

| Step | Result |
|---|---|
| Chunking | 0.2 s |
| Embedding, bge-small, CPU | 198 s |
| Re-ingest with no changes | 0 chunks re-embedded |

## Roadmap

1. ~~Ingestion, vector RAG, CLI, API~~
2. ~~Hybrid search, reranker, PR ingestion, eval set v1, retrieval ablation~~
3. ~~LangGraph agent with tools, citation verification, guardrails, JSON tracing; agent
   vs RAG eval~~. Next: check claims against their cited sources, not just identifiers.
4. Local vs hosted model comparison, streaming API and UI, CI smoke eval
5. Deployed demo, results table, MCP server
