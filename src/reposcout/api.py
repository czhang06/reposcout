"""HTTP API. Run with: uvicorn reposcout.api:app --reload"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from reposcout.config import get_settings
from reposcout.embed import get_embedder
from reposcout.llm import get_chat_model, model_name
from reposcout.rag import answer_question
from reposcout.retrieval.retriever import build_retriever
from reposcout.store import get_store


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=50)


class CitationOut(BaseModel):
    n: int
    location: str
    url: str | None


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    invalid_citations: list[int]
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    retrieve_ms: float
    generate_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.settings = s
    app.state.embedder = get_embedder(s.embed_model, s.embed_batch_size)
    app.state.store = get_store(s, app.state.embedder.dim)
    app.state.retriever, app.state.retrieval_cfg = build_retriever(s, app.state.embedder, app.state.store)
    app.state.llm = get_chat_model(s.chat_model, s.max_answer_tokens)
    from reposcout.agent.run import ToolboxFactory, make_agent

    app.state.agent = make_agent(s)
    repo = s.repo_dir if s.repo_dir.is_dir() else None
    app.state.toolboxes = ToolboxFactory(s, app.state.embedder, app.state.store, repo)
    yield


app = FastAPI(title="RepoScout", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": app.state.store.count()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    s = app.state.settings
    if app.state.store.count() == 0:
        raise HTTPException(503, "Index is empty; run `reposcout ingest` first.")
    a = answer_question(
        req.question, app.state.retriever, app.state.llm,
        model_name(s.chat_model), req.k or s.top_k, s.github_repo, app.state.retrieval_cfg,
    )
    return AskResponse(
        answer=a.answer,
        citations=[CitationOut(**c.__dict__) for c in a.citations],
        invalid_citations=a.invalid_citations,
        model=s.chat_model,
        input_tokens=a.input_tokens,
        output_tokens=a.output_tokens,
        cost_usd=a.cost_usd,
        retrieve_ms=round(a.retrieve_ms, 1),
        generate_ms=round(a.generate_ms, 1),
    )


class AgentCitation(BaseModel):
    sid: str
    location: str
    url: str | None
    kind: str


class AgentStep(BaseModel):
    t_ms: float
    node: str
    detail: dict


class AgentResponse(BaseModel):
    answer: str
    citations: list[AgentCitation]
    unresolved_issues: list[str]
    tool_rounds: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    stop_reason: str
    steps: list[AgentStep]


@app.post("/agent", response_model=AgentResponse)
def agent(req: AskRequest) -> AgentResponse:
    """Answer with the LangGraph agent. Slower than /ask; returns every step it took."""
    if app.state.store.count() == 0:
        raise HTTPException(503, "Index is empty; run `reposcout ingest` first.")
    r = app.state.agent.run(req.question, app.state.toolboxes())
    return AgentResponse(
        answer=r.answer,
        citations=[AgentCitation(**{k: c[k] for k in ("sid", "location", "url", "kind")}) for c in r.citations],
        unresolved_issues=r.issues,
        tool_rounds=r.rounds,
        tool_calls=r.tool_calls,
        input_tokens=r.input_tokens,
        output_tokens=r.output_tokens,
        cost_usd=r.cost_usd,
        latency_ms=round(r.latency_ms, 1),
        stop_reason=r.stop_reason,
        steps=[AgentStep(t_ms=e["t_ms"], node=e["node"], detail={k: v for k, v in e.items() if k not in ("t_ms", "node")})
               for e in r.trace],
    )
