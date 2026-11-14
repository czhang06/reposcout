"""Baseline RAG: retrieve top-k chunks, answer from them only, cite them as [n].

This is the "no agent" baseline in the ablation table. The LangGraph agent in week 3
builds on the same retrieval and citation format.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from reposcout.config import cost_usd
from reposcout.llm import message_text, token_usage
from reposcout.retrieval.retriever import RetrievalConfig, Retriever
from reposcout.store import Hit

SYSTEM_PROMPT = """You answer developer questions about the Apache Airflow codebase.

Use only the numbered sources you are given. After each claim, cite the source it came \
from as [n], for example "Notes are loaded eagerly [2]." If the sources do not contain \
the answer, say what is missing instead of guessing. Keep the answer short and \
concrete: name files, classes and functions where they help."""

_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Citation:
    n: int
    location: str
    url: str | None


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation]
    invalid_citations: list[int]  # [n] markers that point at no source
    retrieved: list[Hit] = field(repr=False)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0


def format_sources(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        label = f"[{i}] {c.citation}" + (f" ({c.symbol})" if c.symbol else "")
        blocks.append(f"{label}\n```\n{c.text}\n```")
    return "\n\n".join(blocks)


def github_url(repo: str, hit: Hit) -> str | None:
    c = hit.chunk
    if c.source_type == "pr":
        return f"https://github.com/{repo}/pull/{c.extra['pr']}" if c.extra.get("pr") else None
    if c.source_type == "issue" or not c.commit:
        return None
    return f"https://github.com/{repo}/blob/{c.commit}/{c.path}#L{c.start_line}-L{c.end_line}"


def parse_citations(text: str, hits: list[Hit], repo: str) -> tuple[list[Citation], list[int]]:
    cited = sorted({int(n) for n in _CITE.findall(text)})
    valid = [n for n in cited if 1 <= n <= len(hits)]
    invalid = [n for n in cited if n not in valid]
    return [Citation(n, hits[n - 1].chunk.citation, github_url(repo, hits[n - 1])) for n in valid], invalid


def answer_question(
    question: str,
    retriever: Retriever,
    llm: BaseChatModel,
    model: str,
    k: int = 8,
    repo: str = "apache/airflow",
    cfg: RetrievalConfig | None = None,
) -> Answer:
    t0 = time.perf_counter()
    hits = retriever.retrieve(question, k, cfg or RetrievalConfig()).hits
    t1 = time.perf_counter()

    prompt = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"
    reply = llm.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)])
    t2 = time.perf_counter()

    text = message_text(reply)
    citations, invalid = parse_citations(text, hits, repo)
    tin, tout = token_usage(reply)
    return Answer(
        question=question,
        answer=text,
        citations=citations,
        invalid_citations=invalid,
        retrieved=hits,
        input_tokens=tin,
        output_tokens=tout,
        cost_usd=cost_usd(model, tin, tout),
        retrieve_ms=(t1 - t0) * 1000,
        generate_ms=(t2 - t1) * 1000,
    )
