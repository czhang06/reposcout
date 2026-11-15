"""Wiring: build an agent and its toolbox from Settings, and save run traces."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, replace
from pathlib import Path

from reposcout.agent.graph import AgentResult, Budget, RepoAgent
from reposcout.agent.tools import SourceRegistry, Toolbox
from reposcout.llm import get_chat_model, model_name
from reposcout.retrieval.retriever import build_retriever


def make_agent(settings, budget: Budget | None = None) -> RepoAgent:
    return RepoAgent(get_chat_model(settings.chat_model, settings.max_answer_tokens), model_name(settings.chat_model), budget)


class ToolboxFactory:
    """Builds a fresh Toolbox (and source registry) for each question."""

    def __init__(self, settings, embedder, store, repo_root: Path | None):
        self.settings = settings
        self.retriever, self.cfg = build_retriever(settings, embedder, store)
        self.repo_root = repo_root
        # find_symbol scans code chunks; the in-memory store has them loaded.
        self.code_chunks = [c for c in getattr(store, "chunks", []) if c.source_type == "code"]

    def __call__(self, before_date: str | None = None) -> Toolbox:
        return Toolbox(self.retriever, replace(self.cfg, before_date=before_date), self.repo_root,
                       SourceRegistry(self.settings.github_repo), self.code_chunks)


def save_trace(result: AgentResult, out_dir: Path, tag: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = tag or re.sub(r"[^a-z0-9]+", "-", result.question.lower())[:40].strip("-")
    path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}.json"
    path.write_text(json.dumps(asdict(result), indent=1, default=str), encoding="utf-8")
    return path
