"""Settings read from environment variables (prefix REPOSCOUT_) or a .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REPOSCOUT_", env_file=".env", extra="ignore")

    # "memory" stores vectors in a local .npz file; "pg" uses Postgres + pgvector.
    store: str = "memory"
    index_dir: Path = Path("data/index")
    repo_dir: Path = Path("data/airflow")  # indexed checkout, for the agent's read_file tool
    database_url: str = "postgresql+psycopg://reposcout:reposcout@localhost:5432/reposcout"

    # Small and fast on CPU. Switch to BAAI/bge-m3 once running on the GPU.
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_batch_size: int = 64

    # "<provider>:<model>", e.g. "anthropic:claude-haiku-5-5" or "ollama:qwen3:8b"
    chat_model: str = "anthropic:claude-haiku-5-5"
    max_answer_tokens: int = 4000
    top_k: int = 8
    # Retrieval: hybrid (vector + BM25) by default; set a cross-encoder to rerank.
    keyword_search: bool = True
    rerank_model: str | None = None
    rerank_candidates: int = 30

    github_repo: str = "apache/airflow"  # for citation links

    # Answer-eval judge: "claude-code:<model>" runs the Claude Code CLI on a Claude
    # subscription (no API key); "anthropic:<model>" or "ollama:<model>" use LangChain.
    judge: str = "claude-code:sonnet"
    claude_bin: str = "claude"  # path to the Claude Code CLI if it is not on PATH


@lru_cache
def get_settings() -> Settings:
    # Provider SDKs read their keys (e.g. ANTHROPIC_API_KEY) from the process
    # environment, not from Settings, so load .env into it. Real env vars win.
    from dotenv import load_dotenv

    load_dotenv(".env", override=False)
    return Settings()


# USD per 1M tokens (input, output), for cost-per-query tracking.
# Anthropic list prices as of 2026-11; Haiku 5.5 rate applies to prompts up to 100K tokens.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))  # local models cost 0
    return (input_tokens * pin + output_tokens * pout) / 1_000_000
