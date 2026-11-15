"""One factory for chat models so hosted and local models can be swapped from config."""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

# Ollama defaults to a 4,096-token context and silently drops anything beyond it, which
# is smaller than one agent turn with tool results. 12K fits an 8B Q4 model plus its
# KV cache in 8 GB of VRAM. Override with REPOSCOUT_OLLAMA_NUM_CTX.
OLLAMA_NUM_CTX = 12_288


def get_chat_model(spec: str, max_tokens: int = 4000) -> BaseChatModel:
    """spec is "<provider>:<model>", e.g. "anthropic:claude-haiku-5-5" or "ollama:qwen3:8b"."""
    provider, _, model = spec.partition(":")
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # No temperature: current Claude models reject non-default sampling params.
        return ChatAnthropic(model=model, max_tokens=max_tokens)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        num_ctx = int(os.environ.get("REPOSCOUT_OLLAMA_NUM_CTX", OLLAMA_NUM_CTX))
        return ChatOllama(model=model, num_predict=max_tokens, num_ctx=num_ctx)
    raise ValueError(f"Unknown chat model provider in {spec!r}; use anthropic:<model> or ollama:<model>")


def model_name(spec: str) -> str:
    return spec.partition(":")[2]


def message_text(msg: AIMessage) -> str:
    """Final text of a reply, skipping thinking blocks and other non-text content."""
    if isinstance(msg.content, str):
        return msg.content
    parts = []
    for block in msg.content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def token_usage(msg: AIMessage) -> tuple[int, int]:
    usage = getattr(msg, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
