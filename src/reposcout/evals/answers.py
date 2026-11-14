"""Answer-quality eval: an LLM judge grades each answer against the real fix.

The reference for each question is the merged fix commit (PR title, description and
files changed). The judge sees the question, the reference, the answer, and the exact
sources the answer was given, and returns a structured verdict:

- correctness (0-2): 0 wrong or no answer, 1 partly right, 2 matches the reference
- faithful (bool): every claim in the answer is supported by its cited sources
- abstained (bool): the answer says the sources do not contain the answer

Judges make mistakes, so `agreement()` compares judge labels with hand labels.

Two judge backends: `LangChainJudge` (API key or Ollama) and `ClaudeCodeJudge`, which
calls the Claude Code CLI headlessly so judging runs on a Claude subscription.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Literal, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from reposcout.config import cost_usd
from reposcout.rag import answer_question, format_sources
from reposcout.retrieval.retriever import RetrievalConfig, Retriever

JUDGE_PROMPT = """You grade answers to developer questions about the Apache Airflow codebase.

You get the question, a reference describing the merged change that answers it, the \
numbered sources the answer was allowed to use, and the answer.

Grade strictly:
- correctness: 2 if the answer explains the same behavior or mechanism as the reference; \
1 if it is on the right code but misses or blurs the key point; 0 if it is wrong, \
unrelated, or only says it cannot answer.
- faithful: true only if every factual claim in the answer is supported by the source \
it cites. Uncited claims about code that the sources do not show make it false.
- abstained: true if the answer says the sources do not contain the answer.

The reference describes a fix; the current code already includes it. An answer that \
correctly describes the current, fixed behavior is correct."""


class Verdict(BaseModel):
    reasoning: str = Field(description="Two or three sentences comparing the answer to the reference.")
    correctness: Literal[0, 1, 2]
    faithful: bool
    abstained: bool


def reference_for(repo: Path, commit: str, max_chars: int = 4000) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "show", "--stat", "--format=%s%n%n%b", commit],
        capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    return out[:max_chars]


def _judge_message(question: str, reference: str, sources: str, answer: str) -> str:
    return (
        f"Question:\n{question}\n\nReference (the merged change):\n{reference}\n\n"
        f"Sources given to the answer:\n{sources}\n\nAnswer:\n{answer}"
    )


class Judge(Protocol):
    name: str

    def grade(self, question: str, reference: str, sources: str, answer: str) -> tuple[Verdict, float]:
        """Return the verdict and its API cost in USD (0 when it runs on a subscription)."""
        ...


class LangChainJudge:
    """Judge through a LangChain chat model (Anthropic API key or Ollama)."""

    def __init__(self, spec: str):
        from reposcout.llm import get_chat_model, model_name

        self.name = spec
        self._model = model_name(spec)
        self._llm = get_chat_model(spec, 2000)

    def grade(self, question: str, reference: str, sources: str, answer: str) -> tuple[Verdict, float]:
        structured = self._llm.with_structured_output(Verdict, method="json_schema", include_raw=True)
        out = structured.invoke(
            [SystemMessage(JUDGE_PROMPT), HumanMessage(_judge_message(question, reference, sources, answer))]
        )
        if out.get("parsing_error") or out.get("parsed") is None:
            raise RuntimeError(f"judge output did not parse: {out.get('parsing_error')}")
        usage = getattr(out["raw"], "usage_metadata", None) or {}
        return out["parsed"], cost_usd(self._model, usage.get("input_tokens", 0), usage.get("output_tokens", 0))


class ClaudeCodeJudge:
    """Judge through Claude Code in headless mode (`claude -p`), signed in with a
    Claude subscription, so judging needs no API key.

    The CLI validates the reply against Verdict's JSON schema. It runs with no tools,
    no MCP servers and no saved session, from an empty temp folder so no project
    CLAUDE.md is loaded. Uses the subscription's usage limits, so it is meant for
    local runs, not CI.
    """

    def __init__(self, binary: str = "claude", model: str = "sonnet", timeout_s: int = 300):
        self.binary = binary
        self.model = model
        self.timeout_s = timeout_s
        self.name = f"claude-code:{model}"
        self._schema = json.dumps({**Verdict.model_json_schema(), "additionalProperties": False})

    def _resolve_binary(self) -> str:
        """Find the real executable. On Windows, npm installs a claude.cmd wrapper, and
        cmd.exe mangles the quotes in the JSON schema argument, so call the claude.exe
        the wrapper points at instead."""
        found = shutil.which(self.binary) or self.binary
        if found.lower().endswith((".cmd", ".bat")):
            exe = Path(found).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
            if exe.exists():
                return str(exe)
        return found

    def _cmd(self) -> list[str]:
        return [
            self._resolve_binary(), "-p", "--output-format", "json", "--model", self.model,
            "--tools", "", "--strict-mcp-config", "--no-session-persistence",
            "--system-prompt", JUDGE_PROMPT, "--json-schema", self._schema,
        ]

    @staticmethod
    def _env() -> dict[str, str]:
        """Environment for the CLI with Claude-session variables removed.

        When the eval is started from inside Claude Code (e.g. the desktop app), the
        parent's CLAUDE_CODE_* variables make the child CLI act as a hosted session and
        ignore the user's own login, and an ANTHROPIC_API_KEY would bill the API
        instead of the subscription. CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`)
        is kept, since it is the documented way to sign in a headless CLI.
        """
        keep = {"CLAUDE_CODE_OAUTH_TOKEN"}
        return {
            k: v for k, v in os.environ.items()
            if k in keep or not k.startswith(("CLAUDE_CODE_", "ANTHROPIC_"))
        }

    def grade(self, question: str, reference: str, sources: str, answer: str) -> tuple[Verdict, float]:
        with tempfile.TemporaryDirectory() as cwd:
            try:
                proc = subprocess.run(
                    self._cmd(), input=_judge_message(question, reference, sources, answer),
                    capture_output=True, text=True, encoding="utf-8", timeout=self.timeout_s, cwd=cwd,
                    env=self._env(),
                )
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"Claude Code CLI not found at {self.binary!r}. Install it and run `claude` once to "
                    "sign in, or set REPOSCOUT_CLAUDE_BIN to its path."
                ) from e
        return parse_claude_output(proc.stdout, proc.stderr), 0.0


def parse_claude_output(stdout: str, stderr: str = "") -> Verdict:
    """Turn `claude -p --output-format json` output into a Verdict, or raise clearly."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"claude -p returned non-JSON output: {(stdout or stderr)[:300]!r}") from e
    if data.get("is_error"):
        raise RuntimeError(f"claude -p failed: {data.get('result')!r}")
    payload = data.get("structured_output")
    if payload is None:  # fall back to JSON in the text result
        text = (data.get("result") or "").strip().removeprefix("```json").removesuffix("```").strip()
        payload = json.loads(text)
    return Verdict.model_validate(payload)


def make_judge(spec: str, claude_bin: str = "claude") -> Judge:
    """'claude-code:<model>' uses the Claude Code CLI; anything else is a LangChain model spec."""
    if spec.startswith("claude-code"):
        model = spec.partition(":")[2] or "sonnet"
        return ClaudeCodeJudge(claude_bin, model)
    return LangChainJudge(spec)


def run_answer_eval(
    questions: list[dict],
    repo: Path,
    retriever: Retriever,
    cfg: RetrievalConfig,
    llm: BaseChatModel,
    llm_model: str,
    judge: Judge,
    k: int = 8,
) -> dict:
    rows = []
    for q in questions:
        a = answer_question(q["question"], retriever, llm, llm_model, k, cfg=replace(cfg, before_date=q.get("date")))
        verdict, judge_cost = judge.grade(q["question"], reference_for(repo, q["commit"]),
                                          format_sources(a.retrieved), a.answer)
        gold = set(q["gold_files"])
        rows.append({
            "id": q["id"],
            "answer": a.answer,
            "correctness": verdict.correctness,
            "faithful": verdict.faithful,
            "abstained": verdict.abstained,
            "judge_reasoning": verdict.reasoning,
            "cited_gold_file": any(c.location.split(":")[0] in gold for c in a.citations),
            "invalid_citations": len(a.invalid_citations),
            "latency_ms": a.retrieve_ms + a.generate_ms,
            "answer_cost_usd": a.cost_usd,
            "judge_cost_usd": judge_cost,
        })
    n = len(rows) or 1
    lat = sorted(r["latency_ms"] for r in rows)
    return {
        "config": f"{cfg.label} -> {llm_model}",
        "judge": judge.name,
        "n": len(rows),
        "correct (2)": sum(r["correctness"] == 2 for r in rows) / n,
        "mean_correctness": statistics.fmean(r["correctness"] for r in rows) if rows else 0.0,
        "faithful": sum(r["faithful"] for r in rows) / n,
        "abstained": sum(r["abstained"] for r in rows) / n,
        "cited_gold_file": sum(r["cited_gold_file"] for r in rows) / n,
        "answers_with_invalid_citation": sum(r["invalid_citations"] > 0 for r in rows) / n,
        "p50_latency_ms": lat[len(lat) // 2] if lat else 0.0,
        "cost_per_answer_usd": sum(r["answer_cost_usd"] for r in rows) / n,
        "judge_cost_usd": sum(r["judge_cost_usd"] for r in rows),
        "per_question": rows,
    }


def agreement(judged: list[dict], hand_labels: dict[str, int]) -> float:
    """Share of hand-labelled questions where the judge's correctness matches yours."""
    pairs = [(r["correctness"], hand_labels[r["id"]]) for r in judged if r["id"] in hand_labels]
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else float("nan")


def save(result: dict, out_dir: Path, meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"answers-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"meta": meta, "result": result}, indent=1), encoding="utf-8")
    return path
