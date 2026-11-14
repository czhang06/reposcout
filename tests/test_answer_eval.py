import json
import subprocess

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from reposcout.chunk import Chunk
from reposcout.embed import HashingEmbedder
from reposcout.evals import answers
from reposcout.evals.answers import (
    ClaudeCodeJudge, LangChainJudge, Verdict, agreement, make_judge, parse_claude_output, run_answer_eval,
)
from reposcout.ingest.pipeline import index_chunks
from reposcout.retrieval.retriever import RetrievalConfig, Retriever
from reposcout.store.memory import MemoryStore


class FakeJudge:
    name = "fake"

    def __init__(self, verdicts, cost=0.01):
        self.verdicts, self.cost, self.calls = iter(verdicts), cost, []

    def grade(self, question, reference, sources, answer):
        self.calls.append((question, reference, answer))
        return next(self.verdicts), self.cost


def test_run_answer_eval_aggregates(monkeypatch, tmp_path):
    emb = HashingEmbedder()
    store = MemoryStore(emb.dim)
    index_chunks([Chunk("code", "models/dagrun.py", 1, 9, "def update_state(): set run state", "DagRun.update_state", "s")], emb, store)
    monkeypatch.setattr(answers, "reference_for", lambda repo, commit: "Fix update_state")

    judge = FakeJudge([Verdict(reasoning="ok", correctness=2, faithful=True, abstained=False),
                       Verdict(reasoning="meh", correctness=1, faithful=False, abstained=False)])
    qs = [{"id": f"q{i}", "question": "How is DagRun state updated?", "gold_files": ["models/dagrun.py"],
           "commit": "c", "date": "2026-01-01"} for i in range(2)]
    llm = FakeListChatModel(responses=["It is set in update_state [1].", "No idea [4]."])
    r = run_answer_eval(qs, tmp_path, Retriever(emb, store), RetrievalConfig(), llm, "claude-haiku-5-5", judge, k=3)

    assert r["n"] == 2 and r["judge"] == "fake"
    assert r["correct (2)"] == 0.5 and r["mean_correctness"] == 1.5
    assert r["faithful"] == 0.5
    assert r["cited_gold_file"] == 0.5  # the second answer cites [4], which does not exist
    assert r["answers_with_invalid_citation"] == 0.5
    assert r["judge_cost_usd"] == pytest.approx(0.02)
    assert judge.calls[0][1] == "Fix update_state"
    assert agreement(r["per_question"], {"q0": 2, "q1": 0}) == 0.5


VERDICT = {"reasoning": "Matches the fix.", "correctness": 2, "faithful": True, "abstained": False}


def test_parse_claude_output_structured_and_text_fallback():
    assert parse_claude_output(json.dumps({"is_error": False, "structured_output": VERDICT})).correctness == 2
    text = "```json\n" + json.dumps(VERDICT) + "\n```"
    assert parse_claude_output(json.dumps({"is_error": False, "result": text})).faithful is True


def test_parse_claude_output_errors_are_clear():
    # This is what an unauthenticated CLI returns.
    with pytest.raises(RuntimeError, match="Not logged in"):
        parse_claude_output(json.dumps({"is_error": True, "result": "Not logged in · Please run /login"}))
    with pytest.raises(RuntimeError, match="non-JSON"):
        parse_claude_output("command crashed")


def test_claude_code_judge_invokes_cli_headless(monkeypatch):
    seen = {}

    def fake_run(cmd, input, **kw):
        seen["cmd"], seen["input"], seen["cwd"] = cmd, input, kw["cwd"]
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"is_error": False, "structured_output": VERDICT}), "")

    monkeypatch.setattr(answers.subprocess, "run", fake_run)
    monkeypatch.setattr(answers.shutil, "which", lambda b: None)
    verdict, cost = ClaudeCodeJudge("claude", "sonnet").grade("Q?", "REF", "SRC", "ANS")
    assert verdict.correctness == 2 and cost == 0.0
    cmd = seen["cmd"]
    assert cmd[:2] == ["claude", "-p"] and cmd[cmd.index("--model") + 1] == "sonnet"
    assert cmd[cmd.index("--tools") + 1] == ""  # no tools: the judge only reads
    schema = json.loads(cmd[cmd.index("--json-schema") + 1])
    assert set(schema["required"]) == {"reasoning", "correctness", "faithful", "abstained"}
    # The long prompt goes over stdin, not the command line.
    assert "REF" in seen["input"] and "ANS" in seen["input"] and "REF" not in " ".join(cmd)


def test_claude_code_judge_missing_binary(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(answers.subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="REPOSCOUT_CLAUDE_BIN"):
        ClaudeCodeJudge("nope").grade("q", "r", "s", "a")


def test_make_judge_picks_backend():
    j = make_judge("claude-code:opus", "C:/bin/claude.exe")
    assert isinstance(j, ClaudeCodeJudge) and j.model == "opus" and j.binary == "C:/bin/claude.exe"
    assert make_judge("claude-code").model == "sonnet"
    assert isinstance(make_judge("anthropic:claude-sonnet-5-5"), LangChainJudge)


def test_windows_cmd_wrapper_resolves_to_exe(tmp_path, monkeypatch):
    exe = tmp_path / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    monkeypatch.setattr(answers.shutil, "which", lambda b: str(tmp_path / "claude.cmd"))
    assert ClaudeCodeJudge("claude")._resolve_binary() == str(exe)


def test_cli_env_drops_host_session_vars(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "desktop")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok")
    monkeypatch.setenv("PATH_LIKE_VAR", "keep")
    env = ClaudeCodeJudge._env()
    assert "CLAUDE_CODE_ENTRYPOINT" not in env and "ANTHROPIC_API_KEY" not in env
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "tok" and env["PATH_LIKE_VAR"] == "keep"
