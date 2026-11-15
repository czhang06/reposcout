"""Agent tests with a scripted chat model: the graph, guardrails and verify step run for
real; only the LLM's replies are fixed in advance."""

from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from reposcout.agent.graph import Budget, RepoAgent, check_answer
from reposcout.agent.tools import SourceRegistry, ToolError, Toolbox
from reposcout.chunk import Chunk
from reposcout.embed import HashingEmbedder
from reposcout.ingest.pipeline import index_chunks
from reposcout.retrieval.retriever import RetrievalConfig, Retriever
from reposcout.store.memory import MemoryStore

FIXTURE = Path(__file__).parent / "fixtures" / "minirepo"


class ScriptedModel(BaseChatModel):
    """Returns pre-written AIMessages in order and records what it was sent."""

    script: list
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        reply = self.script.pop(0)
        reply.usage_metadata = {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050}
        return ChatResult(generations=[ChatGeneration(message=reply)])


def call(name, args, i):
    return {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}


def toolbox():
    emb = HashingEmbedder()
    store = MemoryStore(emb.dim)
    chunks = [
        Chunk("code", "pkg/dag.py", 5, 15, "class DAG:\n    def add_task(self, task):\n        self.tasks.append(task)", "DAG", "abc"),
        Chunk("code", "pkg/dag.py", 18, 19, "def get_dag(dag_id: str) -> DAG:\n    return DAG(dag_id)", "get_dag", "abc"),
        Chunk("pr", "pr/7", 1, 1, "Fix add_task ordering", "Fix add_task ordering", "c7", {"date": "2026-02-01", "pr": 7}),
    ]
    index_chunks(chunks, emb, store)
    code = [c for c in chunks if c.source_type == "code"]
    return Toolbox(Retriever(emb, store), RetrievalConfig(), FIXTURE, SourceRegistry(), code)


def run(script, budget=None, tb=None):
    model = ScriptedModel(script=script, seen=[])
    tb = tb or toolbox()
    return RepoAgent(model, "claude-haiku-5-5", budget).run("How do I add a task to a DAG?", tb), model, tb


def test_search_read_then_cited_answer():
    r, model, tb = run([
        AIMessage("", tool_calls=[call("search_code", {"query": "add task to dag"}, 1)]),
        AIMessage("", tool_calls=[call("read_file", {"path": "pkg/dag.py", "start_line": 5, "end_line": 15}, 2)]),
        AIMessage("Call `add_task` on the `DAG`; it appends to `self.tasks` [S1]."),
    ])
    assert r.stop_reason == "answered" and r.issues == []
    assert r.rounds == 2 and r.tool_calls == 2 and r.tools_used == ["search_code", "read_file"]
    assert [c["location"] for c in r.citations] == ["pkg/dag.py:5-15"]
    assert r.citations[0]["url"].endswith("pkg/dag.py#L5-L15")
    # read_file on a location the search already returned keeps its id (S1) but now
    # holds the real file lines
    assert len(tb.registry.sources) == 2
    assert "self.dag_id = dag_id" in tb.registry.sources["S1"].text
    assert r.input_tokens == 3000 and r.cost_usd == pytest.approx((3000 * 0.10 + 150 * 0.50) / 1e6)
    # tool results went back to the model as ToolMessages tied to the call ids
    tool_msgs = [m for m in model.seen[1] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].tool_call_id == "call_1" and "[S1]" in tool_msgs[0].content


def test_verify_sends_bad_citations_back_once():
    r, model, _ = run([
        AIMessage("", tool_calls=[call("search_code", {"query": "add task"}, 1)]),
        AIMessage("Use `add_task` [S1] or `insert_task` [S9]."),
        AIMessage("Use `add_task` on the DAG [S1]."),
    ])
    assert r.stop_reason == "answered" and r.issues == []
    feedback = [m for m in model.seen[2] if isinstance(m, HumanMessage)][-1].content
    assert "S9" in feedback and "insert_task" in feedback
    assert [e["node"] for e in r.trace].count("verify") == 2


def test_unfixed_answer_is_returned_with_issues():
    r, _, _ = run([
        AIMessage("It uses `insert_task`."),
        AIMessage("Still `insert_task` [S4]."),
    ])
    assert r.stop_reason == "unverified"
    assert any("S4" in i for i in r.issues)


def test_round_budget_forces_final_answer():
    loop = [AIMessage("", tool_calls=[call("search_code", {"query": f"dag task {i}"}, i)]) for i in range(2)]
    final = AIMessage("Use `add_task` [S1].")
    r, model, _ = run(loop + [final], budget=Budget(max_rounds=1))
    assert r.stop_reason == "max_rounds" and r.rounds == 1
    last_call = model.seen[-1]
    # the unexecuted tool call got a result, so the conversation stays valid
    assert isinstance(last_call[-2], ToolMessage) and "budget" in last_call[-2].content
    assert "budget" in last_call[-1].content.lower()


def test_tool_call_limit_within_a_round():
    calls = [call("search_code", {"query": f"dag {i}"}, i) for i in range(4)]
    r, model, _ = run([AIMessage("", tool_calls=calls), AIMessage("Use the add_task method [S1].")], budget=Budget(max_tool_calls=2))
    assert r.tool_calls == 2
    errors = [e for e in r.trace if e["node"] == "tool" and e["error"] == "limit"]
    assert len(errors) == 2


def test_invalid_tool_args_return_an_error_to_the_model():
    r, model, _ = run([
        AIMessage("", tool_calls=[call("read_file", {"path": "../secrets.txt", "start_line": 1, "end_line": 5}, 1),
                                  call("search_code", {"query": "x"}, 2)]),
        AIMessage("I could not find it."),
    ])
    msgs = [m.content for m in model.seen[1] if isinstance(m, ToolMessage)]
    assert msgs[0].startswith("Error: invalid arguments for read_file") and "'..'" in msgs[0]
    assert msgs[1].startswith("Error: invalid arguments for search_code")
    assert r.stop_reason == "answered"  # abstaining without citations passes


def test_tools_directly():
    tb = toolbox()
    text, n = tb.run("find_symbol", {"name": "DAG.add_task"})
    assert n == 1 and "Definitions" in text and "pkg/dag.py:5-15" in text  # method inside the class chunk
    text, n = tb.run("find_symbol", {"name": "DAG"})
    assert "Used in" in text and "pkg/dag.py:18-19" in text  # get_dag uses DAG
    text, n = tb.run("search_prs", {"query": "add_task ordering"})
    assert n == 1 and "PR #7" in text
    with pytest.raises(ToolError, match="no such file"):
        tb.run("read_file", {"path": "pkg/missing.py", "start_line": 1, "end_line": 2})
    with pytest.raises(ToolError, match="unknown tool"):
        tb.run("delete_repo", {})
    text, _ = tb.run("read_file", {"path": "pkg/dag.py", "start_line": 17, "end_line": 999})
    assert "def get_dag" in text and "pkg/dag.py:17-18" in text  # clipped to the file


def test_prs_respect_date_cutoff():
    tb = toolbox()
    tb.cfg = RetrievalConfig(before_date="2026-01-15")
    text, n = tb.run("search_prs", {"query": "add_task ordering"})
    assert n == 0


def test_check_answer_rules():
    reg = SourceRegistry()
    reg.add(Chunk("code", "a.py", 1, 3, "def clear_task_instances(tis): pass", "clear_task_instances"))
    assert check_answer("Use `clear_task_instances` [S1].", reg) == []
    assert check_answer("Use `models.clear_task_instances()` [S1].", reg) == []  # dotted, with parens
    assert "no tool returned" in check_answer("See [S2].", reg)[0]
    assert "without citing" in check_answer("It clears tasks.", reg)[0]
    assert check_answer("The sources do not contain the answer.", reg) == []
    assert "`reset_dag_run` is not in its cited sources" in check_answer("It calls `reset_dag_run` [S1].", reg)[0]


def test_read_file_is_not_truncated_and_judge_sees_what_agent_saw(monkeypatch):
    from reposcout.agent import tools

    monkeypatch.setattr(tools, "MAX_SOURCE_CHARS", 40)
    tb = toolbox()
    search_text, _ = tb.run("search_code", {"query": "add task to dag", "k": 1})
    assert "(truncated" in search_text
    src = next(iter(tb.registry.sources.values()))
    assert len(src.seen_text()) == 40  # the judge sees only the truncated part
    read_text, _ = tb.run("read_file", {"path": "pkg/dag.py", "start_line": 1, "end_line": 19})
    assert "(truncated" not in read_text and "def get_dag" in read_text
    seen = tb.registry.render_seen()
    assert "def get_dag" in seen  # full read_file output is what the judge gets
