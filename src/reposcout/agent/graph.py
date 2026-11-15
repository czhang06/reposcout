r"""The agent as a LangGraph state machine.

    START -> agent --(tool calls, budget left)--> tools -> agent ...
                   \--(tool calls, budget spent)--> finalize -> verify
                   \--(final text)--------------------------> verify
    verify --(problems, retries left)--> agent
    verify --(ok, or out of retries)---> END

- agent: the LLM with tools bound decides to call tools or answer.
- tools: runs each call through Toolbox (argument validation, source registry).
- finalize: when the step or token budget runs out, asks for an answer with no tools.
- verify: checks the citations before the answer is returned (see check_answer).

Guardrails: a maximum number of tool rounds, a maximum number of tool calls, a cap on
input tokens, and a wall-clock timeout.
"""

from __future__ import annotations

import operator
import re
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from reposcout.agent.tools import TOOL_SPECS, SourceRegistry, ToolError, Toolbox
from reposcout.config import cost_usd
from reposcout.llm import message_text, token_usage

SYSTEM_PROMPT = """You answer developer questions about the Apache Airflow codebase by \
searching and reading it.

How to work:
- Start with search_code; use search_docs for concepts and search_prs for why code \
changed. If a search result is cut off or you need surrounding lines, use read_file. \
Use find_symbol to locate a definition by name.
- Make focused searches with concrete words and identifiers. Stop as soon as you have \
enough to answer; you have at most {max_rounds} rounds of tool calls.

How to answer:
- Every tool result is labelled with a source id like [S3]. After each claim, cite the \
source ids that support it, e.g. "The run is set back to QUEUED [S2]." Cite only ids \
that appeared in tool results.
- Put code identifiers in backticks, e.g. `clear_task_instances`; each one you mention \
must appear in a source you cite in that sentence.
- State only what the sources show. Do not infer causes, motives or behavior the code \
you read does not show; if the question asks why something happens and the sources do \
not say, read more or say it is not shown.
- If the code you found does not answer the question, say what you found and what is \
missing. Do not guess.
- Be short and concrete."""

FINALIZE_PROMPT = (
    "The tool budget for this question is used up. Answer now using only the sources "
    "above, citing their [S#] ids. If they are not enough, say what is missing."
)

_SID = re.compile(r"\[(S\d+)\]")
_BACKTICK = re.compile(r"`([^`\n]{2,80})`")
_ABSTAIN = re.compile(
    r"(could not find|couldn't find|do(?:es)? not (?:contain|show|answer)|not enough information|"
    r"no (?:relevant )?(?:code|source)s? (?:found|that))", re.I,
)


@dataclass
class Budget:
    max_rounds: int = 6
    max_tool_calls: int = 15
    max_input_tokens: int = 120_000
    timeout_s: float = 120.0
    max_verify_retries: int = 1


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    rounds: int
    tool_calls: int
    verify_retries: int
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    issues: list[str]
    stop_reason: str


@dataclass
class AgentResult:
    question: str
    answer: str
    citations: list[dict]
    issues: list[str]  # problems left after verification
    rounds: int
    tool_calls: int
    tools_used: list[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    stop_reason: str
    trace: list[dict] = field(repr=False, default_factory=list)


def check_answer(text: str, registry: SourceRegistry) -> list[str]:
    """Problems with an answer's citations. An empty list means it passed.

    1. Every [S#] must be an id a tool returned.
    2. A non-abstaining answer must cite at least one source.
    3. Each `identifier` in a cited sentence must appear in one of that sentence's
       cited sources, which catches invented function and class names.
    """
    issues = []
    cited = _SID.findall(text)
    unknown = sorted({s for s in cited if s not in registry.sources}, key=lambda s: int(s[1:]))
    if unknown:
        issues.append(f"cites sources that no tool returned: {', '.join(unknown)}")
    if not cited and not _ABSTAIN.search(text):
        issues.append("makes claims without citing any [S#] source")
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        ids = [s for s in _SID.findall(sentence) if s in registry.sources]
        if not ids:
            continue
        support = "\n".join(registry.sources[s].text for s in ids)
        for ident in _BACKTICK.findall(sentence):
            name = ident.strip().rstrip("()")
            last = re.split(r"[.:]", name)[-1]
            if name not in support and last not in support:
                issues.append(f"`{ident}` is not in its cited sources ({', '.join(ids)})")
    return issues


def _lc_tools() -> list[StructuredTool]:
    """Tool definitions for bind_tools. Execution happens in the tools node, not here."""
    def _unused(**_: Any) -> str:  # pragma: no cover - never called
        raise RuntimeError("tools run in the graph's tools node")

    return [StructuredTool.from_function(_unused, name=n, description=d, args_schema=s) for n, (s, d) in TOOL_SPECS.items()]


class RepoAgent:
    def __init__(self, llm: BaseChatModel, model: str, budget: Budget | None = None):
        self.llm = llm
        self.model = model
        self.budget = budget or Budget()
        tools = _lc_tools()
        self.llm_with_tools = llm.bind_tools(tools)
        # For the forced final answer. Claude ties its thinking blocks to the exact
        # tools list they were made with, so dropping the tools there is a 400. Keep
        # the same tools bound and forbid calling them with tool_choice "none".
        if type(llm).__name__ == "ChatAnthropic":
            self.llm_final = llm.bind_tools(tools, tool_choice={"type": "none"})
        else:
            self.llm_final = llm

    def build(self, toolbox: Toolbox, trace: list[dict], t0: float):
        b = self.budget
        system = SystemMessage(SYSTEM_PROMPT.format(max_rounds=b.max_rounds))

        def log(node: str, **kw):
            trace.append({"t_ms": round((time.perf_counter() - t0) * 1000, 1), "node": node, **kw})

        def call_llm(llm, messages):
            t = time.perf_counter()
            ai = llm.invoke(messages)
            tin, tout = token_usage(ai)
            return ai, tin, tout, (time.perf_counter() - t) * 1000

        def agent(state: AgentState):
            ai, tin, tout, ms = call_llm(self.llm_with_tools, [system, *state["messages"]])
            log("agent", ms=round(ms), input_tokens=tin, output_tokens=tout,
                tool_calls=[{"name": c["name"], "args": c["args"]} for c in ai.tool_calls])
            return {"messages": [ai], "input_tokens": tin, "output_tokens": tout}

        def over_budget(state: AgentState) -> str | None:
            if state["rounds"] >= b.max_rounds:
                return "max_rounds"
            if state["tool_calls"] >= b.max_tool_calls:
                return "max_tool_calls"
            if state["input_tokens"] >= b.max_input_tokens:
                return "max_input_tokens"
            if time.perf_counter() - t0 >= b.timeout_s:
                return "timeout"
            return None

        def route_agent(state: AgentState) -> str:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "finalize" if over_budget(state) else "tools"
            return "verify"

        def tools(state: AgentState):
            calls = state["messages"][-1].tool_calls
            room = b.max_tool_calls - state["tool_calls"]
            out = []
            for i, call in enumerate(calls):
                t = time.perf_counter()
                if i >= room:
                    content, n, err = "Not run: the tool-call limit for this question was reached.", 0, "limit"
                else:
                    try:
                        content, n = toolbox.run(call["name"], call["args"])
                        err = None
                    except ToolError as e:
                        content, n, err = f"Error: {e}", 0, str(e)
                out.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
                log("tool", tool=call["name"], args=call["args"], results=n, error=err,
                    ms=round((time.perf_counter() - t) * 1000, 1))
            return {"messages": out, "rounds": state["rounds"] + 1,
                    "tool_calls": state["tool_calls"] + min(len(calls), max(room, 0))}

        def finalize(state: AgentState):
            reason = over_budget(state) or "budget"
            last = state["messages"][-1]
            # Every tool_use needs a tool_result before the conversation can continue.
            skipped = [ToolMessage(content="Not run: budget used up.", tool_call_id=c["id"], name=c["name"])
                       for c in getattr(last, "tool_calls", [])]
            msgs = [system, *state["messages"], *skipped, HumanMessage(FINALIZE_PROMPT)]
            ai, tin, tout, ms = call_llm(self.llm_final, msgs)
            if ai.tool_calls:  # a model that ignores tool_choice: keep only its text
                ai = AIMessage(content=message_text(ai), usage_metadata=ai.usage_metadata)
            log("finalize", reason=reason, ms=round(ms), input_tokens=tin, output_tokens=tout)
            return {"messages": [*skipped, HumanMessage(FINALIZE_PROMPT), ai],
                    "input_tokens": tin, "output_tokens": tout, "stop_reason": reason}

        def verify(state: AgentState):
            issues = check_answer(message_text(state["messages"][-1]), toolbox.registry)
            log("verify", issues=issues)
            if issues and state["verify_retries"] < b.max_verify_retries and not state.get("stop_reason"):
                feedback = ("Your answer failed a citation check:\n- " + "\n- ".join(issues) +
                            "\nFix it: cite only [S#] ids from tool results, and only name identifiers "
                            "that appear in the sources you cite. You may search or read more first. "
                            "Then reply with the full answer only, as if answering for the first time.")
                return {"messages": [HumanMessage(feedback)], "verify_retries": state["verify_retries"] + 1,
                        "issues": issues}
            return {"issues": issues, "stop_reason": state.get("stop_reason") or ("answered" if not issues else "unverified")}

        def route_verify(state: AgentState) -> str:
            last = state["messages"][-1]
            return "agent" if isinstance(last, HumanMessage) else END

        g = StateGraph(AgentState)
        g.add_node("agent", agent)
        g.add_node("tools", tools)
        g.add_node("finalize", finalize)
        g.add_node("verify", verify)
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", route_agent, ["tools", "finalize", "verify"])
        g.add_edge("tools", "agent")
        g.add_edge("finalize", "verify")
        g.add_conditional_edges("verify", route_verify, ["agent", END])
        return g.compile()

    def run(self, question: str, toolbox: Toolbox) -> AgentResult:
        t0 = time.perf_counter()
        trace: list[dict] = []
        graph = self.build(toolbox, trace, t0)
        init: AgentState = {"messages": [HumanMessage(question)], "rounds": 0, "tool_calls": 0, "verify_retries": 0,
                            "input_tokens": 0, "output_tokens": 0, "issues": [], "stop_reason": ""}
        # Each round is agent + tools; leave room for finalize, verify and one retry.
        state = graph.invoke(init, {"recursion_limit": 4 * self.budget.max_rounds + 12})

        text = message_text(state["messages"][-1])
        reg = toolbox.registry
        cited = list(dict.fromkeys(s for s in _SID.findall(text) if s in reg.sources))
        return AgentResult(
            question=question,
            answer=text,
            citations=[{"sid": s, "location": reg.sources[s].location, "url": reg.sources[s].url,
                        "kind": reg.sources[s].kind, "path": reg.sources[s].path} for s in cited],
            issues=state["issues"],
            rounds=state["rounds"],
            tool_calls=state["tool_calls"],
            tools_used=[e["tool"] for e in trace if e["node"] == "tool"],
            input_tokens=state["input_tokens"],
            output_tokens=state["output_tokens"],
            cost_usd=cost_usd(self.model, state["input_tokens"], state["output_tokens"]),
            latency_ms=(time.perf_counter() - t0) * 1000,
            stop_reason=state["stop_reason"],
            trace=trace,
        )
