"""Tools the agent can call, and the registry that makes its citations checkable.

Every piece of text a tool returns is registered as a source with an id like S3 and
an exact location (file and line range, or PR number). The agent may only cite those
ids, so the verify step can check each citation against text the agent really saw.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

from reposcout.chunk import Chunk
from reposcout.rag import github_url
from reposcout.retrieval.retriever import RetrievalConfig, Retriever
from reposcout.store import Hit

MAX_SOURCE_CHARS = 1800
MAX_READ_LINES = 150


@dataclass
class Source:
    sid: str  # "S1", "S2", ...
    kind: str  # code | doc | pr
    location: str  # "path:start-end" or "PR #123"
    text: str
    url: str | None
    path: str
    shown_chars: int = 0  # how much of `text` the agent has been shown, at most

    def seen_text(self) -> str:
        """The part of the source the agent actually saw, for grading faithfulness."""
        return self.text[: self.shown_chars]


@dataclass
class SourceRegistry:
    repo: str = "apache/airflow"
    sources: dict[str, Source] = field(default_factory=dict)
    _by_location: dict[str, str] = field(default_factory=dict)

    def add(self, chunk: Chunk) -> Source:
        loc = chunk.citation
        if loc in self._by_location:
            # Same place seen again (e.g. read_file on a search hit): keep the id, but
            # take the newest text, which for read_file is the file itself.
            src = self.sources[self._by_location[loc]]
            src.text = chunk.text
            return src
        sid = f"S{len(self.sources) + 1}"
        src = Source(sid, chunk.source_type, loc, chunk.text, github_url(self.repo, Hit(chunk, 0.0)), chunk.path)
        self.sources[sid] = src
        self._by_location[loc] = sid
        return src

    def render(self, src: Source, symbol: str | None = None, full: bool = False) -> str:
        """Text for the model. Search results are cut to MAX_SOURCE_CHARS; read_file
        output (full=True) is already bounded by MAX_READ_LINES and is shown whole."""
        shown = len(src.text) if full else min(len(src.text), MAX_SOURCE_CHARS)
        text = src.text[:shown]
        if shown < len(src.text):
            text += "\n... (truncated; use read_file for more)"
        src.shown_chars = max(src.shown_chars, shown)
        label = f"[{src.sid}] {src.kind} {src.location}" + (f" ({symbol})" if symbol else "")
        return f"{label}\n```\n{text}\n```"

    def render_seen(self) -> str:
        """Every source as the agent saw it. This is what the judge grades against."""
        return "\n\n".join(
            f"[{s.sid}] {s.kind} {s.location}\n```\n{s.seen_text()}\n```" for s in self.sources.values()
        )


# ---- Argument schemas: the LLM sees these, and every call is validated against them.

class SearchArgs(BaseModel):
    query: str = Field(min_length=3, max_length=300, description="What to look for, in words or identifiers.")
    k: int = Field(default=5, ge=1, le=10, description="How many results to return.")


class ReadFileArgs(BaseModel):
    path: str = Field(description="Repo-relative path, e.g. airflow-core/src/airflow/models/dagrun.py")
    start_line: int = Field(ge=1, description="First line to read (1-based).")
    end_line: int = Field(ge=1, description=f"Last line to read; at most {MAX_READ_LINES} lines per call.")

    @field_validator("path")
    @classmethod
    def relative_path(cls, v: str) -> str:
        v = v.replace("\\", "/").strip()
        if v.startswith("/") or re.match(r"^[A-Za-z]:", v) or ".." in v.split("/"):
            raise ValueError("path must be relative to the repository root, without '..'")
        return v


class FindSymbolArgs(BaseModel):
    name: str = Field(min_length=2, max_length=120, pattern=r"^[A-Za-z_][\w.]*$",
                      description="Function, method or class name, e.g. clear_task_instances or DagRun.update_state.")


TOOL_SPECS = {
    "search_code": (SearchArgs, "Hybrid (keyword + semantic) search over Airflow source code. Returns functions and classes with their file and line range."),
    "search_docs": (SearchArgs, "Search Airflow's documentation (concepts, how-tos)."),
    "search_prs": (SearchArgs, "Search merged pull requests: their titles, descriptions and changed files. Good for why something behaves the way it does."),
    "read_file": (ReadFileArgs, "Read exact lines of a file, e.g. to see the rest of a function a search returned."),
    "find_symbol": (FindSymbolArgs, "Find where a function, method or class is defined, plus places that use it."),
}


class ToolError(Exception):
    pass


class Toolbox:
    """Runs tool calls against the index and the checked-out repo, registering sources."""

    def __init__(self, retriever: Retriever, cfg: RetrievalConfig, repo_root: Path | None, registry: SourceRegistry,
                 code_chunks: list[Chunk] | None = None):
        self.retriever = retriever
        self.cfg = cfg
        self.repo_root = repo_root.resolve() if repo_root else None
        self.registry = registry
        self._code_chunks = code_chunks
        self._commit = next((c.commit for c in (code_chunks or []) if c.commit), None)

    # -- dispatch

    def run(self, name: str, args: dict) -> tuple[str, int]:
        """Validate and run one call. Returns (text for the model, number of sources returned)."""
        if name not in TOOL_SPECS:
            raise ToolError(f"unknown tool {name!r}; available: {', '.join(TOOL_SPECS)}")
        schema = TOOL_SPECS[name][0]
        try:
            parsed = schema.model_validate(args)
        except ValidationError as e:
            msgs = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            raise ToolError(f"invalid arguments for {name}: {msgs}") from None
        return getattr(self, name)(parsed)

    # -- tools

    def _search(self, args: SearchArgs, source_type: str) -> tuple[str, int]:
        hits = self.retriever.retrieve(args.query, args.k, replace(self.cfg, source_types=[source_type])).hits
        if not hits:
            return f"No {source_type} results for {args.query!r}.", 0
        return "\n\n".join(self.registry.render(self.registry.add(h.chunk), h.chunk.symbol) for h in hits), len(hits)

    def search_code(self, args: SearchArgs) -> tuple[str, int]:
        return self._search(args, "code")

    def search_docs(self, args: SearchArgs) -> tuple[str, int]:
        return self._search(args, "doc")

    def search_prs(self, args: SearchArgs) -> tuple[str, int]:
        return self._search(args, "pr")

    def read_file(self, args: ReadFileArgs) -> tuple[str, int]:
        if self.repo_root is None:
            raise ToolError("read_file is unavailable: no repository checkout configured")
        path = (self.repo_root / args.path).resolve()
        if self.repo_root not in path.parents or not path.is_file():
            raise ToolError(f"no such file in the indexed checkout: {args.path}")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = min(args.start_line, len(lines))
        end = min(max(args.end_line, start), len(lines), start + MAX_READ_LINES - 1)
        text = "\n".join(lines[start - 1 : end])
        kind = "code" if path.suffix == ".py" else "doc"
        chunk = Chunk(kind, args.path, start, end, text, None, self._commit)
        note = "" if end - start + 1 < MAX_READ_LINES else f"\n(stopped at {MAX_READ_LINES} lines; the file has {len(lines)})"
        return self.registry.render(self.registry.add(chunk), full=True) + note, 1

    def find_symbol(self, args: FindSymbolArgs) -> tuple[str, int]:
        chunks = self._code_chunks or []
        short = args.name.rsplit(".", 1)[-1]
        # A definition is a chunk named after the symbol, or one whose text defines it
        # (methods of small classes live inside the class's chunk).
        defines = re.compile(rf"^\s*(?:async\s+def|def|class)\s+{re.escape(short)}\b", re.M)
        defs = [c for c in chunks if (c.symbol and (c.symbol == args.name or c.symbol.endswith("." + short)
                                                     or c.symbol == short)) or defines.search(c.text)]
        # Prefer the first window of long, windowed definitions.
        defs = sorted(defs, key=lambda c: (c.extra.get("part", 1), c.path, c.start_line))[:4]
        pattern = re.compile(rf"\b{re.escape(short)}\b")
        def_ids = {c.id for c in defs}
        uses = [c for c in chunks if c.id not in def_ids and pattern.search(c.text)][:10]
        if not defs and not uses:
            return f"No definition or use of {args.name!r} in the index.", 0
        parts = []
        if defs:
            parts.append("Definitions:\n\n" + "\n\n".join(self.registry.render(self.registry.add(c), c.symbol) for c in defs))
        if uses:
            parts.append("Used in (read_file to see more):\n" + "\n".join(f"- {c.citation} ({c.symbol})" for c in uses))
        return "\n\n".join(parts), len(defs)
