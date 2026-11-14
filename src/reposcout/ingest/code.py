"""Split Python source into one chunk per function or class, using the stdlib ast module.

Large classes are split further: a header chunk (class line, docstring, attributes)
and one chunk per method, so a single retrieval hit points at the method that matters.
Code that sits at module level (imports, constants, the module docstring) becomes
one "module" chunk. Files that do not parse fall back to fixed line windows.
"""

from __future__ import annotations

import ast

from reposcout.chunk import Chunk

MAX_CLASS_LINES = 120
# Embedding models read ~512 tokens (~50 lines of code), so long functions are
# windowed with overlap; each window repeats the def line for context.
MAX_DEF_LINES = 80
WINDOW_LINES = 60
WINDOW_OVERLAP = 10


def _span(node: ast.AST) -> tuple[int, int]:
    start = node.lineno
    decorators = getattr(node, "decorator_list", None) or []
    if decorators:
        start = min(start, min(d.lineno for d in decorators))
    return start, node.end_lineno or node.lineno


def _text(lines: list[str], start: int, end: int) -> str:
    return "".join(lines[start - 1 : end]).rstrip("\n")


def _windows(path: str, lines: list[str], commit: str | None) -> list[Chunk]:
    chunks = []
    for start in range(1, len(lines) + 1, WINDOW_LINES):
        end = min(start + WINDOW_LINES - 1, len(lines))
        text = _text(lines, start, end)
        if text.strip():
            chunks.append(Chunk("code", path, start, end, text, None, commit))
    return chunks


def _def_chunks(
    path: str, lines: list[str], node: ast.AST, symbol: str, commit: str | None, context: str = ""
) -> list[Chunk]:
    """One chunk for a short def; overlapping windows for a long one."""
    start, end = _span(node)
    if end - start + 1 <= MAX_DEF_LINES:
        return [Chunk("code", path, start, end, context + _text(lines, start, end), symbol, commit)]
    def_line = lines[node.lineno - 1].rstrip("\n")
    out, s = [], start
    while s <= end:
        e = min(s + WINDOW_LINES - 1, end)
        head = context if s == start else f"{context}{def_line}\n    ...\n"
        last = e
        while last > s and not lines[last - 1].strip():
            last -= 1
        out.append(Chunk("code", path, s, last, head + _text(lines, s, last), symbol, commit, {"part": len(out) + 1}))
        if e == end:
            break
        s = e + 1 - WINDOW_OVERLAP
    return out


def chunk_python(path: str, source: str, commit: str | None = None) -> list[Chunk]:
    lines = source.splitlines(keepends=True)
    if not lines:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _windows(path, lines, commit)

    chunks: list[Chunk] = []
    covered: set[int] = set()
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    for node in tree.body:
        if not isinstance(node, defs):
            continue
        start, end = _span(node)
        covered.update(range(start, end + 1))

        if isinstance(node, ast.ClassDef) and end - start + 1 > MAX_CLASS_LINES:
            methods = [n for n in node.body if isinstance(n, defs)]
            first_method_start = min((_span(m)[0] for m in methods), default=end + 1)
            header_end = max(start, first_method_start - 1)
            while header_end > start and not lines[header_end - 1].strip():
                header_end -= 1
            chunks.append(
                Chunk("code", path, start, header_end, _text(lines, start, header_end), node.name, commit)
            )
            # Prefix the class line so each method chunk keeps its context.
            context = lines[node.lineno - 1].rstrip("\n") + "\n    ...\n"
            for m in methods:
                chunks.extend(_def_chunks(path, lines, m, f"{node.name}.{m.name}", commit, context))
        else:
            chunks.extend(_def_chunks(path, lines, node, node.name, commit))

    # Everything not inside a top-level def/class (imports, constants, docstring),
    # as contiguous runs of lines between defs, windowed if long.
    run_start = None
    for i in range(1, len(lines) + 2):
        inside = i <= len(lines) and i not in covered
        if inside and run_start is None:
            run_start = i
        elif not inside and run_start is not None:
            for s in range(run_start, i, WINDOW_LINES):
                e = min(s + WINDOW_LINES - 1, i - 1)
                text = _text(lines, s, e).strip("\n")
                if text.strip():
                    first = s + next(n for n, l in enumerate(lines[s - 1 : e]) if l.strip())
                    last = e - next(n for n, l in enumerate(reversed(lines[s - 1 : e])) if l.strip())
                    chunks.append(Chunk("code", path, first, last, _text(lines, first, last), "<module>", commit))
            run_start = None

    return sorted(chunks, key=lambda c: (c.start_line, c.end_line))
