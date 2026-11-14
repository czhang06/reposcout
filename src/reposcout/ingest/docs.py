"""Split Markdown and reStructuredText docs into sections by heading.

Each chunk carries its heading path (e.g. "Concepts > DAGs > Default Arguments") as
its symbol. Sections longer than MAX_CHARS are split on blank lines.
"""

from __future__ import annotations

import re

from reposcout.chunk import Chunk

MAX_CHARS = 2000
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")


def _headings_md(lines: list[str]) -> list[tuple[int, int, str]]:
    """Return (line_index, level, title) for each Markdown heading, skipping fenced code."""
    out, in_fence = [], False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        m = None if in_fence else _MD_HEADING.match(line)
        if m:
            out.append((i, len(m.group(1)), m.group(2)))
    return out


def _headings_rst(lines: list[str]) -> list[tuple[int, int, str]]:
    """RST levels come from the order in which underline characters first appear."""
    out, levels = [], {}
    i = 0
    while i < len(lines) - 1:
        title, under = lines[i].rstrip(), lines[i + 1].rstrip()
        overline = i > 0 and _RST_UNDERLINE.match(lines[i - 1].rstrip() or "") is not None
        if title.strip() and _RST_UNDERLINE.match(under) and len(under) >= len(title.strip()):
            key = (under[0], overline)
            levels.setdefault(key, len(levels) + 1)
            out.append((i, levels[key], title.strip()))
            i += 2
            continue
        i += 1
    return out


def _split_long(text: str, start_line: int) -> list[tuple[int, int, str]]:
    """Split into pieces under MAX_CHARS, preferring blank lines as cut points.

    Returns (first_line, last_line, text) with 1-based line numbers. A run of text
    with no blank lines (a long list or table) is cut at the last line that fits.
    """
    if len(text) <= MAX_CHARS:
        return [(start_line, start_line + text.count("\n"), text)]
    lines = text.split("\n")
    pieces: list[tuple[int, int, str]] = []

    def emit(lo: int, hi: int) -> None:  # lines[lo:hi], trimmed of blank edges
        while lo < hi and not lines[lo].strip():
            lo += 1
        while hi > lo and not lines[hi - 1].strip():
            hi -= 1
        if lo < hi:
            pieces.append((start_line + lo, start_line + hi - 1, "\n".join(lines[lo:hi])))

    lo, size, last_blank = 0, 0, None
    for i, line in enumerate(lines):
        if size + len(line) + 1 > MAX_CHARS and i > lo:
            cut = last_blank if last_blank is not None and last_blank > lo else i
            emit(lo, cut)
            lo = cut
            size = sum(len(l) + 1 for l in lines[lo:i])
            last_blank = None
        if not line.strip():
            last_blank = i
        size += len(line) + 1
    emit(lo, len(lines))
    return pieces


def chunk_doc(path: str, source: str, commit: str | None = None) -> list[Chunk]:
    lines = source.splitlines()
    if not lines:
        return []
    heads = _headings_rst(lines) if path.endswith(".rst") else _headings_md(lines)
    bounds = [h[0] for h in heads] + [len(lines)]

    chunks: list[Chunk] = []
    if not heads or heads[0][0] > 0:
        pre_end = heads[0][0] if heads else len(lines)
        body = "\n".join(lines[:pre_end]).strip("\n")
        if body.strip():
            first = next(i for i, l in enumerate(lines) if l.strip()) + 1
            for s, e, t in _split_long(body, first):
                chunks.append(Chunk("doc", path, s, e, t, None, commit))

    stack: list[tuple[int, str]] = []
    for (idx, level, title), next_idx in zip(heads, bounds[1:]):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        section = "\n".join(lines[idx:next_idx]).rstrip("\n")
        heading_lines = 1 if _MD_HEADING.match(lines[idx]) else 2  # rst: title + underline
        if not "\n".join(lines[idx + heading_lines : next_idx]).strip():
            continue  # section is only a heading
        heading_path = " > ".join(t for _, t in stack)
        for s, e, t in _split_long(section, idx + 1):
            chunks.append(Chunk("doc", path, s, e, t, heading_path, commit))
    return chunks
