"""Walk a git checkout and turn its Python files and docs into chunks."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

from reposcout.chunk import Chunk
from reposcout.ingest.code import chunk_python
from reposcout.ingest.docs import chunk_doc

DOC_SUFFIXES = {".md", ".rst"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "build", "dist", ".mypy_cache"}
MAX_FILE_BYTES = 1_000_000


def head_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def iter_files(root: Path, include: list[str] | None = None) -> Iterator[Path]:
    """Yield .py/.md/.rst files under root (or under each include subpath)."""
    bases = [root / p for p in include] if include else [root]
    for base in bases:
        if base.is_file():
            yield base
            continue
        for p in sorted(base.rglob("*")):
            if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
                continue
            if p.is_file() and (p.suffix == ".py" or p.suffix in DOC_SUFFIXES):
                if p.stat().st_size <= MAX_FILE_BYTES:
                    yield p


def chunk_repo(root: Path, include: list[str] | None = None) -> list[Chunk]:
    commit = head_commit(root)
    chunks: list[Chunk] = []
    for path in iter_files(root, include):
        rel = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".py":
            chunks.extend(chunk_python(rel, source, commit))
        else:
            chunks.extend(chunk_doc(rel, source, commit))
    return chunks
