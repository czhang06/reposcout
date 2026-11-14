"""Turn merged pull requests into chunks, read from git history.

Airflow squash-merges PRs, so each commit on main holds the PR title, the PR number
and the description (the squashed commit list). Reading git avoids the GitHub API
rate limit. Every chunk keeps its merge date in `extra["date"]`, so evals can hide
PRs merged on or after a question's date and the fix cannot leak into its own question.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from reposcout.chunk import Chunk

MAX_BODY_CHARS = 3000
_PR = re.compile(r"\(#(\d+)\)\s*$")
_NOISE = re.compile(r"^\s*(co-authored-by|signed-off-by|claude-session|generated-by):", re.I)
RS, US = "\x1e", "\x1f"


def _clean_body(body: str) -> str:
    lines = [l for l in body.splitlines() if not _NOISE.match(l) and l.strip() != "-" * len(l.strip()) or not l.strip()]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return text[:MAX_BODY_CHARS]


def parse_git_log(output: str) -> list[dict]:
    """Parse `git log --format=RS%H US %ad US %s US %b US --name-only` output."""
    records = []
    for raw in output.split(RS)[1:]:
        parts = raw.split(US)
        if len(parts) < 5:
            continue
        sha, date, title, body, files = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3], parts[4]
        m = _PR.search(title)
        records.append({
            "sha": sha, "date": date, "title": title, "body": _clean_body(body),
            "pr": int(m.group(1)) if m else None,
            "files": [f for f in files.split() if f],
        })
    return records


def pr_chunks(records: list[dict]) -> list[Chunk]:
    chunks = []
    for r in records:
        path = f"pr/{r['pr']}" if r["pr"] else f"commit/{r['sha'][:10]}"
        files = "\n".join(f"- {f}" for f in r["files"][:30])
        text = f"{r['title']}\nMerged {r['date']}\n\n{r['body']}\n\nFiles changed:\n{files}".strip()
        chunks.append(Chunk(
            "pr", path, 1, 1, text, r["title"], r["sha"],
            {"date": r["date"], "pr": r["pr"], "files": r["files"]},
        ))
    return chunks


def read_prs(repo: Path, since: str, paths: list[str] | None = None, rev: str = "HEAD") -> list[Chunk]:
    cmd = [
        "git", "-C", str(repo), "log", rev, f"--since={since}", "--date=short", "--name-only",
        f"--format={RS}%H{US}%ad{US}%s{US}%b{US}",
    ]
    if paths:
        cmd += ["--", *paths]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return pr_chunks(parse_git_log(out))
