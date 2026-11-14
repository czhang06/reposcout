from reposcout.ingest.prs import RS, US, parse_git_log, pr_chunks


def _record(sha, date, title, body, files):
    return f"{RS}{sha}{US}{date}{US}{title}{US}{body}{US}\n\n" + "\n".join(files) + "\n"


def test_parse_and_chunk_prs():
    out = _record("a" * 40, "2026-01-02", "Fix N+1 in notes (#123)",
                  "* Eager-load notes\n\nCo-authored-by: X <x@y>\n---------\n", ["airflow/models/taskinstance.py"])
    out += _record("b" * 40, "2026-01-03", "Direct commit without PR", "", ["README.md"])
    recs = parse_git_log(out)
    assert [r["pr"] for r in recs] == [123, None]
    assert "Co-authored-by" not in recs[0]["body"] and "Eager-load notes" in recs[0]["body"]
    assert recs[0]["files"] == ["airflow/models/taskinstance.py"]
    chunks = pr_chunks(recs)
    assert chunks[0].path == "pr/123" and chunks[0].citation == "PR #123"
    assert chunks[0].extra["date"] == "2026-01-02"
    assert "airflow/models/taskinstance.py" in chunks[0].text
    assert chunks[1].path == "commit/bbbbbbbbbb"
