from reposcout.ingest.docs import chunk_doc

MD = """# Guide

Intro paragraph
about things.

## Install

Run pip install
then configure.

```python
# not a heading
x = 1
```

## Usage

Call the thing
and see results.
"""

RST = """Concepts
========

Overview text
spanning lines.

DAGs
----

A DAG is a graph
of tasks.
"""


def test_markdown_sections_and_heading_paths():
    chunks = chunk_doc("docs/guide.md", MD)
    symbols = [c.symbol for c in chunks]
    assert symbols == ["Guide", "Guide > Install", "Guide > Usage"]
    install = chunks[1]
    assert "# not a heading" in install.text  # fenced code is not split
    assert MD.splitlines()[install.start_line - 1] == "## Install"


def test_rst_headings():
    chunks = chunk_doc("docs/concepts.rst", RST)
    assert [c.symbol for c in chunks] == ["Concepts", "Concepts > DAGs"]
    assert chunks[1].start_line == 7


def test_long_section_is_split(monkeypatch):
    from reposcout.ingest import docs

    monkeypatch.setattr(docs, "MAX_CHARS", 60)
    text = "# T\n\n" + "\n\n".join(f"Paragraph number {i} with words." for i in range(6)) + "\n"
    chunks = chunk_doc("a.md", text)
    assert len(chunks) > 1
    assert all(len(c.text) <= 120 for c in chunks)
    lines = text.splitlines()
    for c in chunks:
        assert c.text.splitlines()[0] == lines[c.start_line - 1]


def test_list_without_blank_lines_is_split(monkeypatch):
    from reposcout.ingest import docs

    monkeypatch.setattr(docs, "MAX_CHARS", 100)
    text = "# Users\n\n" + "\n".join(f"1. Company number {i}" for i in range(30)) + "\n"
    chunks = chunk_doc("u.md", text)
    assert len(chunks) > 3
    assert all(len(c.text) <= 100 for c in chunks)
    lines = text.splitlines()
    for c in chunks:
        assert c.text.splitlines()[0] == lines[c.start_line - 1]
        assert c.text.splitlines()[-1] == lines[c.end_line - 1]
