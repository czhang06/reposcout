from reposcout.ingest import code
from reposcout.ingest.code import chunk_python

SRC = '''"""Module docstring."""
import os

X = 1


@decorator
def top(a):
    return a


class Small:
    def m(self):
        pass
'''


def test_one_chunk_per_def_plus_module_chunk():
    chunks = chunk_python("pkg/mod.py", SRC, "abc")
    by_symbol = {c.symbol: c for c in chunks}
    assert set(by_symbol) == {"<module>", "top", "Small"}
    # decorator line is included in the function's span
    assert by_symbol["top"].start_line == 7
    assert by_symbol["top"].text.startswith("@decorator")
    assert by_symbol["Small"].start_line == 12 and by_symbol["Small"].end_line == 14
    mod = by_symbol["<module>"]
    assert "import os" in mod.text and "def top" not in mod.text
    assert all(c.commit == "abc" and c.path == "pkg/mod.py" for c in chunks)


def test_line_numbers_match_source():
    lines = SRC.splitlines()
    for c in chunk_python("m.py", SRC):
        if c.symbol != "<module>":
            assert c.text.splitlines()[0] == lines[c.start_line - 1]
            assert c.text.splitlines()[-1] == lines[c.end_line - 1]


def test_large_class_split_into_methods(monkeypatch):
    monkeypatch.setattr(code, "MAX_CLASS_LINES", 5)
    src = "class Big:\n    '''doc'''\n    attr = 1\n\n    def a(self):\n        return 1\n\n    def b(self):\n        return 2\n"
    chunks = chunk_python("big.py", src)
    symbols = [c.symbol for c in chunks]
    assert symbols == ["Big", "Big.a", "Big.b"]
    method_a = chunks[1]
    assert method_a.start_line == 5 and method_a.end_line == 6
    assert method_a.text.startswith("class Big:")  # class context kept


def test_syntax_error_falls_back_to_windows():
    chunks = chunk_python("bad.py", "def broken(:\n    pass\n")
    assert len(chunks) == 1 and chunks[0].symbol is None


def test_chunk_id_stable_and_content_sensitive():
    a = chunk_python("m.py", SRC)
    b = chunk_python("m.py", SRC)
    assert [c.id for c in a] == [c.id for c in b]
    changed = chunk_python("m.py", SRC.replace("return a", "return a + 1"))
    assert {c.id for c in a} != {c.id for c in changed}


def test_long_function_is_windowed(monkeypatch):
    monkeypatch.setattr(code, "MAX_DEF_LINES", 10)
    monkeypatch.setattr(code, "WINDOW_LINES", 6)
    monkeypatch.setattr(code, "WINDOW_OVERLAP", 2)
    body = "".join(f"    x{i} = {i}\n" for i in range(20))
    src = "def long_fn():\n" + body
    chunks = chunk_python("l.py", src)
    assert len(chunks) > 1
    assert all(c.symbol == "long_fn" for c in chunks)
    assert chunks[0].start_line == 1 and chunks[-1].end_line == 21
    # windows overlap and cover every line
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_line == a.end_line + 1 - 2
    assert chunks[1].text.startswith("def long_fn():")
