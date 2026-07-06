import os
import textwrap

from codeseeker.chunking import (
    chunk_file,
    iter_source_files,
    language_for_path,
    read_source,
)

PY_SOURCE = textwrap.dedent(
    '''\
    """Module docstring."""

    import os


    def top_level(x):
        """A top level function."""
        return x + 1


    class Widget:
        """A widget class."""

        def method_one(self):
            return 1

        async def method_two(self, value):
            return value
    '''
)


def test_language_detection():
    assert language_for_path("a/b/c.py") == "python"
    assert language_for_path("main.go") == "go"
    assert language_for_path("weird.xyz") == "text"


def test_python_chunking_extracts_symbols():
    chunks = chunk_file("mod.py", PY_SOURCE)
    symbols = {c.symbol: c for c in chunks}

    assert "top_level" in symbols
    assert symbols["top_level"].kind == "function"
    assert symbols["top_level"].docstring == "A top level function."

    assert "Widget" in symbols
    assert symbols["Widget"].kind == "class"

    # Nested methods use a qualified name.
    assert "Widget.method_one" in symbols
    assert symbols["Widget.method_one"].kind == "method"
    assert "Widget.method_two" in symbols

    # Module docstring chunk should be present.
    assert any(c.kind == "module" for c in chunks)


def test_python_chunk_line_spans_are_valid():
    chunks = chunk_file("mod.py", PY_SOURCE)
    lines = PY_SOURCE.splitlines()
    for chunk in chunks:
        assert 1 <= chunk.start_line <= chunk.end_line <= len(lines)
        assert chunk.text.strip()


def test_syntax_error_falls_back_to_generic():
    broken = "def oops(:\n    pass\n"
    chunks = chunk_file("broken.py", broken)
    assert chunks
    assert all(c.kind == "block" for c in chunks)


def test_generic_chunking_for_non_python():
    js = "\n".join(f"const v{i} = {i};" for i in range(200))
    chunks = chunk_file("app.js", js)
    assert len(chunks) > 1
    assert all(c.language == "javascript" for c in chunks)
    # Windows should overlap, so consecutive chunks share lines.
    assert chunks[1].start_line < chunks[0].end_line


def test_iter_source_files_respects_excludes(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "b.py").write_text("y = 2\n")
    (tmp_path / "notes.txt").write_text("hello\n")

    found = list(iter_source_files(str(tmp_path)))
    rels = {os.path.relpath(p, tmp_path) for p in found}
    assert os.path.join("pkg", "a.py") in rels
    assert not any("node_modules" in r for r in rels)
    assert "notes.txt" not in rels


def test_read_source_missing(tmp_path):
    assert read_source(str(tmp_path / "nope.py")) is None
