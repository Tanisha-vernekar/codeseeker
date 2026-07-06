"""Split source files into meaningful, searchable code chunks.

Python files are parsed with the :mod:`ast` module so that functions and
classes become individual chunks (carrying their qualified name and any
docstring). Every other language falls back to a line-window chunker that
keeps a small overlap between windows so context is not lost at boundaries.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Iterable, Iterator

# Map file extensions to a human friendly language label.
LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".sql": "sql",
}

DEFAULT_EXTENSIONS: tuple[str, ...] = tuple(LANGUAGE_BY_EXT.keys())

# Directories that almost never contain source worth indexing.
DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".idea",
        ".vscode",
        ".tox",
        ".eggs",
        "site-packages",
    }
)

# Reasonable ceiling so we never try to embed a giant generated/minified file.
MAX_FILE_BYTES = 1_000_000


@dataclass(frozen=True)
class CodeChunk:
    """A contiguous, searchable region of a source file."""

    path: str
    language: str
    kind: str  # "function", "class", "method", "module", or "block"
    symbol: str  # qualified name (e.g. "MyClass.method") or "" for blocks
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    text: str
    docstring: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "language": self.language,
            "kind": self.kind,
            "symbol": self.symbol,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
            "docstring": self.docstring,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CodeChunk":
        return cls(
            path=data["path"],
            language=data["language"],
            kind=data["kind"],
            symbol=data.get("symbol", ""),
            start_line=int(data["start_line"]),
            end_line=int(data["end_line"]),
            text=data["text"],
            docstring=data.get("docstring", ""),
            extra=data.get("extra", {}) or {},
        )


def language_for_path(path: str) -> str:
    _, ext = os.path.splitext(path)
    return LANGUAGE_BY_EXT.get(ext.lower(), "text")


def _slice_lines(lines: list[str], start_line: int, end_line: int) -> str:
    return "".join(lines[start_line - 1 : end_line])


def _python_chunks(path: str, source: str) -> list[CodeChunk]:
    """Extract function/class chunks from a Python source string."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fall back to a generic chunker for files we cannot parse.
        return _generic_chunks(path, source, language="python")

    lines = source.splitlines(keepends=True)
    total_lines = len(lines)
    chunks: list[CodeChunk] = []

    def end_of(node: ast.AST, default: int) -> int:
        end = getattr(node, "end_lineno", None)
        return int(end) if end is not None else default

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}" if prefix else child.name
                start = child.lineno
                # Include decorators in the captured span.
                for dec in getattr(child, "decorator_list", []):
                    start = min(start, dec.lineno)
                end = end_of(child, start)
                if isinstance(child, ast.ClassDef):
                    kind = "class"
                elif prefix:
                    kind = "method"
                else:
                    kind = "function"
                text = _slice_lines(lines, start, end)
                chunks.append(
                    CodeChunk(
                        path=path,
                        language="python",
                        kind=kind,
                        symbol=name,
                        start_line=start,
                        end_line=end,
                        text=text,
                        docstring=ast.get_docstring(child) or "",
                    )
                )
                child_prefix = f"{name}."
                visit(child, child_prefix)

    visit(tree, "")

    # Capture module-level docstring / top-level code as a "module" chunk so
    # small scripts without functions still yield something searchable.
    module_doc = ast.get_docstring(tree) or ""
    if not chunks:
        return _generic_chunks(path, source, language="python")

    if module_doc:
        # Find the span of the leading docstring expression.
        first = tree.body[0] if tree.body else None
        if isinstance(first, ast.Expr):
            start = first.lineno
            end = end_of(first, start)
            chunks.insert(
                0,
                CodeChunk(
                    path=path,
                    language="python",
                    kind="module",
                    symbol=os.path.basename(path),
                    start_line=start,
                    end_line=min(end, total_lines),
                    text=_slice_lines(lines, start, min(end, total_lines)),
                    docstring=module_doc,
                ),
            )

    chunks.sort(key=lambda c: (c.start_line, c.end_line))
    return chunks


def _generic_chunks(
    path: str,
    source: str,
    language: str | None = None,
    window: int = 60,
    overlap: int = 15,
) -> list[CodeChunk]:
    """Sliding-window chunker used for non-Python or unparseable files."""
    if language is None:
        language = language_for_path(path)

    lines = source.splitlines(keepends=True)
    total = len(lines)
    if total == 0:
        return []

    step = max(1, window - overlap)
    chunks: list[CodeChunk] = []
    start = 0
    while start < total:
        end = min(total, start + window)
        text = "".join(lines[start:end])
        if text.strip():
            chunks.append(
                CodeChunk(
                    path=path,
                    language=language,
                    kind="block",
                    symbol="",
                    start_line=start + 1,
                    end_line=end,
                    text=text,
                )
            )
        if end >= total:
            break
        start += step
    return chunks


def chunk_file(path: str, source: str) -> list[CodeChunk]:
    """Return searchable chunks for ``source`` originating from ``path``."""
    language = language_for_path(path)
    if language == "python":
        return _python_chunks(path, source)
    return _generic_chunks(path, source, language=language)


def iter_source_files(
    root: str,
    extensions: Iterable[str] | None = None,
    exclude_dirs: Iterable[str] | None = None,
    max_bytes: int = MAX_FILE_BYTES,
) -> Iterator[str]:
    """Yield paths of indexable source files under ``root``.

    ``root`` may be a single file or a directory tree.
    """
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (extensions or DEFAULT_EXTENSIONS)}
    excluded = set(exclude_dirs) if exclude_dirs is not None else set(DEFAULT_EXCLUDE_DIRS)

    if os.path.isfile(root):
        _, ext = os.path.splitext(root)
        if ext.lower() in exts:
            yield root
        return

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in place for efficiency.
        dirnames[:] = [d for d in sorted(dirnames) if d not in excluded]
        for filename in sorted(filenames):
            _, ext = os.path.splitext(filename)
            if ext.lower() not in exts:
                continue
            full = os.path.join(dirpath, filename)
            try:
                if os.path.getsize(full) > max_bytes:
                    continue
            except OSError:
                continue
            yield full


def read_source(path: str) -> str | None:
    """Read a text file, returning ``None`` if it cannot be decoded."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except (UnicodeDecodeError, OSError):
        return None
