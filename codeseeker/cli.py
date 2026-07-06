"""Command-line interface for codeseeker."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Sequence

from codeseeker import __version__
from codeseeker.index import CodeIndex, SearchResult, default_index_dir

# ANSI colours, disabled automatically when output is not a TTY.
_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    if not _COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _parse_extensions(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [e.strip() for e in value.split(",") if e.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codeseeker",
        description="Semantic code search over a local codebase.",
    )
    parser.add_argument("--version", action="version", version=f"codeseeker {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # index ---------------------------------------------------------------
    p_index = sub.add_parser("index", help="Build a semantic index for a codebase.")
    p_index.add_argument("path", nargs="?", default=".", help="Root directory (or file) to index.")
    p_index.add_argument(
        "--index-dir",
        default=None,
        help="Where to store the index (default: <path>/.codeseeker).",
    )
    p_index.add_argument(
        "--ext",
        default=None,
        help="Comma-separated list of file extensions to include (e.g. .py,.js).",
    )
    p_index.add_argument(
        "--exclude",
        default=None,
        help="Comma-separated directory names to exclude (adds to defaults).",
    )
    p_index.add_argument(
        "--backend",
        default="tfidf",
        help="Embedding backend: 'tfidf' (default, offline) or 'sentence-transformers'.",
    )
    p_index.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Model name for the sentence-transformers backend.",
    )
    p_index.set_defaults(func=_cmd_index)

    # search --------------------------------------------------------------
    p_search = sub.add_parser("search", help="Search an existing index.")
    p_search.add_argument("query", help="Natural-language or code query.")
    p_search.add_argument("path", nargs="?", default=".", help="Root of the indexed codebase.")
    p_search.add_argument("--index-dir", default=None, help="Index location.")
    p_search.add_argument("-k", "--top-k", type=int, default=5, help="Number of results.")
    p_search.add_argument("--snippet-lines", type=int, default=8, help="Max lines of code shown per hit.")
    p_search.add_argument("--json", action="store_true", help="Emit results as JSON.")
    p_search.set_defaults(func=_cmd_search)

    return parser


def _resolve_index_dir(args) -> str:
    if args.index_dir:
        return args.index_dir
    return default_index_dir(args.path)


def _cmd_index(args) -> int:
    extensions = _parse_extensions(args.ext)
    extra_excludes = _parse_extensions(args.exclude)
    exclude_dirs = None
    if extra_excludes:
        from codeseeker.chunking import DEFAULT_EXCLUDE_DIRS

        exclude_dirs = set(DEFAULT_EXCLUDE_DIRS) | set(extra_excludes)

    if not os.path.exists(args.path):
        print(_c(f"error: path not found: {args.path}", "31"), file=sys.stderr)
        return 2

    index_dir = _resolve_index_dir(args)
    start = time.perf_counter()
    print(f"Indexing {_c(os.path.abspath(args.path), '36')} ...")

    index = CodeIndex.build(
        root=args.path,
        extensions=extensions,
        exclude_dirs=exclude_dirs,
        backend=args.backend,
        model_name=args.model,
        progress=lambda msg: print(f"  {msg}"),
    )
    index.save(index_dir)
    elapsed = time.perf_counter() - start
    print(
        _c("✓", "32")
        + f" Indexed {len(index)} chunks in {elapsed:.2f}s "
        + f"-> {_c(index_dir, '36')}"
    )
    if len(index) == 0:
        print(_c("warning: no indexable source files were found.", "33"), file=sys.stderr)
    return 0


def _format_result(result: SearchResult, snippet_lines: int) -> str:
    chunk = result.chunk
    header_bits = [
        _c(f"{result.score:.3f}", "33"),
        _c(chunk.location, "36"),
    ]
    if chunk.symbol:
        header_bits.append(_c(f"{chunk.kind} {chunk.symbol}", "32"))
    elif chunk.kind:
        header_bits.append(_c(chunk.kind, "32"))
    header = "  ".join(header_bits)

    lines = chunk.text.rstrip("\n").splitlines()
    if len(lines) > snippet_lines:
        shown = lines[:snippet_lines]
        shown.append(_c(f"    ... (+{len(lines) - snippet_lines} more lines)", "90"))
    else:
        shown = lines
    body = "\n".join(f"    {line}" for line in shown)
    return f"{header}\n{body}"


def _cmd_search(args) -> int:
    index_dir = _resolve_index_dir(args)
    if not os.path.isdir(index_dir):
        print(
            _c(f"error: no index found at {index_dir}. Run 'codeseeker index' first.", "31"),
            file=sys.stderr,
        )
        return 2

    index = CodeIndex.load(index_dir)
    results = index.search(args.query, top_k=args.top_k)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0

    if not results:
        print(_c("No matches found.", "33"))
        return 0

    print(_c(f"Top {len(results)} results for: ", "1") + _c(args.query, "35") + "\n")
    for i, result in enumerate(results, 1):
        print(_c(f"[{i}]", "1") + " " + _format_result(result, args.snippet_lines))
        print()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the CLI user
        print(_c(f"error: {exc}", "31"), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
