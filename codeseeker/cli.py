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
from codeseeker.repo import DEFAULT_CACHE_DIR, is_remote_source, resolve_source

# ANSI colours, disabled automatically when output is not a TTY.
_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    if not _COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _err(msg: str) -> None:
    print(_c(f"error: {msg}", "31"), file=sys.stderr)


def _parse_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [e.strip() for e in value.split(",") if e.strip()]


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codeseeker",
        description="LLM-powered semantic code search over local and remote repositories.",
    )
    parser.add_argument("--version", action="version", version=f"codeseeker {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # index ---------------------------------------------------------------
    p_index = sub.add_parser("index", help="Build a semantic index for a local dir or remote repo.")
    p_index.add_argument(
        "source",
        nargs="?",
        default=".",
        help="Local path, git URL, or 'owner/repo' shorthand to index.",
    )
    p_index.add_argument("--index-dir", default=None, help="Where to store the index.")
    p_index.add_argument("--ext", default=None, help="Comma-separated extensions to include (e.g. .py,.js).")
    p_index.add_argument("--exclude", default=None, help="Extra directory names to exclude.")
    p_index.add_argument(
        "--backend",
        default="tfidf",
        help="Engine: 'tfidf' (simple/offline) or 'sentence-transformers' (deep semantic).",
    )
    p_index.add_argument("--model", default="all-MiniLM-L6-v2", help="Model for the neural backend.")
    p_index.add_argument(
        "--faiss",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use FAISS for search ('auto' uses it when installed).",
    )
    p_index.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Where cloned repos are cached.")
    p_index.add_argument("--depth", type=int, default=1, help="Clone depth for remote repos.")
    p_index.add_argument("--branch", default=None, help="Branch to clone for remote repos.")
    p_index.add_argument("--update", action="store_true", help="Refresh an existing clone before indexing.")
    p_index.set_defaults(func=_cmd_index)

    # clone ---------------------------------------------------------------
    p_clone = sub.add_parser("clone", help="Pull a remote repository to the local machine.")
    p_clone.add_argument("source", help="Git URL or 'owner/repo' shorthand.")
    p_clone.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Destination cache directory.")
    p_clone.add_argument("--depth", type=int, default=1, help="Clone depth.")
    p_clone.add_argument("--branch", default=None, help="Branch to clone.")
    p_clone.add_argument("--update", action="store_true", help="Update if already cloned.")
    p_clone.set_defaults(func=_cmd_clone)

    # search --------------------------------------------------------------
    p_search = sub.add_parser("search", help="Semantic search over an existing index.")
    p_search.add_argument("query", nargs="?", default=None, help="Natural-language or code query.")
    p_search.add_argument("path", nargs="?", default=".", help="Root of the indexed codebase.")
    p_search.add_argument("--index-dir", default=None, help="Index location.")
    p_search.add_argument("-k", "--top-k", type=int, default=5, help="Number of results.")
    p_search.add_argument("--lang", default=None, help="Filter by language(s), comma-separated.")
    p_search.add_argument("--kind", default=None, help="Filter by kind(s): function,class,method,module,block.")
    p_search.add_argument("--path", dest="path_contains", default=None, help="Only match paths containing this substring.")
    p_search.add_argument("--snippet-lines", type=int, default=8, help="Max lines of code shown per hit.")
    p_search.add_argument(
        "--mode",
        choices=["hybrid", "semantic"],
        default="hybrid",
        help="Ranking mode: hybrid (semantic+keywords, recommended) or semantic only.",
    )
    p_search.add_argument(
        "--semantic-weight",
        type=float,
        default=0.8,
        help="Hybrid blend weight for semantic score in [0,1] (default 0.8).",
    )
    p_search.add_argument("--json", action="store_true", help="Emit results as JSON.")
    p_search.add_argument("-i", "--interactive", action="store_true", help="Interactive search REPL.")
    p_search.set_defaults(func=_cmd_search)

    # explain -------------------------------------------------------------
    p_explain = sub.add_parser("explain", help="Explain what an indexed project does, in short.")
    p_explain.add_argument("path", nargs="?", default=".", help="Root of the indexed codebase.")
    p_explain.add_argument("--index-dir", default=None, help="Index location.")
    p_explain.add_argument("--llm", choices=["auto", "on", "off"], default="auto", help="Use an LLM if available.")
    p_explain.add_argument("--json", action="store_true", help="Emit the summary as JSON.")
    p_explain.set_defaults(func=_cmd_explain)

    # ask -----------------------------------------------------------------
    p_ask = sub.add_parser("ask", help="Ask a natural-language question grounded in the code (RAG).")
    p_ask.add_argument("question", help="Your question about the codebase.")
    p_ask.add_argument("path", nargs="?", default=".", help="Root of the indexed codebase.")
    p_ask.add_argument("--index-dir", default=None, help="Index location.")
    p_ask.add_argument("-k", "--top-k", type=int, default=6, help="How many chunks to retrieve.")
    p_ask.add_argument("--llm", choices=["auto", "on", "off"], default="auto", help="Use an LLM if available.")
    p_ask.add_argument("--json", action="store_true", help="Emit the answer as JSON.")
    p_ask.set_defaults(func=_cmd_ask)

    # stats ---------------------------------------------------------------
    p_stats = sub.add_parser("stats", help="Show statistics about an existing index.")
    p_stats.add_argument("path", nargs="?", default=".", help="Root of the indexed codebase.")
    p_stats.add_argument("--index-dir", default=None, help="Index location.")
    p_stats.add_argument("--json", action="store_true", help="Emit stats as JSON.")
    p_stats.set_defaults(func=_cmd_stats)

    # web -----------------------------------------------------------------
    p_web = sub.add_parser("web", help="Launch the web UI (opens in your browser).")
    p_web.add_argument("--host", default="127.0.0.1", help="Host to bind (default 127.0.0.1).")
    p_web.add_argument("--port", type=int, default=8000, help="Port to bind (default 8000).")
    p_web.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser.")
    p_web.set_defaults(func=_cmd_web)

    return parser


def _resolve_index_dir(args, base_path: str | None = None) -> str:
    if getattr(args, "index_dir", None):
        return args.index_dir
    return default_index_dir(base_path if base_path is not None else args.path)


def _faiss_pref(value: str) -> bool | str:
    return {"auto": "auto", "on": True, "off": False}[value]


def _llm_pref(value: str) -> bool | str:
    return {"auto": "auto", "on": True, "off": False}[value]


def _load_index(index_dir: str) -> CodeIndex | None:
    if not os.path.isdir(index_dir):
        _err(f"no index found at {index_dir}. Run 'codeseeker index' first.")
        return None
    return CodeIndex.load(index_dir)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def _cmd_index(args) -> int:
    extensions = _parse_list(args.ext)
    extra_excludes = _parse_list(args.exclude)
    exclude_dirs = None
    if extra_excludes:
        from codeseeker.chunking import DEFAULT_EXCLUDE_DIRS

        exclude_dirs = set(DEFAULT_EXCLUDE_DIRS) | set(extra_excludes)

    is_remote = is_remote_source(args.source)
    try:
        if is_remote:
            print(f"Fetching {_c(args.source, '36')} to local machine ...")
        repo = resolve_source(
            args.source,
            cache_dir=args.cache_dir,
            depth=args.depth,
            update=args.update,
            branch=args.branch,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        _err(str(exc))
        return 2

    if repo.is_remote:
        state = "cloned" if repo.cloned else "using cached clone"
        print(_c("✓", "32") + f" {state}: {_c(repo.local_path, '36')}")

    # The index lives inside the (possibly cloned) repo unless overridden.
    index_dir = _resolve_index_dir(args, base_path=repo.local_path)

    start = time.perf_counter()
    print(f"Indexing {_c(repo.local_path, '36')} ...")
    index = CodeIndex.build(
        root=repo.local_path,
        extensions=extensions,
        exclude_dirs=exclude_dirs,
        backend=args.backend,
        model_name=args.model,
        origin=repo.origin,
        is_remote=repo.is_remote,
        prefer_faiss=_faiss_pref(args.faiss),
        progress=lambda msg: print(f"  {msg}"),
    )
    index.save(index_dir)
    elapsed = time.perf_counter() - start
    print(
        _c("✓", "32")
        + f" Indexed {len(index)} chunks in {elapsed:.2f}s "
        + f"[backend: {index.backend_name}] -> {_c(index_dir, '36')}"
    )
    if len(index) == 0:
        print(_c("warning: no indexable source files were found.", "33"), file=sys.stderr)
    return 0


def _cmd_clone(args) -> int:
    try:
        repo = resolve_source(
            args.source,
            cache_dir=args.cache_dir,
            depth=args.depth,
            update=args.update,
            branch=args.branch,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        _err(str(exc))
        return 2
    state = "Cloned" if repo.cloned else "Already present"
    print(_c("✓", "32") + f" {state}: {_c(repo.local_path, '36')}")
    print("Index it with: " + _c(f"codeseeker index {repo.local_path}", "35"))
    return 0


def _format_result(result: SearchResult, snippet_lines: int) -> str:
    chunk = result.chunk
    header_bits = [_c(f"{result.score:.3f}", "33"), _c(chunk.location, "36")]
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


def _run_query(index: CodeIndex, query: str, args) -> None:
    results = index.search(
        query,
        top_k=args.top_k,
        languages=_parse_list(args.lang),
        kinds=_parse_list(args.kind),
        path_contains=args.path_contains,
        mode=args.mode,
        semantic_weight=args.semantic_weight,
    )
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return
    if not results:
        print(_c("No matches found.", "33"))
        return
    print(_c(f"Top {len(results)} results for: ", "1") + _c(query, "35") + "\n")
    for i, result in enumerate(results, 1):
        print(_c(f"[{i}]", "1") + " " + _format_result(result, args.snippet_lines))
        print()


def _cmd_search(args) -> int:
    index_dir = _resolve_index_dir(args)
    index = _load_index(index_dir)
    if index is None:
        return 2

    if args.interactive:
        print(_c("codeseeker interactive search", "1") + f"  [backend: {index.backend_name}]")
        print("Type a query and press Enter. Empty line or Ctrl-D to quit.\n")
        while True:
            try:
                query = input(_c("search> ", "35")).strip()
            except EOFError:
                print()
                break
            if not query:
                break
            _run_query(index, query, args)
        return 0

    if not args.query:
        _err("a query is required (or use --interactive).")
        return 2
    _run_query(index, args.query, args)
    return 0


def _cmd_explain(args) -> int:
    from codeseeker.summary import summarize_repo

    index_dir = _resolve_index_dir(args)
    index = _load_index(index_dir)
    if index is None:
        return 2

    summary = summarize_repo(index, root=index.root or args.path, use_llm=_llm_pref(args.llm))
    if args.json:
        print(json.dumps(summary.to_dict(), indent=2))
        return 0

    print(_c("Project explanation", "1") + (_c("  (LLM)", "90") if summary.llm_used else ""))
    print(_c(summary.name or summary.root, "36"))
    print()
    print(summary.description or "(no description available)")
    print()
    if summary.languages:
        langs = ", ".join(f"{lang} ({count})" for lang, count in summary.languages[:6])
        print(_c("Languages: ", "1") + langs)
    print(_c("Chunks: ", "1") + f"{summary.num_chunks} across {summary.num_files} files")
    if summary.notable_symbols:
        print(_c("Notable: ", "1") + ", ".join(summary.notable_symbols[:8]))
    return 0


def _cmd_ask(args) -> int:
    from codeseeker.qa import answer_question

    index_dir = _resolve_index_dir(args)
    index = _load_index(index_dir)
    if index is None:
        return 2

    result = answer_question(index, args.question, top_k=args.top_k, use_llm=_llm_pref(args.llm))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(_c("Q: ", "1") + _c(args.question, "35"))
    print()
    print(result.answer)
    if result.sources:
        print()
        print(_c("Sources:", "1"))
        for i, src in enumerate(result.sources, 1):
            label = f"{src.chunk.kind} {src.chunk.symbol}".strip() if src.chunk.symbol else src.chunk.kind
            print(f"  [{i}] {_c(src.chunk.location, '36')}  ({label}, {src.score:.3f})")
    return 0


def _cmd_stats(args) -> int:
    from collections import Counter

    index_dir = _resolve_index_dir(args)
    index = _load_index(index_dir)
    if index is None:
        return 2

    languages = Counter(c.language for c in index.chunks)
    kinds = Counter(c.kind for c in index.chunks)
    files = {c.path for c in index.chunks}
    dim = int(index.vectors.shape[1]) if index.vectors.ndim == 2 and index.vectors.size else 0

    if args.json:
        print(
            json.dumps(
                {
                    "root": index.root,
                    "origin": index.origin,
                    "is_remote": index.is_remote,
                    "num_files": len(files),
                    "num_chunks": len(index),
                    "embedder": index.embedder.name,
                    "search_backend": index.backend_name,
                    "vector_dim": dim,
                    "languages": languages.most_common(),
                    "kinds": dict(kinds),
                },
                indent=2,
            )
        )
        return 0

    print(_c("Index statistics", "1"))
    print(_c("Origin: ", "1") + (index.origin or index.root))
    print(_c("Files: ", "1") + str(len(files)) + _c("   Chunks: ", "1") + str(len(index)))
    print(_c("Embedder: ", "1") + index.embedder.name + _c("   Search backend: ", "1") + index.backend_name + f"  (dim {dim})")
    if languages:
        print(_c("Languages:", "1"))
        for lang, count in languages.most_common():
            print(f"  {lang:<14} {count}")
    if kinds:
        print(_c("Kinds:", "1"))
        for kind, count in kinds.most_common():
            print(f"  {kind:<14} {count}")
    return 0


def _cmd_web(args) -> int:
    from codeseeker.webapp import run_server

    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)
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
        _err(str(exc))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
