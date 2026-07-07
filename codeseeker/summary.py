"""Produce a short, human-readable explanation of a repository.

The summary is built from cheap, deterministic signals (README, language mix,
directory layout, and the most prominent functions/classes discovered while
indexing). When an LLM is configured, those same signals are handed to it to
generate a polished natural-language description; otherwise a solid heuristic
summary is returned so the feature works fully offline.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field

from codeseeker.embeddings import tokenize
from codeseeker.index import CodeIndex
from codeseeker.llm import LLMClient

_README_NAMES = ("README.md", "README.rst", "README.txt", "README", "readme.md")


@dataclass
class RepoSummary:
    root: str
    name: str
    num_files: int
    num_chunks: int
    languages: list[tuple[str, int]]
    kinds: dict[str, int]
    top_dirs: list[tuple[str, int]]
    readme_excerpt: str
    notable_symbols: list[str]
    description: str = ""
    llm_used: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "name": self.name,
            "num_files": self.num_files,
            "num_chunks": self.num_chunks,
            "languages": self.languages,
            "kinds": self.kinds,
            "top_dirs": self.top_dirs,
            "readme_excerpt": self.readme_excerpt,
            "notable_symbols": self.notable_symbols,
            "description": self.description,
            "llm_used": self.llm_used,
            "components": self.extra.get("components", []),
            "entry_points": self.extra.get("entry_points", []),
            "readme_overview": self.extra.get("readme_overview", ""),
            "suggested_questions": self.extra.get("suggested_ask_questions", []),
            "suggested_ask_questions": self.extra.get("suggested_ask_questions", []),
            "suggested_search_queries": self.extra.get("suggested_search_queries", []),
        }


def _read_readme(root: str, max_chars: int = 1500) -> str:
    for name in _README_NAMES:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read().strip()
            except OSError:
                continue
            return text[:max_chars]
    return ""


def _first_sentence(text: str, limit: int = 160) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    for sep in (". ", "! ", "? "):
        idx = text.find(sep)
        if 0 < idx < limit:
            return text[: idx + 1]
    return text[:limit] + ("…" if len(text) > limit else "")


def _collect_signals(index: CodeIndex) -> dict:
    files = {c.path for c in index.chunks}
    languages = Counter(c.language for c in index.chunks)
    kinds = Counter(c.kind for c in index.chunks)

    top_dirs: Counter = Counter()
    for path in files:
        head = path.split(os.sep)[0] if os.sep in path else path.split("/")[0]
        if head and not head.endswith((".py", ".js", ".ts")) or (os.sep in path or "/" in path):
            top_dirs[head] += 1

    # Notable symbols: prefer documented classes, then documented functions,
    # then anything with a symbol name. De-duplicate while preserving order.
    documented = [c for c in index.chunks if c.symbol and c.docstring]
    documented.sort(key=lambda c: (c.kind != "class", -len(c.docstring)))
    plain = [c for c in index.chunks if c.symbol and not c.docstring]

    notable: list[str] = []
    components: list[dict] = []
    seen: set[str] = set()
    for chunk in documented + plain:
        if chunk.symbol in seen:
            continue
        seen.add(chunk.symbol)
        notable.append(f"{chunk.symbol} ({chunk.kind})")
        if chunk.docstring and len(components) < 8:
            components.append(
                {
                    "symbol": chunk.symbol,
                    "kind": chunk.kind,
                    "location": chunk.location,
                    "summary": _first_sentence(chunk.docstring),
                }
            )
        if len(notable) >= 12 and len(components) >= 8:
            break

    # Detect likely entry points (main scripts / app / cli).
    entry_points = sorted(
        p for p in files
        if os.path.basename(p).lower() in {"main.py", "app.py", "cli.py", "__main__.py", "manage.py", "server.py", "run.py"}
    )

    return {
        "num_files": len(files),
        "languages": languages.most_common(),
        "kinds": dict(kinds),
        "top_dirs": top_dirs.most_common(8),
        "notable_symbols": notable,
        "components": components,
        "entry_points": entry_points[:5],
    }


def _clean_project_name(origin: str, root: str) -> str:
    """Derive a clean, human-friendly project name.

    Prefers the repository name from ``origin`` (e.g. ``psf/requests`` -> requests)
    and avoids exposing internal cache folder names like ``requests-ab12cd``.
    """
    candidate = origin or root or ""
    candidate = candidate.rstrip("/")
    if candidate.endswith(".git"):
        candidate = candidate[:-4]
    # Take the last path component of a URL/shorthand/path.
    for sep in ("/", "\\"):
        if sep in candidate:
            candidate = candidate.split(sep)[-1]
    if not candidate or candidate in {".", ".."}:
        candidate = os.path.basename(os.path.abspath(root)) if root else "This project"
    return candidate


def _detect_project_type(signals: dict, readme: str, name: str) -> str:
    """Guess the project category from code structure and README."""
    text = (readme + " " + name).lower()
    kinds = signals.get("kinds", {})
    dirs = [d for d, _ in signals.get("top_dirs", [])]
    components = signals.get("components", [])

    if any(w in text for w in ("machine learning", "regression", "classification", "forecast", "dataset", "model training", "neural")):
        return "machine learning / data science"
    if any(w in text for w in ("web app", "flask", "django", "fastapi", "api server", "rest api")):
        return "web application / API"
    if kinds.get("cell") or kinds.get("note"):
        return "data science / notebook project"
    if any("api" in c["symbol"].lower() or "route" in c["symbol"].lower() for c in components[:5]):
        return "API / web service"
    if any(w in text for w in ("library", "package", "module", "sdk", "client")):
        return "software library"
    if signals.get("entry_points"):
        return "application"
    if "src" in dirs or "lib" in dirs:
        return "software library"
    return "software project"


def _synthesize_explanation(name: str, signals: dict, readme: str) -> str:
    """Write a human explanation by understanding README + code — not copy-paste."""
    lines: list[str] = []
    project_type = _detect_project_type(signals, readme, name)

    # 1. What it is — one clear sentence from README context + code signals.
    readme_lead = _first_meaningful_paragraph(readme)
    if readme_lead:
        # Use README as input, but rewrite as a clean statement.
        lead = readme_lead.rstrip(".")
        if not lead.lower().startswith(name.lower()):
            lines.append(f"{name} is a {project_type} project. {lead}.")
        else:
            lines.append(f"{lead}.")
    else:
        lines.append(f"{name} is a {project_type} project.")

    # 2. What it does — inferred from key components and their docstrings.
    components = signals.get("components") or []
    if components:
        roles = []
        for comp in components[:4]:
            if comp.get("summary"):
                roles.append(f"{comp['symbol']} ({comp['summary'].rstrip('.')})")
            else:
                roles.append(comp["symbol"])
        if roles:
            lines.append(
                "The main building blocks are: " + "; ".join(roles) + "."
            )

    # 3. How it's organized.
    langs = signals["languages"]
    top_dirs = signals.get("top_dirs") or []
    org_parts = []
    if langs:
        main_lang = langs[0][0]
        org_parts.append(f"written mainly in {main_lang}")
    meaningful_dirs = [d for d, _ in top_dirs if d not in {".", name, main_lang if langs else ""}][:3]
    if meaningful_dirs:
        org_parts.append(f"organized into {', '.join(meaningful_dirs)}")
    kinds = signals.get("kinds", {})
    code_units = []
    for label in ("class", "function", "method"):
        if kinds.get(label):
            code_units.append(f"{kinds[label]} {label}{'es' if label == 'class' else 's'}")
    if code_units:
        org_parts.append("containing " + ", ".join(code_units))
    if org_parts:
        lines.append("The codebase is " + ", ".join(org_parts) + ".")

    # 4. How to run / entry point.
    if signals.get("entry_points"):
        lines.append(
            "To run it, start from: " + ", ".join(signals["entry_points"]) + "."
        )

    return "\n\n".join(lines)


def _clean_block(block: str) -> str:
    return " ".join(
        line.strip().lstrip("#").strip()
        for line in block.splitlines()
        if line.strip()
        and not line.strip().startswith(("![", "[!", "<", "```", "|", "---", "==="))
    ).strip()


def _first_meaningful_paragraph(readme: str) -> str:
    for block in readme.split("\n\n"):
        cleaned = _clean_block(block)
        if len(cleaned) > 40:
            return cleaned
    return ""


def _readme_overview(readme: str, max_chars: int = 700) -> str:
    """Return the README's leading prose (a few paragraphs), cleaned up.

    This is what powers the human-readable 'what this project is' explanation.
    """
    if not readme:
        return ""
    paras: list[str] = []
    used = 0
    for block in readme.split("\n\n"):
        cleaned = _clean_block(block)
        if len(cleaned) < 25:
            continue
        paras.append(cleaned)
        used += len(cleaned)
        if used >= max_chars or len(paras) >= 3:
            break
    text = "\n\n".join(paras)
    return text[:max_chars].rstrip()


# Symbol vocabulary -> short code-search phrases (find snippets, not Q&A).
_SEARCH_THEMES = {
    "load": "load configuration from file",
    "read": "read input data",
    "parse": "parsing logic",
    "connect": "database connection setup",
    "auth": "authentication handler",
    "login": "login flow",
    "train": "model training loop",
    "predict": "prediction function",
    "fit": "model fitting",
    "plot": "plotting / visualization",
    "config": "configuration setup",
    "request": "HTTP request handling",
    "route": "routing / URL mapping",
    "render": "template rendering",
    "validate": "input validation",
    "cache": "caching layer",
    "encode": "encoding logic",
    "decode": "decoding logic",
    "serialize": "serialization",
    "forecast": "forecasting logic",
    "clean": "data cleaning / preprocessing",
    "split": "train test split",
    "evaluate": "model evaluation",
    "score": "scoring function",
    "retry": "retry with backoff",
    "session": "session management",
    "cookie": "cookie handling",
    "redirect": "redirect logic",
    "middleware": "middleware pipeline",
    "handler": "request handler",
}


# Function-name themes -> a natural-language question a newcomer might ask.
_ASK_THEMES = {
    "load": "How is data loaded?",
    "read": "How is input data read?",
    "parse": "How is parsing handled?",
    "connect": "How are connections established?",
    "auth": "How is authentication handled?",
    "login": "How does login work?",
    "train": "How is the model trained?",
    "predict": "How does prediction work?",
    "fit": "How is the model fitted to the data?",
    "plot": "How are results visualized?",
    "config": "How is configuration handled?",
    "request": "How are requests processed?",
    "route": "How is routing handled?",
    "render": "How is rendering done?",
    "validate": "How is input validated?",
    "cache": "How is caching implemented?",
    "encode": "How is data encoded?",
    "decode": "How is data decoded?",
    "serialize": "How is data serialized?",
    "forecast": "How is forecasting performed?",
    "clean": "How is the data cleaned/preprocessed?",
    "split": "How is the data split?",
    "evaluate": "How is the model evaluated?",
    "score": "How are results scored?",
}


def suggest_ask_questions(index: CodeIndex, max_questions: int = 6) -> list[str]:
    """Generate how/why/what questions grounded in the indexed code."""
    signals = _collect_signals(index)
    questions: list[str] = []
    seen: set[str] = set()

    def add(q: str) -> None:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            questions.append(q)

    for comp in signals.get("components", []):
        sym = comp["symbol"]
        kind = comp["kind"]
        if kind == "class":
            add(f"What is the role of {sym}?")
        elif kind in {"function", "method"}:
            add(f"How does {sym} work?")
        else:
            add(f"What does {sym} do?")
        if len(questions) >= 3:
            break

    tokens: set[str] = set()
    for chunk in index.chunks:
        tokens.update(tokenize(chunk.symbol))
    for key, q in _ASK_THEMES.items():
        if key in tokens:
            add(q)

    add("What are the main entry points of this project?")
    add("How is this project structured?")

    return questions[:max_questions]


def suggest_search_queries(index: CodeIndex, max_queries: int = 6) -> list[str]:
    """Generate short code-finding queries — what to type in Search, not Ask."""
    signals = _collect_signals(index)
    queries: list[str] = []
    seen: set[str] = set()

    def add(q: str) -> None:
        key = q.lower()
        if key not in seen and q.strip():
            seen.add(key)
            queries.append(q)

    for comp in signals.get("components", []):
        if comp["kind"] not in {"function", "method", "class"}:
            continue
        summary = (comp.get("summary") or "").strip().rstrip(".")
        if summary and len(summary) > 12:
            add(summary[0].lower() + summary[1:] if summary else summary)
        else:
            add(comp["symbol"].replace("_", " "))
        if len(queries) >= 2:
            break

    tokens: set[str] = set()
    for chunk in index.chunks:
        tokens.update(tokenize(chunk.symbol))
    for key, phrase in _SEARCH_THEMES.items():
        if key in tokens:
            add(phrase)

    skip_stems = {"__init__", "main", "index", "utils", "util", "test", "tests", "conftest", "setup"}
    files = {c.path for c in index.chunks}
    for path in sorted(files):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem and stem.lower() not in skip_stems and len(stem) > 3:
            add(stem.replace("_", " "))
        if len(queries) >= max_queries - 2:
            break

    add("main entry point")
    add("error handling")

    return queries[:max_queries]


def suggest_questions(index: CodeIndex, max_questions: int = 6) -> list[str]:
    """Backward-compatible alias for :func:`suggest_ask_questions`."""
    return suggest_ask_questions(index, max_questions=max_questions)


def _llm_description(client: LLMClient, name: str, signals: dict, readme: str) -> str:
    context_lines = [
        f"Repository name: {name}",
        f"Indexed source files: {signals['num_files']}",
        f"Languages (by chunk count): {signals['languages']}",
        f"Symbol kinds: {signals['kinds']}",
        f"Top-level directories: {signals['top_dirs']}",
        f"Notable symbols: {signals['notable_symbols']}",
    ]
    if readme:
        context_lines.append("README excerpt:\n" + readme[:1200])
    system = (
        "You are a senior engineer who writes concise, accurate summaries of "
        "software repositories for other developers. Base your answer only on "
        "the provided facts. Respond in 3-5 sentences."
    )
    user = (
        "Summarise what this project does, its main components, and its likely "
        "purpose:\n\n" + "\n".join(context_lines)
    )
    return client.complete(system, user)


def summarize_repo(
    index: CodeIndex,
    root: str | None = None,
    use_llm: bool | str = "auto",
    llm_client: LLMClient | None = None,
) -> RepoSummary:
    """Build a :class:`RepoSummary` for an indexed repository."""
    root = root or index.root or "."
    signals = _collect_signals(index)
    readme = _read_readme(root)
    name = _clean_project_name(getattr(index, "origin", "") or "", root)

    client = llm_client or LLMClient()
    want_llm = client.available() if use_llm == "auto" else bool(use_llm)

    description = ""
    llm_used = False
    if want_llm and client.available():
        try:
            description = _llm_description(client, name, signals, readme)
            llm_used = bool(description)
        except Exception:
            description = ""
    if not description:
        description = _synthesize_explanation(name, signals, readme)

    return RepoSummary(
        root=root,
        name=name,
        num_files=signals["num_files"],
        num_chunks=len(index),
        languages=signals["languages"],
        kinds=signals["kinds"],
        top_dirs=signals["top_dirs"],
        readme_excerpt=_first_meaningful_paragraph(readme),
        notable_symbols=signals["notable_symbols"],
        description=description,
        llm_used=llm_used,
        extra={
            "components": signals.get("components", []),
            "entry_points": signals.get("entry_points", []),
            "readme_overview": _readme_overview(readme),
            "suggested_ask_questions": suggest_ask_questions(index),
            "suggested_search_queries": suggest_search_queries(index),
        },
    )
