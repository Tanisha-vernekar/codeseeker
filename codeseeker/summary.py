"""Produce a short, human-readable explanation of a repository.

Uses multi-pass analysis (config files, import graph, architecture layers,
symbol centrality, and semantic retrieval over the index) to *understand* a
project before explaining it. When an LLM is configured, the structured
profile plus retrieved code snippets are passed as grounded context.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from codeseeker.analysis import RepoProfile, analyze_repo, _clean_docstring
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
            "project_type": self.extra.get("project_type", ""),
            "tech_stack": self.extra.get("tech_stack", []),
            "architecture_layers": self.extra.get("architecture_layers", []),
            "workflows": self.extra.get("workflows", []),
            "components": self.extra.get("components", []),
            "entry_points": self.extra.get("entry_points", []),
            "insights": self.extra.get("insights", []),
            "readme_overview": self.extra.get("readme_overview", ""),
            "suggested_questions": self.extra.get("suggested_ask_questions", []),
            "suggested_ask_questions": self.extra.get("suggested_ask_questions", []),
            "suggested_search_queries": self.extra.get("suggested_search_queries", []),
        }


def _read_readme(root: str, max_chars: int = 2000) -> str:
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


def _clean_project_name(origin: str, root: str) -> str:
    candidate = origin or root or ""
    candidate = candidate.rstrip("/")
    if candidate.endswith(".git"):
        candidate = candidate[:-4]
    for sep in ("/", "\\"):
        if sep in candidate:
            candidate = candidate.split(sep)[-1]
    if not candidate or candidate in {".", ".."}:
        candidate = os.path.basename(os.path.abspath(root)) if root else "This project"
    return candidate


def _clean_block(block: str) -> str:
    return " ".join(
        line.strip().lstrip("#").strip()
        for line in block.splitlines()
        if line.strip()
        and not line.strip().startswith(("![", "[!", "<", "```", "|", "---", "==="))
    ).strip()


def _first_meaningful_paragraph(readme: str) -> str:
    for block in readme.split("\n\n"):
        cleaned = _clean_docstring(_clean_block(block))
        if len(cleaned) > 40:
            return cleaned
    return ""


def _readme_overview(readme: str, max_chars: int = 700) -> str:
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
    return "\n\n".join(paras)[:max_chars].rstrip()


def _synthesize_from_profile(profile: RepoProfile, readme: str) -> str:
    """Multi-section explanation from deep analysis — not README copy-paste."""
    sections: list[str] = []

    # --- 1. What it is (README + semantic purpose retrieval) ---
    lead = _first_meaningful_paragraph(readme)
    purpose = next((i for i in profile.insights if i.get("aspect") == "purpose" and i.get("summary")), None)
    opener = f"{profile.name} is a {profile.project_type}."
    if lead:
        body = lead.rstrip(".")
        if body.lower().startswith(profile.name.lower()):
            sections.append(body + ".")
        else:
            sections.append(f"{opener} {body}.")
    elif purpose and purpose.get("summary"):
        sections.append(f"{opener} {purpose['summary'].rstrip('.')}.")
    else:
        sections.append(opener)

    # --- 2. Tech stack ---
    if profile.tech_stack:
        sections.append("Tech stack: " + ", ".join(profile.tech_stack) + ".")

    # --- 3. Architecture ---
    arch_parts: list[str] = []
    if profile.architecture_layers:
        layer_bits = [f"{name} ({count})" for name, count in profile.architecture_layers[:4]]
        arch_parts.append("layers include " + ", ".join(layer_bits))
    if profile.languages:
        arch_parts.append(f"primary language is {profile.languages[0][0]}")
    meaningful_dirs = [
        d for d, _ in profile.top_dirs
        if d not in {".", profile.name, profile.languages[0][0] if profile.languages else ""}
    ][:3]
    if meaningful_dirs:
        arch_parts.append("top folders are " + ", ".join(meaningful_dirs))
    if arch_parts:
        sections.append("Structure: " + "; ".join(arch_parts) + ".")

    # --- 4. Core components (ranked by centrality + docs) ---
    if profile.components:
        bullets: list[str] = []
        for comp in profile.components[:6]:
            line = comp.symbol
            if comp.summary:
                line += f" — {comp.summary.rstrip('.')}"
            if comp.role:
                line += f" ({comp.role})"
            bullets.append("• " + line)
        sections.append("Core components:\n" + "\n".join(bullets))

    # --- 5. How it works (workflows + semantic retrieval) ---
    flow_lines = list(profile.workflows)
    for aspect in ("workflow", "api", "data"):
        hit = next((i for i in profile.insights if i.get("aspect") == aspect and i.get("summary")), None)
        if hit:
            sym = hit.get("symbol") or hit.get("location", "")
            line = f"{aspect.title()} insight: {sym} — {hit['summary'].rstrip('.')}."
            if line not in flow_lines:
                flow_lines.append(line)
    if flow_lines:
        sections.append("How it works:\n" + "\n".join("• " + f.rstrip(".") + "." for f in flow_lines[:4]))

    # --- 6. Getting started ---
    if profile.entry_points:
        sections.append("Start exploring from: " + ", ".join(profile.entry_points) + ".")

    return "\n\n".join(sections)


# Ask themes — how/why questions.
_ASK_THEMES = {
    "load": "How is data loaded?",
    "read": "How is input data read?",
    "parse": "How is parsing handled?",
    "connect": "How are connections established?",
    "auth": "How is authentication handled?",
    "login": "How does login work?",
    "train": "How is the model trained?",
    "predict": "How does prediction work?",
    "request": "How are HTTP requests processed?",
    "route": "How is routing handled?",
    "cache": "How is caching implemented?",
    "session": "How are sessions managed?",
    "middleware": "What middleware runs on each request?",
    "config": "How is configuration loaded?",
}

# Search themes — code-finding phrases.
_SEARCH_THEMES = {
    "load": "configuration loader function",
    "read": "read input data parser",
    "parse": "parsing implementation",
    "connect": "database connection open",
    "auth": "authentication verify credentials",
    "login": "login handler",
    "request": "HTTP request send retry",
    "route": "URL route handler mapping",
    "cache": "cache store invalidate",
    "session": "session cookie jar",
    "middleware": "middleware chain hook",
    "redirect": "redirect follow logic",
    "cookie": "cookie merge set",
    "retry": "retry backoff adapter",
    "validate": "input validation schema",
    "error": "error handling exception",
}


def suggest_ask_questions(index: CodeIndex, profile: RepoProfile | None = None, max_questions: int = 6) -> list[str]:
    profile = profile or analyze_repo(index, index.root or ".", _clean_project_name(getattr(index, "origin", "") or "", index.root or "."))
    questions: list[str] = []
    seen: set[str] = set()

    def add(q: str) -> None:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            questions.append(q)

    for comp in profile.components[:4]:
        if comp.kind == "class":
            add(f"What is the role of {comp.symbol} in the architecture?")
        elif comp.kind in {"function", "method"}:
            add(f"How does {comp.symbol} work?")
        if len(questions) >= 3:
            break

    for layer, _ in profile.architecture_layers[:2]:
        if "API" in layer:
            add("How does a request flow through the API layer?")
        elif "auth" in layer.lower() or "security" in layer.lower():
            add("How is security enforced across requests?")
        elif "model" in layer.lower() or "data" in layer.lower():
            add("How is data modeled and persisted?")

    if profile.main_package:
        add(f"What lives in the {profile.main_package.split('/')[-1]} package?")

    tokens: set[str] = set()
    for chunk in index.chunks:
        tokens.update(tokenize(chunk.symbol))
    for key, q in _ASK_THEMES.items():
        if key in tokens:
            add(q)

    add("What are the main entry points of this project?")
    add("How is this project structured?")

    return questions[:max_questions]


def suggest_search_queries(index: CodeIndex, profile: RepoProfile | None = None, max_queries: int = 6) -> list[str]:
    profile = profile or analyze_repo(index, index.root or ".", _clean_project_name(getattr(index, "origin", "") or "", index.root or "."))
    queries: list[str] = []
    seen: set[str] = set()

    def add(q: str) -> None:
        key = q.lower()
        if key not in seen and q.strip():
            seen.add(key)
            queries.append(q)

    for comp in profile.components[:3]:
        if comp.role:
            add(comp.role.lower())
        elif comp.summary:
            phrase = comp.summary.rstrip(".")[:60]
            if len(phrase) > 12:
                add(phrase[0].lower() + phrase[1:])
        elif comp.kind == "class":
            add(comp.symbol.replace("_", " ") + " class")
        else:
            add(comp.symbol.replace("_", " "))

    if profile.main_package:
        prefix = profile.main_package.replace("\\", "/")
        module_hints = {
            "auth": "authentication handler",
            "cookies": "cookie handling",
            "sessions": "session management",
            "adapters": "HTTP adapter transport",
            "models": "request response models",
            "utils": "utility helpers",
        }
        for path in sorted({c.path for c in index.chunks}):
            norm = path.replace("\\", "/")
            if not norm.startswith(prefix + "/") or not norm.endswith(".py"):
                continue
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem.startswith("_") or stem in {"__init__", "__version__", "compat", "certs", "packages"}:
                continue
            add(module_hints.get(stem, stem.replace("_", " ")))
            if len(queries) >= max_queries - 2:
                break

    for layer, _ in profile.architecture_layers[:2]:
        add(layer.lower())

    for ins in profile.insights:
        if ins.get("aspect") in {"api", "data", "workflow", "config"} and ins.get("symbol"):
            sym = ins["symbol"].split(".")[0]
            if sym and sym.lower() not in {"api", "models", "utils"}:
                add(sym.replace("_", " "))
        if len(queries) >= max_queries - 2:
            break

    tokens: set[str] = set()
    for chunk in index.chunks:
        tokens.update(tokenize(chunk.symbol))
    for key, phrase in _SEARCH_THEMES.items():
        if key in tokens:
            add(phrase)

    skip = {"__init__", "main", "index", "utils", "util", "test", "tests", "conftest", "setup", "compat", "packages"}
    if not profile.main_package:
        for path in sorted({c.path for c in index.chunks}):
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem and stem.lower() not in skip and len(stem) > 3:
                add(stem.replace("_", " "))
            if len(queries) >= max_queries - 1:
                break

    add("error handling")

    return queries[:max_queries]


def suggest_questions(index: CodeIndex, max_questions: int = 6) -> list[str]:
    return suggest_ask_questions(index, max_questions=max_questions)


def _build_llm_context(profile: RepoProfile, readme: str) -> str:
    lines = [
        f"Repository: {profile.name}",
        f"Project type: {profile.project_type}",
        f"Files indexed: {profile.num_files}",
        f"Languages: {profile.languages}",
        f"Tech stack: {profile.tech_stack}",
        f"Dependencies (sample): {profile.dependencies[:15]}",
        f"Architecture layers: {profile.architecture_layers}",
        f"Entry points: {profile.entry_points}",
        f"Workflows: {profile.workflows}",
        f"Top components: {[c.to_dict() for c in profile.components[:8]]}",
        f"Semantic retrieval insights: {profile.insights[:10]}",
    ]
    if readme:
        lines.append("README excerpt:\n" + readme[:1000])
    return "\n".join(lines)


def _llm_description(client: LLMClient, profile: RepoProfile, readme: str) -> str:
    system = (
        "You are a principal engineer writing a production-quality project brief "
        "for developers joining the team. Use ONLY the provided analysis. "
        "Write 4-6 short paragraphs covering: (1) what the project is and why it "
        "exists, (2) tech stack and architecture, (3) core modules and their "
        "responsibilities, (4) how data/request flow works, (5) where to start "
        "reading the code. Do not copy the README verbatim — synthesize and explain."
    )
    user = "Analyse and explain this repository:\n\n" + _build_llm_context(profile, readme)
    return client.complete(system, user)


def summarize_repo(
    index: CodeIndex,
    root: str | None = None,
    use_llm: bool | str = "auto",
    llm_client: LLMClient | None = None,
) -> RepoSummary:
    """Build a :class:`RepoSummary` for an indexed repository."""
    root = root or index.root or "."
    readme = _read_readme(root)
    name = _clean_project_name(getattr(index, "origin", "") or "", root)
    profile = analyze_repo(index, root, name, readme=readme)

    client = llm_client or LLMClient()
    want_llm = client.available() if use_llm == "auto" else bool(use_llm)

    description = ""
    llm_used = False
    if want_llm and client.available():
        try:
            description = _llm_description(client, profile, readme)
            llm_used = bool(description)
        except Exception:
            description = ""
    if not description:
        description = _synthesize_from_profile(profile, readme)

    components = [c.to_dict() for c in profile.components[:8]]

    return RepoSummary(
        root=root,
        name=name,
        num_files=profile.num_files,
        num_chunks=len(index),
        languages=profile.languages,
        kinds=profile.kinds,
        top_dirs=profile.top_dirs,
        readme_excerpt=_first_meaningful_paragraph(readme),
        notable_symbols=profile.notable_symbols,
        description=description,
        llm_used=llm_used,
        extra={
            "project_type": profile.project_type,
            "tech_stack": profile.tech_stack,
            "architecture_layers": profile.architecture_layers,
            "workflows": profile.workflows,
            "insights": profile.insights,
            "components": components,
            "entry_points": profile.entry_points,
            "readme_overview": _readme_overview(readme),
            "suggested_ask_questions": suggest_ask_questions(index, profile),
            "suggested_search_queries": suggest_search_queries(index, profile),
        },
    )
