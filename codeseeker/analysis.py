"""Deep, deterministic repository analysis for production-grade explanations.

Builds a structured :class:`RepoProfile` from indexed code, config files,
import patterns, semantic retrieval, and layout signals. Works fully offline;
no LLM required.
"""

from __future__ import annotations

import ast
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from codeseeker.embeddings import tokenize
from codeseeker.index import CodeIndex, SearchResult

# Config files that reveal tech stack and purpose.
_CONFIG_FILES = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "requirements-dev.txt",
    "Pipfile",
    "package.json",
    "go.mod",
    "Cargo.toml",
    "pom.xml",
    "build.gradle",
    "Gemfile",
    "composer.json",
)

# Path segment -> architectural role.
_LAYER_PATTERNS: list[tuple[str, str]] = [
    (r"(^|/)(api|routes|controllers|handlers|views|endpoints)(/|$)", "API / routing"),
    (r"(^|/)(models|schema|entities|domain)(/|$)", "Data models"),
    (r"(^|/)(services|service|core|engine)(/|$)", "Core services"),
    (r"(^|/)(cli|cmd|commands)(/|$)", "CLI"),
    (r"(^|/)(utils|util|helpers|common|lib)(/|$)", "Utilities"),
    (r"(^|/)(middleware|auth|security)(/|$)", "Middleware / security"),
    (r"(^|/)(config|settings)(/|$)", "Configuration"),
    (r"(^|/)(db|database|storage|repository|repos)(/|$)", "Persistence"),
    (r"(^|/)(tests?|spec)(/|$)", "Tests"),
]

# Import / dependency name fragments -> human label.
_FRAMEWORK_HINTS: dict[str, str] = {
    "flask": "Flask",
    "django": "Django",
    "fastapi": "FastAPI",
    "starlette": "Starlette",
    "requests": "Requests",
    "sqlalchemy": "SQLAlchemy",
    "pandas": "pandas",
    "numpy": "NumPy",
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "sklearn": "scikit-learn",
    "pytest": "pytest",
    "react": "React",
    "express": "Express",
    "gin": "Gin",
    "axios": "axios",
    "pydantic": "Pydantic",
    "celery": "Celery",
    "redis": "Redis",
    "boto3": "AWS SDK",
    "grpc": "gRPC",
    "graphql": "GraphQL",
    "jwt": "JWT",
    "oauth": "OAuth",
}

_ENTRY_BASENAMES = frozenset(
    {"main.py", "app.py", "cli.py", "__main__.py", "manage.py", "server.py", "run.py", "index.js", "main.go"}
)

# Semantic retrieval queries for multi-aspect understanding (offline RAG).
_ASPECT_QUERIES: list[tuple[str, str]] = [
    ("purpose", "main purpose core functionality what does this project do"),
    ("architecture", "architecture design modules layers package structure"),
    ("api", "public API interface client handler route endpoint"),
    ("data", "data model schema database persistence storage"),
    ("workflow", "request flow pipeline process orchestration lifecycle"),
    ("config", "configuration settings environment setup initialization"),
]


@dataclass
class ComponentInfo:
    symbol: str
    kind: str
    location: str
    summary: str
    role: str = ""
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "kind": self.kind,
            "location": self.location,
            "summary": self.summary,
            "role": self.role,
            "score": round(self.score, 3),
        }


@dataclass
class RepoProfile:
    """Structured understanding of a repository."""

    name: str
    project_type: str
    tech_stack: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    architecture_layers: list[tuple[str, int]] = field(default_factory=list)
    workflows: list[str] = field(default_factory=list)
    components: list[ComponentInfo] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    insights: list[dict] = field(default_factory=list)
    num_files: int = 0
    languages: list[tuple[str, int]] = field(default_factory=list)
    kinds: dict[str, int] = field(default_factory=dict)
    top_dirs: list[tuple[str, int]] = field(default_factory=list)
    notable_symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "project_type": self.project_type,
            "tech_stack": self.tech_stack,
            "dependencies": self.dependencies[:12],
            "architecture_layers": self.architecture_layers,
            "workflows": self.workflows,
            "components": [c.to_dict() for c in self.components],
            "entry_points": self.entry_points,
            "insights": self.insights,
            "num_files": self.num_files,
            "languages": self.languages,
            "kinds": self.kinds,
            "top_dirs": self.top_dirs,
            "notable_symbols": self.notable_symbols,
        }


def _first_sentence(text: str, limit: int = 200) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    for sep in (". ", "! ", "? "):
        idx = text.find(sep)
        if 0 < idx < limit:
            return text[: idx + 1]
    return text[:limit] + ("…" if len(text) > limit else "")


def _read_text_file(path: str, max_bytes: int = 50_000) -> str:
    try:
        with open(path, "rb") as fh:
            data = fh.read(max_bytes)
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _parse_requirements(text: str) -> list[str]:
    deps: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip()
        if name:
            deps.append(name)
    return deps


def _parse_package_json(text: str) -> list[str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    deps: list[str] = []
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        section = data.get(key) or {}
        if isinstance(section, dict):
            deps.extend(section.keys())
    return deps


def _parse_pyproject(text: str) -> list[str]:
    deps: list[str] = []
    in_deps = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "dependencies" in stripped.lower():
            in_deps = True
            continue
        if stripped.startswith("[") and in_deps:
            break
        if in_deps and "=" in stripped:
            name = stripped.split("=", 1)[0].strip().strip('"').strip("'")
            if name and name not in {"python"}:
                deps.append(name)
    # Also catch poetry-style inline lists.
    for match in re.finditer(r'["\']([a-zA-Z0-9_.-]+)["\']\s*=', text):
        deps.append(match.group(1))
    return deps


def _scan_dependencies(root: str) -> list[str]:
    """Read known config files and extract dependency names."""
    found: list[str] = []
    seen: set[str] = set()
    parsers = {
        "requirements.txt": _parse_requirements,
        "requirements-dev.txt": _parse_requirements,
        "package.json": _parse_package_json,
        "pyproject.toml": _parse_pyproject,
    }
    for fname in _CONFIG_FILES:
        path = os.path.join(root, fname)
        if not os.path.isfile(path):
            continue
        text = _read_text_file(path)
        if not text:
            continue
        parser = parsers.get(fname)
        names = parser(text) if parser else []
        if not names and fname.endswith(".txt"):
            names = _parse_requirements(text)
        for name in names:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                found.append(name)
    return found


def _frameworks_from_deps(deps: list[str]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    blob = " ".join(deps).lower()
    for key, label in _FRAMEWORK_HINTS.items():
        if key in blob and label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _extract_imports_from_chunk(text: str) -> set[str]:
    names: set[str] = set()
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
    except SyntaxError:
        for line in text.splitlines():
            m = re.match(r"^\s*(?:from|import)\s+([\w.]+)", line)
            if m:
                names.add(m.group(1).split(".")[0])
    return names


def _collect_imports(index: CodeIndex) -> Counter:
    counts: Counter = Counter()
    for chunk in index.chunks:
        if chunk.language != "python":
            continue
        for name in _extract_imports_from_chunk(chunk.text):
            counts[name] += 1
    return counts


def _classify_layer(path: str) -> str | None:
    norm = path.replace("\\", "/").lower()
    for pattern, label in _LAYER_PATTERNS:
        if re.search(pattern, norm):
            return label
    return None


def _symbol_centrality(index: CodeIndex) -> Counter:
    """How often each symbol name appears across the indexed codebase."""
    counts: Counter = Counter()
    symbols = [c.symbol for c in index.chunks if c.symbol]
    if not symbols:
        return counts
    # Build one blob per chunk for substring search (cheap, good enough offline).
    bodies = [c.text for c in index.chunks]
    for sym in set(symbols):
        base = sym.split(".")[-1]
        if len(base) < 3:
            continue
        pattern = re.compile(r"\b" + re.escape(base) + r"\b")
        for body in bodies:
            if pattern.search(body):
                counts[sym] += 1
    return counts


def _rank_components(index: CodeIndex, centrality: Counter) -> list[ComponentInfo]:
    """Pick the most architecturally important symbols."""
    candidates: list[ComponentInfo] = []
    seen: set[str] = set()
    kind_weight = {"class": 3.0, "function": 2.0, "method": 1.5, "module": 1.0}

    for chunk in index.chunks:
        if not chunk.symbol or chunk.symbol in seen:
            continue
        if chunk.kind in {"block", "cell", "note"}:
            continue
        seen.add(chunk.symbol)
        layer = _classify_layer(chunk.path) or ""
        score = centrality.get(chunk.symbol, 0) * 2.0
        score += kind_weight.get(chunk.kind, 1.0)
        score += min(len(chunk.docstring), 200) / 50.0
        if layer and layer != "Tests":
            score += 2.0
        if os.path.basename(chunk.path).lower() in _ENTRY_BASENAMES:
            score += 3.0
        role = layer or _infer_role_from_symbol(chunk.symbol, chunk.docstring)
        candidates.append(
            ComponentInfo(
                symbol=chunk.symbol,
                kind=chunk.kind,
                location=chunk.location,
                summary=_first_sentence(chunk.docstring),
                role=role,
                score=score,
            )
        )

    candidates.sort(key=lambda c: (-c.score, c.symbol))
    return candidates[:12]


def _infer_role_from_symbol(symbol: str, docstring: str) -> str:
    text = (symbol + " " + docstring).lower()
    if any(w in text for w in ("client", "session", "request", "response")):
        return "HTTP client / transport"
    if any(w in text for w in ("auth", "login", "token", "credential")):
        return "Authentication"
    if any(w in text for w in ("config", "settings", "option")):
        return "Configuration"
    if any(w in text for w in ("model", "schema", "entity")):
        return "Data model"
    if any(w in text for w in ("handler", "route", "view", "controller")):
        return "Request handling"
    if any(w in text for w in ("parser", "parse", "loader", "load")):
        return "Parsing / loading"
    return ""


def _detect_layers(index: CodeIndex) -> list[tuple[str, int]]:
    layers: Counter = Counter()
    for chunk in index.chunks:
        layer = _classify_layer(chunk.path)
        if layer and layer != "Tests":
            layers[layer] += 1
    return layers.most_common(8)


def _detect_project_type(
    profile_hints: dict,
    readme: str,
    name: str,
) -> str:
    text = (readme + " " + name + " " + " ".join(profile_hints.get("frameworks", []))).lower()
    kinds = profile_hints.get("kinds", {})
    layers = [l for l, _ in profile_hints.get("layers", [])]

    if any(w in text for w in ("machine learning", "regression", "classification", "forecast", "neural", "pytorch", "tensorflow")):
        return "machine learning / data science"
    if kinds.get("cell") or kinds.get("note"):
        return "data science / notebook project"
    if "API / routing" in layers or any(w in text for w in ("rest api", "web api", "http api")):
        return "web API / HTTP service"
    if any(w in text for w in ("flask", "django", "fastapi", "web app")):
        return "web application"
    if any(w in text for w in ("library", "package", "sdk", "client")):
        return "software library"
    if profile_hints.get("entry_points"):
        return "application"
    return "software project"


def _infer_workflows(
    entry_points: list[str],
    components: list[ComponentInfo],
    insights: list[dict],
) -> list[str]:
    """Describe likely execution / data flows from structure + retrieval."""
    flows: list[str] = []
    if entry_points:
        flows.append(
            "Startup likely begins at " + ", ".join(entry_points[:3]) + "."
        )
    api_comps = [c for c in components if "API" in c.role or "HTTP" in c.role or "handler" in c.role.lower()]
    if api_comps:
        names = ", ".join(c.symbol for c in api_comps[:3])
        flows.append(f"HTTP-style handling centers on {names}.")
    auth_comps = [c for c in components if "auth" in c.role.lower() or "auth" in c.symbol.lower()]
    if auth_comps:
        flows.append(f"Authentication logic appears around {auth_comps[0].symbol}.")
    data_comps = [c for c in components if "model" in c.role.lower() or "persist" in c.role.lower()]
    if data_comps:
        flows.append(f"Data/persistence layer includes {data_comps[0].symbol}.")

    for item in insights:
        if item.get("aspect") == "workflow" and item.get("summary"):
            line = item["summary"]
            if line not in flows:
                flows.append(line)
            break

    return flows[:5]


def _retrieve_insights(index: CodeIndex, top_k: int = 2) -> list[dict]:
    """Multi-query semantic retrieval — offline RAG for project understanding."""
    insights: list[dict] = []
    seen: set[str] = set()
    for aspect, query in _ASPECT_QUERIES:
        try:
            results = index.search(query, top_k=top_k, mode="hybrid", semantic_weight=0.75)
        except TypeError:
            results = index.search(query, top_k=top_k)
        for result in results:
            chunk = result.chunk
            key = chunk.location
            if key in seen:
                continue
            seen.add(key)
            summary = _first_sentence(chunk.docstring) if chunk.docstring else ""
            if not summary:
                # Fall back to first meaningful code line.
                for line in chunk.text.splitlines():
                    s = line.strip()
                    if s and not s.startswith(("#", '"""', "'''")):
                        summary = s[:120]
                        break
            insights.append(
                {
                    "aspect": aspect,
                    "location": chunk.location,
                    "symbol": chunk.symbol,
                    "kind": chunk.kind,
                    "score": round(float(result.score), 4),
                    "summary": summary,
                }
            )
    return insights


def analyze_repo(
    index: CodeIndex,
    root: str,
    name: str,
    readme: str = "",
) -> RepoProfile:
    """Build a full structured profile from an indexed repository."""
    files = {c.path for c in index.chunks}
    languages = Counter(c.language for c in index.chunks)
    kinds = Counter(c.kind for c in index.chunks)

    top_dirs: Counter = Counter()
    for path in files:
        head = path.split(os.sep)[0] if os.sep in path else path.split("/")[0]
        if head and (os.sep in path or "/" in path):
            top_dirs[head] += 1

    entry_points = sorted(
        p for p in files if os.path.basename(p).lower() in _ENTRY_BASENAMES
    )

    dependencies = _scan_dependencies(root) if root and os.path.isdir(root) else []
    import_counts = _collect_imports(index)
    import_names = [name for name, _ in import_counts.most_common(30)]
    frameworks = _frameworks_from_deps(dependencies + import_names)

    tech_stack = list(dict.fromkeys(
        [lang for lang, _ in languages.most_common(3)] + frameworks
    ))

    layers = _detect_layers(index)
    centrality = _symbol_centrality(index)
    components = _rank_components(index, centrality)
    insights = _retrieve_insights(index)

    profile_hints = {
        "kinds": dict(kinds),
        "layers": layers,
        "frameworks": frameworks,
        "entry_points": entry_points,
    }
    project_type = _detect_project_type(profile_hints, readme, name)
    workflows = _infer_workflows(entry_points, components, insights)

    notable = [f"{c.symbol} ({c.kind})" for c in components[:12]]

    return RepoProfile(
        name=name,
        project_type=project_type,
        tech_stack=tech_stack,
        dependencies=dependencies[:20],
        architecture_layers=layers,
        workflows=workflows,
        components=components,
        entry_points=entry_points[:5],
        insights=insights,
        num_files=len(files),
        languages=languages.most_common(),
        kinds=dict(kinds),
        top_dirs=top_dirs.most_common(8),
        notable_symbols=notable,
    )
