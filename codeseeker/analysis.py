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

_CODE_LANGUAGES = frozenset(
    {
        "python", "javascript", "typescript", "go", "rust", "java", "kotlin",
        "c", "cpp", "csharp", "ruby", "php", "swift", "scala", "shell", "sql", "r", "julia",
    }
)

_NOISE_DIR_HEADS = frozenset({".github", ".gitlab", "docs", "doc", "examples", "example"})

_GENERIC_METHOD_NAMES = frozenset(
    {
        "get", "post", "put", "patch", "delete", "head", "options",
        "keys", "values", "items", "update", "pop", "clear", "copy",
        "send", "read", "write", "close", "flush", "open", "run",
        "__init__", "__str__", "__repr__", "__enter__", "__exit__",
    }
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
    main_package: str = ""
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
            "main_package": self.main_package,
            "num_files": self.num_files,
            "languages": self.languages,
            "kinds": self.kinds,
            "top_dirs": self.top_dirs,
            "notable_symbols": self.notable_symbols,
        }


def _first_sentence(text: str, limit: int = 200) -> str:
    text = _clean_docstring(text)
    if not text:
        return ""
    for sep in (". ", "! ", "? "):
        idx = text.find(sep)
        if 0 < idx < limit:
            return text[: idx + 1]
    return text[:limit] + ("…" if len(text) > limit else "")


def _clean_docstring(text: str) -> str:
    """Strip RST/Markdown markup so GitHub-repo docstrings read cleanly."""
    if not text:
        return ""
    cleaned = text
    cleaned = re.sub(r":\w+:`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r":\w+:`([^<`]+)\s*<[^>]+>`", r"\1", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"(\w+)\s*<\1>", r"\1", cleaned)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    return " ".join(cleaned.split())


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


def _detect_main_package(root: str, name: str, files: set[str]) -> str:
    """Find the primary source package for monorepos (e.g. src/requests)."""
    norm_files = {f.replace("\\", "/") for f in files}
    candidates = [
        f"src/{name}",
        f"{name}",
        f"lib/{name}",
        f"pkg/{name}",
        f"{name}/{name}",
    ]
    for cand in candidates:
        if any(f.startswith(cand + "/") or f == cand for f in norm_files):
            return cand
    # Fall back: most common second-level directory under src/.
    under_src: Counter = Counter()
    for path in norm_files:
        parts = path.split("/")
        if len(parts) >= 2 and parts[0] == "src":
            under_src[parts[1]] += 1
    if under_src:
        return f"src/{under_src.most_common(1)[0][0]}"
    return ""


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
        for dep in names:
            key = dep.lower()
            if key not in seen:
                seen.add(key)
                found.append(dep)
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


def _rank_components(
    index: CodeIndex,
    centrality: Counter,
    main_package: str = "",
) -> list[ComponentInfo]:
    """Pick the most architecturally important symbols (GitHub-repo aware)."""
    candidates: list[ComponentInfo] = []
    seen: set[str] = set()
    kind_weight = {"class": 5.0, "function": 3.0, "method": 1.0, "module": 2.5}

    for chunk in index.chunks:
        if not chunk.symbol or chunk.symbol in seen:
            continue
        if chunk.kind in {"block", "cell", "note"}:
            continue
        if chunk.language not in _CODE_LANGUAGES:
            continue

        base_name = chunk.symbol.split(".")[-1]
        if chunk.kind == "method" and base_name.lower() in _GENERIC_METHOD_NAMES:
            continue
        if chunk.kind == "function" and base_name.lower() in {"get", "post", "put", "patch", "delete", "head"}:
            continue

        seen.add(chunk.symbol)
        norm_path = chunk.path.replace("\\", "/")
        layer = _classify_layer(chunk.path) or ""
        score = centrality.get(chunk.symbol, 0) * 1.5
        score += kind_weight.get(chunk.kind, 1.0)
        score += min(len(chunk.docstring), 200) / 40.0

        if chunk.kind == "class" and "." not in chunk.symbol:
            score += 6.0
        if chunk.kind == "class" and base_name[:1].isupper():
            score += 4.0
        if chunk.kind == "function" and base_name.islower() and len(base_name) <= 8:
            score -= 4.0
        stem = os.path.splitext(os.path.basename(chunk.path))[0]
        sym_root = chunk.symbol.split(".")[0]
        if sym_root.lower() == stem.lower():
            score += 5.0
        elif sym_root.lower() in stem.lower() or stem.lower().startswith(sym_root.lower()):
            score += 3.0
        if main_package and norm_path.startswith(main_package + "/"):
            score += 3.0
        if layer and layer != "Tests":
            score += 2.0
        if os.path.basename(chunk.path).lower() in _ENTRY_BASENAMES:
            score += 3.0
        if chunk.kind == "module" and norm_path.endswith("__init__.py"):
            score += 2.0

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
    # Prefer diverse, high-level symbols over generic helpers.
    picked: list[ComponentInfo] = []
    used_roots: set[str] = set()
    for comp in candidates:
        root = comp.symbol.split(".")[0]
        if comp.kind == "method" and root in used_roots:
            continue
        if comp.kind == "method" and any(
            c.symbol == root and c.kind == "class" for c in candidates
        ):
            continue
        if comp.symbol == "request" and any(c.symbol == "Session" for c in candidates):
            continue
        picked.append(comp)
        used_roots.add(root)
        if len(picked) >= 12:
            break
    return picked


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
    if any(w in text for w in ("http library", "http client", "http requests")):
        return "HTTP client library"
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
    main_package: str = "",
) -> list[str]:
    """Describe likely execution / data flows from structure + retrieval."""
    flows: list[str] = []
    if entry_points:
        flows.append("Start from " + ", ".join(entry_points[:3]) + ".")

    classes = [c for c in components if c.kind == "class"][:3]
    if classes:
        flows.append(
            "Core types: " + ", ".join(c.symbol for c in classes) + "."
        )

    api_comps = [c for c in components if c.role and ("HTTP" in c.role or "API" in c.role)]
    if api_comps:
        flows.append(
            "Typical HTTP flow goes through "
            + ", ".join(c.symbol for c in api_comps[:3])
            + "."
        )

    auth_comps = [c for c in components if "auth" in c.role.lower() or "auth" in c.symbol.lower()]
    if auth_comps:
        flows.append(f"Auth helpers live in {auth_comps[0].symbol}.")

    if main_package:
        flows.append(f"Main package code is under {main_package}/.")

    for item in insights:
        if item.get("aspect") == "workflow" and item.get("summary"):
            line = _clean_docstring(item["summary"])
            if line and line not in flows:
                flows.append(line.rstrip(".") + ".")
            break

    return flows[:5]


def _retrieve_insights(index: CodeIndex, top_k: int = 2, main_package: str = "") -> list[dict]:
    """Multi-query semantic retrieval — offline RAG for project understanding."""
    insights: list[dict] = []
    seen: set[str] = set()
    for aspect, query in _ASPECT_QUERIES:
        try:
            kwargs: dict = {"top_k": top_k, "mode": "hybrid", "semantic_weight": 0.75}
            if main_package:
                kwargs["path_contains"] = main_package.split("/")[-1]
            results = index.search(query, **kwargs)
        except TypeError:
            results = index.search(query, top_k=top_k)
        for result in results:
            chunk = result.chunk
            if chunk.language not in _CODE_LANGUAGES:
                continue
            if chunk.kind in {"block", "note"}:
                continue
            key = chunk.location
            if key in seen:
                continue
            seen.add(key)
            summary = _first_sentence(chunk.docstring) if chunk.docstring else ""
            if not summary:
                for line in chunk.text.splitlines():
                    s = line.strip()
                    if s and not s.startswith(("#", '"""', "'''")):
                        summary = s[:120]
                        break
            summary = _clean_docstring(summary)
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
    code_chunks = [c for c in index.chunks if c.language in _CODE_LANGUAGES]
    files = {c.path for c in code_chunks}
    languages = Counter(c.language for c in code_chunks)
    kinds = Counter(c.kind for c in code_chunks)

    top_dirs: Counter = Counter()
    for path in files:
        head = path.split(os.sep)[0] if os.sep in path else path.split("/")[0]
        if head and head not in _NOISE_DIR_HEADS and (os.sep in path or "/" in path):
            top_dirs[head] += 1

    entry_points = sorted(
        p for p in files if os.path.basename(p).lower() in _ENTRY_BASENAMES
    )

    main_package = _detect_main_package(root, name, files)
    dependencies = _scan_dependencies(root) if root and os.path.isdir(root) else []
    import_counts = _collect_imports(index)
    import_names = [n for n, _ in import_counts.most_common(30)]
    frameworks = _frameworks_from_deps(dependencies + import_names)

    code_langs = [lang for lang, _ in languages.most_common() if lang in _CODE_LANGUAGES]
    tech_stack = list(dict.fromkeys(code_langs[:2] + frameworks))
    readme_lower = readme.lower()
    if "http" in readme_lower and any(w in readme_lower for w in ("library", "client", "requests")):
        if "HTTP client" not in tech_stack:
            tech_stack.append("HTTP client")

    layers = _detect_layers(index)
    centrality = _symbol_centrality(index)
    components = _rank_components(index, centrality, main_package=main_package)
    insights = _retrieve_insights(index, main_package=main_package)

    profile_hints = {
        "kinds": dict(kinds),
        "layers": layers,
        "frameworks": frameworks,
        "entry_points": entry_points,
    }
    project_type = _detect_project_type(profile_hints, readme, name)
    workflows = _infer_workflows(entry_points, components, insights, main_package=main_package)

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
        main_package=main_package,
        num_files=len(files),
        languages=languages.most_common(),
        kinds=dict(kinds),
        top_dirs=top_dirs.most_common(8),
        notable_symbols=notable,
    )
