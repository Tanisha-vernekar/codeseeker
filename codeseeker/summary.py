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


def _heuristic_description(name: str, signals: dict, readme: str) -> str:
    lines: list[str] = []

    # 1. What it is (from README, the most reliable human summary).
    if readme:
        first_para = _first_meaningful_paragraph(readme)
        if first_para:
            lines.append(f"{name} — {first_para}")
    if not lines:
        lines.append(f"{name} is a software project.")

    # 2. Size & languages.
    langs = signals["languages"]
    if langs:
        lang_str = ", ".join(f"{lang} ({count})" for lang, count in langs[:4])
        lines.append(
            f"\nCodebase: {signals['num_files']} source files · main languages: {lang_str}."
        )

    kinds = signals["kinds"]
    parts = []
    for label in ("class", "function", "method"):
        if kinds.get(label):
            parts.append(f"{kinds[label]} {label}{'es' if label == 'class' else 's'}")
    if parts:
        lines.append("It defines " + ", ".join(parts) + ".")

    # 3. Key components with their own documentation — the useful part.
    components = signals.get("components") or []
    if components:
        lines.append("\nKey components:")
        for comp in components[:6]:
            summary = f" — {comp['summary']}" if comp.get("summary") else ""
            lines.append(f"  • {comp['symbol']} ({comp['kind']}){summary}")

    # 4. Entry points, if any.
    if signals.get("entry_points"):
        lines.append("\nLikely entry points: " + ", ".join(signals["entry_points"]) + ".")

    return "\n".join(lines).strip()


def _first_meaningful_paragraph(readme: str) -> str:
    for block in readme.split("\n\n"):
        cleaned = " ".join(
            line.strip().lstrip("#").strip()
            for line in block.splitlines()
            if line.strip() and not line.strip().startswith(("![", "[!", "<"))
        ).strip()
        # Skip a lone title line; look for a real sentence.
        if len(cleaned) > 40:
            return cleaned
    return ""


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
        description = _heuristic_description(name, signals, readme)

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
        },
    )
