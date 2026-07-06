"""Answer natural-language questions grounded in retrieved code (RAG).

``ask`` retrieves the most relevant code chunks for a question and, when an LLM
is configured, uses them as grounded context to synthesise an answer. Without
an LLM it returns a deterministic, citation-first response that points at the
most relevant code so the feature still adds value offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from codeseeker.index import CodeIndex, SearchResult
from codeseeker.llm import LLMClient


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    llm_used: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "llm_used": self.llm_used,
            "sources": [s.to_dict() for s in self.sources],
        }


def _build_context(results: list[SearchResult], max_chars: int = 6000) -> str:
    blocks: list[str] = []
    used = 0
    for i, result in enumerate(results, 1):
        chunk = result.chunk
        header = f"[{i}] {chunk.location}"
        if chunk.symbol:
            header += f" — {chunk.kind} {chunk.symbol}"
        body = chunk.text.rstrip()
        block = f"{header}\n{body}"
        if used + len(block) > max_chars:
            block = block[: max(0, max_chars - used)]
        blocks.append(block)
        used += len(block)
        if used >= max_chars:
            break
    return "\n\n".join(blocks)


def _first_sentence(text: str, limit: int = 240) -> str:
    """Return the first sentence/line of a docstring, trimmed."""
    text = " ".join((text or "").split())
    if not text:
        return ""
    for sep in (". ", "! ", "? "):
        idx = text.find(sep)
        if 0 < idx < limit:
            return text[: idx + 1]
    return text[:limit] + ("…" if len(text) > limit else "")


def _signature(chunk) -> str:
    """Return the first meaningful code line (def/class signature)."""
    for line in chunk.text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", '"""', "'''", "@")):
            return stripped.rstrip(":")
    return ""


def _heuristic_answer(question: str, results: list[SearchResult]) -> str:
    """Compose a readable, grounded answer directly from retrieved code.

    This works fully offline (no LLM). It reads like an answer instead of a
    raw list of locations, summarising the most relevant symbol and pointing at
    related code.
    """
    if not results:
        return (
            "I couldn't find anything relevant in the indexed code for that "
            "question. Try rephrasing, or index a repository first."
        )

    top = results[0].chunk
    label = f"{top.kind} `{top.symbol}`" if top.symbol else top.kind
    desc = _first_sentence(top.docstring) if top.docstring else ""

    lines: list[str] = []
    lead = f"This is handled mainly by {label} in {top.location}."
    if desc:
        lead += f" {desc}"
    lines.append(lead)

    sig = _signature(top)
    if sig:
        lines.append("")
        lines.append("Key definition:")
        lines.append(f"    {sig}")

    others = results[1:5]
    if others:
        lines.append("")
        lines.append("Related code:")
        for result in others:
            chunk = result.chunk
            lbl = f"{chunk.kind} {chunk.symbol}".strip() if chunk.symbol else chunk.kind
            one = _first_sentence(chunk.docstring, limit=120) if chunk.docstring else ""
            line = f"  • {chunk.location} — {lbl}"
            if one:
                line += f": {one}"
            lines.append(line)

    lines.append("")
    lines.append("(Tip: set OPENAI_API_KEY for full AI-written answers.)")
    return "\n".join(lines)


def _llm_answer(client: LLMClient, question: str, results: list[SearchResult]) -> str:
    context = _build_context(results)
    system = (
        "You are a helpful engineering assistant. Answer the developer's "
        "question using ONLY the provided code context. Cite the sources you "
        "use with their bracket numbers like [1], [2]. If the context is "
        "insufficient, say so clearly."
    )
    user = f"Question: {question}\n\nCode context:\n{context}"
    return client.complete(system, user)


def answer_question(
    index: CodeIndex,
    question: str,
    top_k: int = 6,
    use_llm: bool | str = "auto",
    llm_client: LLMClient | None = None,
    **search_filters,
) -> Answer:
    """Retrieve context for ``question`` and produce an answer."""
    results = index.search(question, top_k=top_k, **search_filters)

    client = llm_client or LLMClient()
    want_llm = client.available() if use_llm == "auto" else bool(use_llm)

    text = ""
    llm_used = False
    if want_llm and client.available() and results:
        try:
            text = _llm_answer(client, question, results)
            llm_used = bool(text)
        except Exception:
            text = ""
    if not text:
        text = _heuristic_answer(question, results)

    return Answer(question=question, answer=text, sources=results, llm_used=llm_used)
