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


def _heuristic_answer(question: str, results: list[SearchResult]) -> str:
    if not results:
        return "No relevant code was found for this question."
    lines = [
        "No LLM is configured, so here are the most relevant code locations "
        "(set OPENAI_API_KEY and install the 'llm' extra for synthesised answers):",
        "",
    ]
    for i, result in enumerate(results, 1):
        chunk = result.chunk
        label = f"{chunk.kind} {chunk.symbol}".strip() if chunk.symbol else chunk.kind
        summary = chunk.docstring.strip().splitlines()[0] if chunk.docstring else ""
        line = f"  [{i}] {chunk.location}  ({label}, score {result.score:.3f})"
        if summary:
            line += f"\n      {summary}"
        lines.append(line)
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
