"""codeseeker: LLM-powered semantic code search over local & remote repos.

Public API:
    - :class:`~codeseeker.chunking.CodeChunk`
    - :func:`~codeseeker.chunking.chunk_file`
    - :class:`~codeseeker.index.CodeIndex`
    - :class:`~codeseeker.index.SearchResult`
    - :func:`~codeseeker.repo.resolve_source`
    - :func:`~codeseeker.summary.summarize_repo`
    - :func:`~codeseeker.qa.answer_question`
"""

from codeseeker.chunking import CodeChunk, chunk_file, iter_source_files
from codeseeker.embeddings import (
    Embedder,
    TfidfEmbedder,
    build_embedder,
    load_embedder,
)
from codeseeker.index import CodeIndex, SearchResult
from codeseeker.qa import Answer, answer_question
from codeseeker.repo import RepoSource, resolve_source
from codeseeker.summary import RepoSummary, summarize_repo
from codeseeker.vectorstore import faiss_available, make_vector_store

__all__ = [
    "CodeChunk",
    "chunk_file",
    "iter_source_files",
    "Embedder",
    "TfidfEmbedder",
    "build_embedder",
    "load_embedder",
    "CodeIndex",
    "SearchResult",
    "RepoSource",
    "resolve_source",
    "RepoSummary",
    "summarize_repo",
    "Answer",
    "answer_question",
    "faiss_available",
    "make_vector_store",
]

__version__ = "0.2.0"
