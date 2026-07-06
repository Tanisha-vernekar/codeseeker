"""codeseeker: semantic code search over local codebases.

Public API:
    - :class:`~codeseeker.chunking.CodeChunk`
    - :func:`~codeseeker.chunking.chunk_file`
    - :class:`~codeseeker.index.CodeIndex`
    - :class:`~codeseeker.index.SearchResult`
"""

from codeseeker.chunking import CodeChunk, chunk_file, iter_source_files
from codeseeker.embeddings import (
    Embedder,
    TfidfEmbedder,
    build_embedder,
    load_embedder,
)
from codeseeker.index import CodeIndex, SearchResult

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
]

__version__ = "0.1.0"
