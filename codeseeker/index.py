"""Build, persist and query a semantic index over a codebase."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from codeseeker.chunking import (
    CodeChunk,
    chunk_file,
    iter_source_files,
    read_source,
)
from codeseeker.embeddings import Embedder, build_embedder, load_embedder

INDEX_DIRNAME = ".codeseeker"
_META_FILE = "meta.json"
_CHUNKS_FILE = "chunks.json"
_VECTORS_FILE = "vectors.npy"
_FORMAT_VERSION = 1


@dataclass
class SearchResult:
    """A single ranked search hit."""

    score: float
    chunk: CodeChunk

    def to_dict(self) -> dict:
        return {"score": self.score, "chunk": self.chunk.to_dict()}


class CodeIndex:
    """An in-memory semantic index that can be saved to and loaded from disk."""

    def __init__(
        self,
        chunks: list[CodeChunk],
        vectors: np.ndarray,
        embedder: Embedder,
        root: str = "",
    ) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors must have matching lengths")
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self.root = root

    def __len__(self) -> int:
        return len(self.chunks)

    # -- construction -----------------------------------------------------
    @classmethod
    def build(
        cls,
        root: str,
        extensions: Iterable[str] | None = None,
        exclude_dirs: Iterable[str] | None = None,
        backend: str = "tfidf",
        embedder: Embedder | None = None,
        progress: Callable[[str], None] | None = None,
        **embedder_kwargs,
    ) -> "CodeIndex":
        """Walk ``root``, chunk every source file, and embed the chunks."""
        root = os.path.abspath(root)
        chunks: list[CodeChunk] = []
        n_files = 0
        for path in iter_source_files(root, extensions, exclude_dirs):
            source = read_source(path)
            if source is None:
                continue
            n_files += 1
            rel = os.path.relpath(path, root) if os.path.isdir(root) else path
            for chunk in chunk_file(path, source):
                # Store paths relative to the index root for portability.
                chunks.append(
                    CodeChunk(
                        path=rel,
                        language=chunk.language,
                        kind=chunk.kind,
                        symbol=chunk.symbol,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        text=chunk.text,
                        docstring=chunk.docstring,
                        extra=chunk.extra,
                    )
                )

        if progress:
            progress(f"Indexed {n_files} files, {len(chunks)} chunks")

        if embedder is None:
            embedder = build_embedder(backend, **embedder_kwargs)

        documents = [_chunk_document(chunk) for chunk in chunks]
        if documents:
            embedder.fit(documents)
            vectors = embedder.transform(documents)
        else:
            # Nothing to index; leave the embedder unfitted and store an
            # empty matrix. ``search`` short-circuits on an empty index.
            vectors = np.zeros((0, 0), dtype=np.float32)

        return cls(chunks=chunks, vectors=vectors, embedder=embedder, root=root)

    # -- querying ---------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return the ``top_k`` chunks most semantically similar to ``query``."""
        if not self.chunks or self.vectors.size == 0:
            return []
        query_vec = self.embedder.transform([query])[0]
        # Vectors are L2-normalised, so cosine similarity == dot product.
        scores = self.vectors @ query_vec
        top_k = max(1, min(top_k, len(self.chunks)))
        # argpartition for efficiency, then sort the small top slice.
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        results = []
        for idx in top_idx:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(SearchResult(score=score, chunk=self.chunks[idx]))
        return results

    # -- persistence ------------------------------------------------------
    def save(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        meta = {
            "format_version": _FORMAT_VERSION,
            "root": self.root,
            "num_chunks": len(self.chunks),
            "embedder": self.embedder.to_dict(),
        }
        with open(os.path.join(index_dir, _META_FILE), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        with open(os.path.join(index_dir, _CHUNKS_FILE), "w", encoding="utf-8") as fh:
            json.dump([c.to_dict() for c in self.chunks], fh)
        np.save(os.path.join(index_dir, _VECTORS_FILE), self.vectors)

    @classmethod
    def load(cls, index_dir: str) -> "CodeIndex":
        with open(os.path.join(index_dir, _META_FILE), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        version = meta.get("format_version")
        if version != _FORMAT_VERSION:
            raise ValueError(
                f"Incompatible index format version {version!r}; "
                f"expected {_FORMAT_VERSION}. Re-run 'codeseeker index'."
            )
        with open(os.path.join(index_dir, _CHUNKS_FILE), "r", encoding="utf-8") as fh:
            chunks = [CodeChunk.from_dict(d) for d in json.load(fh)]
        vectors = np.load(os.path.join(index_dir, _VECTORS_FILE), allow_pickle=False)
        embedder = load_embedder(meta["embedder"])
        return cls(
            chunks=chunks,
            vectors=vectors,
            embedder=embedder,
            root=meta.get("root", ""),
        )


def _chunk_document(chunk: CodeChunk) -> str:
    """Build the text representation of a chunk used for embedding.

    Symbol names and docstrings are weighted a little more heavily by
    repeating them, since they usually carry the strongest semantic signal.
    """
    parts = [chunk.text]
    if chunk.symbol:
        parts.append(chunk.symbol)
        parts.append(chunk.symbol)
    if chunk.docstring:
        parts.append(chunk.docstring)
        parts.append(chunk.docstring)
    return "\n".join(parts)


def default_index_dir(root: str) -> str:
    return os.path.join(os.path.abspath(root), INDEX_DIRNAME)
