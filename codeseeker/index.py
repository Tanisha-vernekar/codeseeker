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
from codeseeker.embeddings import tokenize
from codeseeker.vectorstore import make_vector_store

INDEX_DIRNAME = ".codeseeker"
_META_FILE = "meta.json"
_CHUNKS_FILE = "chunks.json"
_VECTORS_FILE = "vectors.npy"
_FORMAT_VERSION = 2


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
        origin: str = "",
        is_remote: bool = False,
        prefer_faiss: bool | str = "auto",
    ) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors must have matching lengths")
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self.root = root
        self.origin = origin
        self.is_remote = is_remote
        self.prefer_faiss = prefer_faiss
        self._store = None

    def __len__(self) -> int:
        return len(self.chunks)

    def _get_store(self):
        """Lazily build (and cache) the vector-search backend."""
        if self._store is None:
            store = make_vector_store(self.prefer_faiss)
            store.build(self.vectors)
            self._store = store
        return self._store

    @property
    def backend_name(self) -> str:
        if self.vectors.size == 0:
            return "none"
        return self._get_store().backend_name

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
        origin: str = "",
        is_remote: bool = False,
        prefer_faiss: bool | str = "auto",
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
            try:
                embedder.fit(documents)
                vectors = embedder.transform(documents)
            except Exception as exc:  # noqa: BLE001
                # A heavy backend (e.g. neural model download) may be
                # unavailable; fall back to the always-available offline engine
                # so indexing never hard-fails.
                from codeseeker.embeddings import TfidfEmbedder

                if isinstance(embedder, TfidfEmbedder):
                    raise
                if progress:
                    progress(f"'{getattr(embedder, 'name', 'backend')}' unavailable ({exc}); using offline tfidf")
                embedder = TfidfEmbedder()
                embedder.fit(documents)
                vectors = embedder.transform(documents)
        else:
            # Nothing to index; leave the embedder unfitted and store an
            # empty matrix. ``search`` short-circuits on an empty index.
            vectors = np.zeros((0, 0), dtype=np.float32)

        return cls(
            chunks=chunks,
            vectors=vectors,
            embedder=embedder,
            root=root,
            origin=origin or root,
            is_remote=is_remote,
            prefer_faiss=prefer_faiss,
        )

    # -- querying ---------------------------------------------------------
    def _filter_indices(
        self,
        languages: Iterable[str] | None,
        kinds: Iterable[str] | None,
        path_contains: str | None,
    ) -> np.ndarray | None:
        """Return the row indices allowed by the filters, or ``None`` for all."""
        if not languages and not kinds and not path_contains:
            return None
        lang_set = {l.lower() for l in languages} if languages else None
        kind_set = {k.lower() for k in kinds} if kinds else None
        needle = path_contains.lower() if path_contains else None
        allowed = []
        for i, chunk in enumerate(self.chunks):
            if lang_set and chunk.language.lower() not in lang_set:
                continue
            if kind_set and chunk.kind.lower() not in kind_set:
                continue
            if needle and needle not in chunk.path.lower():
                continue
            allowed.append(i)
        return np.asarray(allowed, dtype=np.int64)

    # Non-code chunks are legitimate but usually less relevant as answers, so
    # they get a mild prior below 1.0 to keep real code at the top.
    _KIND_PRIOR = {
        "function": 1.0,
        "method": 1.0,
        "class": 1.0,
        "cell": 0.95,
        "module": 0.8,
        "block": 0.75,
        "note": 0.7,
    }

    def _ensure_lexical_cache(self) -> None:
        """Precompute per-chunk token sets and kind priors (cached)."""
        if getattr(self, "_lex_cache", None) is not None:
            return
        sym_tokens: list[set] = []
        path_tokens: list[set] = []
        doc_tokens: list[set] = []
        body_tokens: list[set] = []
        kind_prior = np.ones(len(self.chunks), dtype=np.float32)
        for i, chunk in enumerate(self.chunks):
            sym_tokens.append(set(tokenize(chunk.symbol)))
            path_tokens.append(set(tokenize(chunk.path)))
            doc_tokens.append(set(tokenize(chunk.docstring)))
            body_tokens.append(set(tokenize(chunk.text)))
            kind_prior[i] = self._KIND_PRIOR.get(chunk.kind, 0.85)
        self._lex_cache = {
            "symbol": sym_tokens,
            "path": path_tokens,
            "doc": doc_tokens,
            "body": body_tokens,
            "kind_prior": kind_prior,
        }

    def _lexical_scores(self, query: str) -> np.ndarray:
        """Lexical relevance that strongly rewards symbol/path matches.

        Matching a query token in a symbol name (e.g. ``cookies`` -> the
        ``cookies`` module / ``Cookie`` classes) or file path is far more
        indicative than a match buried in the body, so those are weighted
        heavily. Scores are normalised to roughly [0, 1].
        """
        q = set(tokenize(query))
        n = len(self.chunks)
        if not q or n == 0:
            return np.zeros(n, dtype=np.float32)
        self._ensure_lexical_cache()
        cache = self._lex_cache
        nq = float(len(q))
        scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            s = 0.0
            # Symbol/path use prefix-aware matching so 'authentication' matches
            # 'auth', 'configuration' matches 'config', etc.
            s += 3.0 * _prefix_overlap(q, cache["symbol"][i]) / nq
            s += 2.0 * _prefix_overlap(q, cache["path"][i]) / nq
            s += 1.5 * len(q & cache["doc"][i]) / nq
            s += 0.5 * len(q & cache["body"][i]) / nq
            scores[i] = s
        peak = float(scores.max())
        if peak > 0:
            scores /= peak
        return scores

    def search(
        self,
        query: str,
        top_k: int = 5,
        languages: Iterable[str] | None = None,
        kinds: Iterable[str] | None = None,
        path_contains: str | None = None,
        mode: str = "hybrid",
        semantic_weight: float = 0.8,
    ) -> list[SearchResult]:
        """Return the ``top_k`` chunks most semantically similar to ``query``.

        Optional ``languages``, ``kinds`` and ``path_contains`` filters restrict
        the candidate set before ranking.
        """
        if not self.chunks or self.vectors.size == 0:
            return []
        mode = mode.lower()
        if mode not in {"semantic", "hybrid"}:
            raise ValueError(f"Unknown search mode {mode!r}; expected 'semantic' or 'hybrid'.")
        semantic_weight = float(max(0.0, min(1.0, semantic_weight)))
        query_vec = self.embedder.transform([query])[0]
        allowed = self._filter_indices(languages, kinds, path_contains)

        if mode == "semantic" and allowed is None:
            # Pure semantic ranking via the (possibly FAISS-backed) store.
            scores, idx = self._get_store().search(query_vec, top_k)
            pairs = list(zip(idx.tolist(), scores.tolist()))
        else:
            sem_scores = self.vectors @ np.ascontiguousarray(query_vec, dtype=np.float32)
            if mode == "hybrid":
                lex_scores = self._lexical_scores(query)
                combined = (semantic_weight * sem_scores) + ((1.0 - semantic_weight) * lex_scores)
                # Favour real code over module/doc/note chunks.
                self._ensure_lexical_cache()
                combined = combined * self._lex_cache["kind_prior"]
            else:
                combined = sem_scores

            if allowed is None:
                candidates = np.arange(sem_scores.shape[0])
            else:
                if allowed.size == 0:
                    return []
                candidates = allowed
            cand_scores = combined[candidates]
            k = max(1, min(top_k, candidates.shape[0]))
            order = np.argpartition(-cand_scores, k - 1)[:k]
            order = order[np.argsort(-cand_scores[order])]
            pairs = [(int(candidates[o]), float(cand_scores[o])) for o in order]

        results = []
        for i, score in pairs:
            if i < 0 or score <= 0:
                continue
            results.append(SearchResult(score=float(score), chunk=self.chunks[i]))
        return results

    # -- persistence ------------------------------------------------------
    def save(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        meta = {
            "format_version": _FORMAT_VERSION,
            "root": self.root,
            "origin": self.origin,
            "is_remote": self.is_remote,
            "num_chunks": len(self.chunks),
            "dim": int(self.vectors.shape[1]) if self.vectors.ndim == 2 and self.vectors.size else 0,
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
            origin=meta.get("origin", ""),
            is_remote=bool(meta.get("is_remote", False)),
        )


def _prefix_overlap(query_tokens: set, chunk_tokens: set, min_prefix: int = 4) -> int:
    """Count query tokens with an exact or shared-prefix match in ``chunk_tokens``.

    A shared prefix of at least ``min_prefix`` characters counts as a match,
    which bridges common morphological gaps in code vocabulary (e.g. the query
    word ``authentication`` matching the identifier token ``auth``).
    """
    if not query_tokens or not chunk_tokens:
        return 0
    count = 0
    for q in query_tokens:
        if q in chunk_tokens:
            count += 1
            continue
        for t in chunk_tokens:
            k = min(len(q), len(t))
            if k >= min_prefix and q[:k] == t[:k]:
                count += 1
                break
    return count


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
