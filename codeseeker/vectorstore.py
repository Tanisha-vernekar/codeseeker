"""Vector similarity backends used to power fast semantic search.

Two backends are provided:

* :class:`NumpyVectorStore` — a dependency-free, exact brute-force search that
  works everywhere. Perfectly fast for small/medium repositories.
* :class:`FaissVectorStore` — uses `FAISS <https://faiss.ai>`_ for exact
  (``IndexFlatIP``) or approximate (``IndexIVFFlat``) nearest-neighbour search,
  keeping query latency low even across very large repositories.

All vectors are expected to be **L2-normalised**, so an inner-product search is
equivalent to cosine similarity. The index vectors themselves are always
persisted separately (as ``vectors.npy``); these stores are (re)built from that
matrix, which keeps the on-disk index portable regardless of whether FAISS is
installed.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def faiss_available() -> bool:
    """Return ``True`` if the optional FAISS dependency can be imported."""
    try:
        import faiss  # noqa: F401
    except Exception:  # pragma: no cover - depends on environment
        return False
    return True


class NumpyVectorStore:
    """Exact brute-force cosine similarity search using NumPy."""

    backend_name = "numpy"

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> "NumpyVectorStore":
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        return self

    @property
    def size(self) -> int:
        return 0 if self._vectors is None else self._vectors.shape[0]

    def search(self, query: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._vectors is None or self._vectors.size == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        query = np.ascontiguousarray(query, dtype=np.float32).reshape(-1)
        scores = self._vectors @ query
        top_k = max(1, min(top_k, scores.shape[0]))
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return scores[idx].astype(np.float32), idx.astype(np.int64)


class FaissVectorStore:
    """FAISS-backed similarity search (exact flat or IVF approximate)."""

    backend_name = "faiss"

    def __init__(self, ivf_threshold: int = 50_000, nlist: int = 256, nprobe: int = 16) -> None:
        # Above ``ivf_threshold`` vectors we switch to an approximate IVF index
        # to keep latency well under 100ms at scale.
        self.ivf_threshold = ivf_threshold
        self.nlist = nlist
        self.nprobe = nprobe
        self._index = None
        self._size = 0

    def build(self, vectors: np.ndarray) -> "FaissVectorStore":
        import faiss

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n, dim = vectors.shape if vectors.ndim == 2 else (0, 0)
        self._size = n
        if n == 0:
            self._index = None
            return self

        if n >= self.ivf_threshold:
            nlist = min(self.nlist, max(1, n // 39))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(vectors)
            index.add(vectors)
            index.nprobe = self.nprobe
        else:
            index = faiss.IndexFlatIP(dim)
            index.add(vectors)
        self._index = index
        return self

    @property
    def size(self) -> int:
        return self._size

    def search(self, query: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None or self._size == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        query = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        top_k = max(1, min(top_k, self._size))
        scores, idx = self._index.search(query, top_k)
        return scores[0].astype(np.float32), idx[0].astype(np.int64)


def make_vector_store(prefer_faiss: bool | str = "auto"):
    """Construct the best available vector store.

    ``prefer_faiss`` may be ``True``/``False`` or the string ``"auto"`` (use
    FAISS when it is installed, otherwise fall back to NumPy).
    """
    if prefer_faiss == "auto":
        prefer_faiss = faiss_available()
    if prefer_faiss:
        if not faiss_available():
            raise RuntimeError(
                "FAISS was requested but is not installed. "
                "Install it with: pip install codeseeker[faiss]"
            )
        return FaissVectorStore()
    return NumpyVectorStore()
