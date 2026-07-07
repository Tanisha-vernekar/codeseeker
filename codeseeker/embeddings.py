"""Embedding backends for turning code text into vectors.

The default :class:`TfidfEmbedder` is implemented in pure NumPy so that
``codeseeker`` works completely offline with no model downloads. It tokenises
code in a way that is friendly to semantic queries: identifiers such as
``getUserById`` or ``get_user_by_id`` are split into their sub-words
(``get``, ``user``, ``by``, ``id``) so a natural-language query like
"fetch user by identifier" can match them.

Additional backends (e.g. :class:`SentenceTransformerEmbedder`) are optional
and only imported on demand, so the heavy dependency is never required.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Protocol, runtime_checkable

import numpy as np

# Split on any non-alphanumeric boundary first ...
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
# ... then break camelCase / PascalCase / digit boundaries into sub-words.
_CAMEL_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)

# Very common tokens that add noise rather than signal.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for",
        "on", "with", "as", "by", "self", "this", "return", "def", "class",
        "import", "from", "if", "else", "elif", "while", "for", "true",
        "false", "none", "null", "var", "let", "const", "function",
    }
)


def tokenize(text: str, min_len: int = 2) -> list[str]:
    """Tokenise source text or a query into normalised sub-word tokens."""
    tokens: list[str] = []
    for raw in _WORD_RE.findall(text):
        for part in _CAMEL_RE.findall(raw):
            token = part.lower()
            if len(token) < min_len:
                continue
            if token in _STOPWORDS:
                continue
            tokens.append(token)
    return tokens


@runtime_checkable
class Embedder(Protocol):
    """Protocol implemented by every embedding backend."""

    name: str

    def fit(self, texts: Iterable[str]) -> "Embedder":
        ...

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        ...

    def to_dict(self) -> dict:
        ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so cosine similarity is a simple dot product."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class TfidfEmbedder:
    """Offline TF-IDF embedder with identifier-aware tokenisation."""

    name = "tfidf"

    def __init__(
        self,
        vocabulary: dict[str, int] | None = None,
        idf: np.ndarray | None = None,
        max_features: int = 20000,
        min_len: int = 2,
    ) -> None:
        self.vocabulary = vocabulary or {}
        self.idf = idf
        self.max_features = max_features
        self.min_len = min_len

    def _counts(self, tokens: Iterable[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return counts

    def fit(self, texts: Iterable[str]) -> "TfidfEmbedder":
        texts = list(texts)
        doc_freq: dict[str, int] = {}
        tokenised: list[dict[str, int]] = []
        for text in texts:
            counts = self._counts(tokenize(text, self.min_len))
            tokenised.append(counts)
            for token in counts:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # Keep the most informative / frequent tokens up to ``max_features``.
        ranked = sorted(doc_freq.items(), key=lambda kv: (-kv[1], kv[0]))
        ranked = ranked[: self.max_features]

        n_docs = max(1, len(texts))
        self.vocabulary = {token: i for i, (token, _) in enumerate(ranked)}
        idf = np.zeros(len(self.vocabulary), dtype=np.float32)
        for token, index in self.vocabulary.items():
            df = doc_freq[token]
            # Smoothed idf (matches sklearn's default formulation).
            idf[index] = math.log((1 + n_docs) / (1 + df)) + 1.0
        self.idf = idf
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        if self.idf is None or not self.vocabulary:
            raise RuntimeError("TfidfEmbedder must be fitted before transform().")
        texts = list(texts)
        matrix = np.zeros((len(texts), len(self.vocabulary)), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = self._counts(tokenize(text, self.min_len))
            if not counts:
                continue
            max_count = max(counts.values())
            for token, count in counts.items():
                index = self.vocabulary.get(token)
                if index is None:
                    continue
                # Sub-linear term frequency, scaled by idf.
                tf = 0.5 + 0.5 * (count / max_count)
                matrix[row, index] = tf * self.idf[index]
        return _normalize(matrix)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "max_features": self.max_features,
            "min_len": self.min_len,
            "vocabulary": self.vocabulary,
            "idf": None if self.idf is None else self.idf.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TfidfEmbedder":
        idf = data.get("idf")
        return cls(
            vocabulary={str(k): int(v) for k, v in (data.get("vocabulary") or {}).items()},
            idf=None if idf is None else np.asarray(idf, dtype=np.float32),
            max_features=int(data.get("max_features", 20000)),
            min_len=int(data.get("min_len", 2)),
        )


class SentenceTransformerEmbedder:
    """Optional dense embedder backed by ``sentence-transformers``.

    Only imported when actually used, keeping it a soft dependency.
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional path
                raise RuntimeError(
                    "The 'sentence-transformers' package is required for this "
                    "backend. Install it with: pip install codeseeker[transformers]"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: Iterable[str]) -> "SentenceTransformerEmbedder":
        self._ensure_model()
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        model = self._ensure_model()
        vectors = model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        return _normalize(np.asarray(vectors, dtype=np.float32))

    def to_dict(self) -> dict:
        return {"name": self.name, "model_name": self.model_name}

    @classmethod
    def from_dict(cls, data: dict) -> "SentenceTransformerEmbedder":
        return cls(model_name=data.get("model_name", "all-MiniLM-L6-v2"))


def sentence_transformers_available() -> bool:
    """Return ``True`` if the optional ``sentence-transformers`` package imports."""
    try:
        import sentence_transformers  # noqa: F401
    except Exception:  # pragma: no cover - depends on environment
        return False
    return True


def build_embedder(backend: str = "auto", **kwargs) -> Embedder:
    """Factory that constructs an embedder for the requested ``backend``.

    ``"auto"`` uses the neural ``sentence-transformers`` backend when it is
    installed (best quality), and otherwise the always-available offline
    TF-IDF backend. If the neural model can't be loaded at index time, the
    index build gracefully falls back to TF-IDF.
    """
    backend = (backend or "auto").lower()
    if backend == "auto":
        backend = "sentence-transformers" if sentence_transformers_available() else "tfidf"
    if backend in ("tfidf", "tf-idf", "default"):
        return TfidfEmbedder(
            max_features=int(kwargs.get("max_features", 20000)),
            min_len=int(kwargs.get("min_len", 2)),
        )
    if backend in ("sentence-transformers", "st", "transformers", "neural"):
        return SentenceTransformerEmbedder(
            model_name=kwargs.get("model_name", "all-MiniLM-L6-v2")
        )
    raise ValueError(f"Unknown embedding backend: {backend!r}")


def load_embedder(data: dict) -> Embedder:
    """Reconstruct an embedder previously serialised with ``to_dict``."""
    name = data.get("name", "tfidf")
    if name == TfidfEmbedder.name:
        return TfidfEmbedder.from_dict(data)
    if name == SentenceTransformerEmbedder.name:
        return SentenceTransformerEmbedder.from_dict(data)
    raise ValueError(f"Cannot load unknown embedder: {name!r}")
