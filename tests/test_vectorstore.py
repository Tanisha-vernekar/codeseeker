import numpy as np
import pytest

from codeseeker.vectorstore import (
    NumpyVectorStore,
    faiss_available,
    make_vector_store,
)


def _normalized(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _random_vectors(n=50, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return _normalized(rng.standard_normal((n, dim)))


def test_numpy_store_ranks_by_cosine():
    vectors = _random_vectors()
    store = NumpyVectorStore().build(vectors)
    query = vectors[7]
    scores, idx = store.search(query, top_k=5)
    assert idx[0] == 7
    assert scores[0] == pytest.approx(1.0, abs=1e-4)
    assert list(scores) == sorted(scores, reverse=True)


def test_numpy_store_empty():
    store = NumpyVectorStore().build(np.zeros((0, 0), dtype=np.float32))
    scores, idx = store.search(np.zeros(4, dtype=np.float32), top_k=3)
    assert scores.size == 0 and idx.size == 0


def test_make_vector_store_off_is_numpy():
    store = make_vector_store(False)
    assert isinstance(store, NumpyVectorStore)


@pytest.mark.skipif(not faiss_available(), reason="faiss not installed")
def test_faiss_matches_numpy_ranking():
    vectors = _random_vectors(n=200, dim=32, seed=1)
    query = _random_vectors(n=1, dim=32, seed=2)[0]

    faiss_store = make_vector_store(True).build(vectors)
    numpy_store = NumpyVectorStore().build(vectors)

    _, faiss_idx = faiss_store.search(query, top_k=10)
    _, numpy_idx = numpy_store.search(query, top_k=10)

    # Exact flat index should match brute force ranking.
    assert list(faiss_idx) == list(numpy_idx)
    assert faiss_store.backend_name == "faiss"
