import numpy as np
import pytest

from codeseeker.embeddings import (
    TfidfEmbedder,
    build_embedder,
    load_embedder,
    tokenize,
)


def test_tokenize_splits_identifiers():
    # "by" is a stopword and is filtered out.
    assert tokenize("getUserById") == ["get", "user", "id"]
    assert tokenize("get_user_by_id") == ["get", "user", "id"]
    assert tokenize("HTTPServerError") == ["http", "server", "error"]


def test_tokenize_filters_stopwords_and_short():
    tokens = tokenize("the return of a value x")
    assert "the" not in tokens
    assert "return" not in tokens
    assert "value" in tokens


def test_tfidf_transform_is_normalised():
    docs = [
        "def load_config(path): read yaml configuration file",
        "def connect_database(url): open database connection",
        "class HttpClient: send request retry backoff",
    ]
    emb = TfidfEmbedder().fit(docs)
    matrix = emb.transform(docs)
    assert matrix.shape[0] == 3
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_tfidf_semantic_ranking():
    docs = [
        "def load_config(path): read and parse configuration file",
        "def connect_database(url): open a database connection",
        "def compute_average(numbers): return the mean value",
    ]
    emb = TfidfEmbedder().fit(docs)
    doc_vecs = emb.transform(docs)
    query = emb.transform(["parse configuration file"])[0]
    scores = doc_vecs @ query
    assert int(np.argmax(scores)) == 0


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        TfidfEmbedder().transform(["hello"])


def test_tfidf_round_trip_serialization():
    docs = ["alpha beta gamma", "beta gamma delta"]
    emb = TfidfEmbedder().fit(docs)
    restored = load_embedder(emb.to_dict())
    assert isinstance(restored, TfidfEmbedder)
    original = emb.transform(docs)
    again = restored.transform(docs)
    assert np.allclose(original, again)


def test_build_embedder_unknown_backend():
    with pytest.raises(ValueError):
        build_embedder("does-not-exist")


def test_unseen_query_tokens_produce_zero_vector():
    emb = TfidfEmbedder().fit(["alpha beta"])
    vec = emb.transform(["completely different words"])[0]
    assert np.allclose(vec, 0.0)
