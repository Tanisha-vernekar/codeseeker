import textwrap

import numpy as np
import pytest

from codeseeker.index import CodeIndex


def _make_project(tmp_path):
    (tmp_path / "config.py").write_text(
        textwrap.dedent(
            '''\
            def load_config(path):
                """Read and parse a YAML configuration file from disk."""
                with open(path) as fh:
                    return fh.read()
            '''
        )
    )
    (tmp_path / "db.py").write_text(
        textwrap.dedent(
            '''\
            def connect_database(url):
                """Open a new database connection and return the session."""
                return url
            '''
        )
    )
    (tmp_path / "http.py").write_text(
        textwrap.dedent(
            '''\
            def fetch_with_retry(url, retries=3):
                """Send an HTTP GET request, retrying with exponential backoff."""
                return url
            '''
        )
    )
    return tmp_path


def test_build_and_search(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root))
    assert len(index) >= 3

    results = index.search("parse configuration file", top_k=3)
    assert results
    assert results[0].chunk.symbol == "load_config"
    assert results[0].score > 0


def test_search_scores_descending(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root))
    results = index.search("database connection", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_relative_paths_stored(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root))
    for chunk in index.chunks:
        assert not chunk.path.startswith(str(root))


def test_save_and_load_round_trip(tmp_path):
    root = _make_project(tmp_path)
    index_dir = tmp_path / ".codeseeker"
    index = CodeIndex.build(str(root))
    index.save(str(index_dir))

    loaded = CodeIndex.load(str(index_dir))
    assert len(loaded) == len(index)

    q = "retry http request with backoff"
    r1 = index.search(q, top_k=3)
    r2 = loaded.search(q, top_k=3)
    assert [r.chunk.location for r in r1] == [r.chunk.location for r in r2]
    assert np.allclose([r.score for r in r1], [r.score for r in r2])


def test_load_rejects_bad_version(tmp_path):
    import json
    import os

    root = _make_project(tmp_path)
    index_dir = tmp_path / ".codeseeker"
    CodeIndex.build(str(root)).save(str(index_dir))

    meta_path = os.path.join(str(index_dir), "meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    meta["format_version"] = 999
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)

    with pytest.raises(ValueError):
        CodeIndex.load(str(index_dir))


def test_empty_project_yields_no_results(tmp_path):
    (tmp_path / "notes.txt").write_text("not indexed")
    index = CodeIndex.build(str(tmp_path))
    assert len(index) == 0
    assert index.search("anything") == []


def test_search_kind_filter(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root))
    results = index.search("configuration", top_k=5, kinds=["function"])
    assert results
    assert all(r.chunk.kind == "function" for r in results)


def test_search_path_filter(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root))
    results = index.search("connection", top_k=5, path_contains="db.py")
    assert results
    assert all("db.py" in r.chunk.path for r in results)


def test_search_language_filter_excludes(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root))
    # No JavaScript in the project, so filtering to it yields nothing.
    assert index.search("connection", languages=["javascript"]) == []


def test_backend_name_reports(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root), prefer_faiss=False)
    assert index.backend_name == "numpy"


def test_metadata_persisted(tmp_path):
    root = _make_project(tmp_path)
    index_dir = tmp_path / ".codeseeker"
    index = CodeIndex.build(str(root), origin="owner/repo", is_remote=True)
    index.save(str(index_dir))
    loaded = CodeIndex.load(str(index_dir))
    assert loaded.origin == "owner/repo"
    assert loaded.is_remote is True


def test_symbol_and_path_match_beats_body_keyword(tmp_path):
    # A decoy module mentions the word "handled" in prose, while the real
    # target lives in cookies.py. Symbol/path weighting should win.
    (tmp_path / "compat.py").write_text(
        '"""This module previously handled compatibility between versions."""\n'
        "def _shim():\n    return 1\n"
    )
    (tmp_path / "cookies.py").write_text(
        'def merge_cookies(jar, other):\n'
        '    """Add cookies to the cookiejar and return a merged jar."""\n'
        "    return jar\n"
    )
    index = CodeIndex.build(str(tmp_path))
    results = index.search("how are cookies handled?", top_k=3)
    assert results
    assert "cookies.py" in results[0].chunk.path


def test_hybrid_mode_boosts_exact_identifier_match(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root))

    # Query contains the exact symbol token.
    sem = index.search("connect_database", top_k=3, mode="semantic")
    hyb = index.search("connect_database", top_k=3, mode="hybrid")

    assert sem and hyb
    # Hybrid should keep the intended function as top hit.
    assert sem[0].chunk.symbol == "connect_database"
    assert hyb[0].chunk.symbol == "connect_database"
