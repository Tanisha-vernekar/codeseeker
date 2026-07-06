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
