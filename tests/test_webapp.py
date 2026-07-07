import textwrap

import pytest

pytest.importorskip("flask")

from codeseeker import webapp
from codeseeker.webapp import STATE, create_app


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    # Reset shared server state between tests.
    STATE.index = None
    STATE.origin = ""
    STATE.local_path = ""
    with app.test_client() as c:
        yield c


def _make_project(tmp_path):
    (tmp_path / "README.md").write_text("# demo\n\nA demo service that loads config and connects to a database.\n")
    (tmp_path / "config.py").write_text(
        textwrap.dedent(
            '''\
            def load_config(path):
                """Read and parse a YAML configuration file."""
                return path
            '''
        )
    )
    (tmp_path / "db.py").write_text(
        textwrap.dedent(
            '''\
            def connect_database(url):
                """Open a database connection."""
                return url
            '''
        )
    )
    return tmp_path


def test_home_serves_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"codeseeker" in res.data
    assert b"Load a GitHub repo" in res.data


def test_status_before_index(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    assert res.get_json() == {"loaded": False}


def test_search_requires_index(client):
    res = client.post("/api/search", json={"query": "anything"})
    assert res.status_code == 400
    assert "No index" in res.get_json()["error"]


def test_index_missing_source(client):
    res = client.post("/api/index", json={})
    assert res.status_code == 400


def test_full_flow(client, tmp_path):
    root = _make_project(tmp_path)

    res = client.post("/api/index", json={"source": str(root), "faiss": "off"})
    assert res.status_code == 200
    stats = res.get_json()["stats"]
    assert stats["num_chunks"] > 0
    assert stats["search_backend"] == "numpy"

    res = client.get("/api/status")
    assert res.get_json()["loaded"] is True

    res = client.post("/api/search", json={"query": "parse configuration file", "top_k": 3})
    assert res.status_code == 200
    results = res.get_json()["results"]
    assert results
    assert results[0]["chunk"]["symbol"] == "load_config"

    res = client.post("/api/search", json={"query": "configuration", "kind": "function"})
    assert all(r["chunk"]["kind"] == "function" for r in res.get_json()["results"])

    res = client.post("/api/search", json={"query": "connect_database", "mode": "hybrid"})
    assert res.status_code == 200
    assert res.get_json()["results"][0]["chunk"]["symbol"] == "connect_database"

    res = client.post("/api/explain", json={})
    assert res.status_code == 200
    assert res.get_json()["description"]

    res = client.post("/api/ask", json={"question": "how to open a database connection?"})
    assert res.status_code == 200
    assert res.get_json()["sources"]

    res = client.get("/api/stats")
    assert res.status_code == 200
    assert "python" in dict(res.get_json()["languages"])

    res = client.get("/api/map")
    assert res.status_code == 200
    payload = res.get_json()
    assert payload["files"]
    assert any("config.py" in f["path"] for f in payload["files"])
