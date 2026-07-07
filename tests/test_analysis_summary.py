import textwrap

import pytest

from codeseeker.analysis import analyze_repo
from codeseeker.index import CodeIndex
from codeseeker.summary import (
    suggest_ask_questions,
    suggest_search_queries,
    summarize_repo,
)


def _make_project(tmp_path):
    (tmp_path / "README.md").write_text(
        "# demo\n\n"
        "A tiny demo service that loads configuration and connects to a database "
        "for integration testing purposes.\n"
    )
    (tmp_path / "requirements.txt").write_text("flask>=2.0\npyyaml>=5.0\n")
    (tmp_path / "config.py").write_text(
        textwrap.dedent(
            '''\
            import yaml

            def load_config(path):
                """Read and parse a YAML configuration file from disk."""
                return yaml.safe_load(open(path))
            '''
        )
    )
    (tmp_path / "db.py").write_text(
        textwrap.dedent(
            '''\
            class Database:
                """A thin wrapper around a database connection."""

                def connect(self, url):
                    """Open a new database connection."""
                    return url
            '''
        )
    )
    return tmp_path


def test_analyze_repo_profile(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root))
    profile = analyze_repo(index, str(root), "demo", readme="A demo service.")

    assert profile.num_files >= 2
    assert profile.components
    assert any(c.symbol == "Database" for c in profile.components)
    assert "python" in profile.tech_stack
    assert profile.insights


def test_summarize_repo_heuristic(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root))
    summary = summarize_repo(index, root=str(root), use_llm=False)

    assert summary.num_files >= 2
    assert not summary.llm_used
    assert summary.description
    assert "demo" in summary.description.lower()
    assert "database" in summary.description.lower() or "Database" in summary.description
    assert summary.extra.get("tech_stack")
    assert summary.extra.get("architecture_layers") is not None
    assert any("Database" in s for s in summary.notable_symbols)


def test_summary_json_roundtrip(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root))
    data = summarize_repo(index, root=str(root), use_llm=False).to_dict()
    assert {"root", "num_files", "languages", "description", "tech_stack"}.issubset(data)


def test_summary_has_clean_name_and_components(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root), origin="octo/demo", is_remote=True)
    summary = summarize_repo(index, root=str(root), use_llm=False)
    assert summary.name == "demo"
    assert any("Database" in c["symbol"] for c in summary.extra.get("components", []))


def test_suggest_ask_questions(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root))
    questions = suggest_ask_questions(index)
    assert questions
    assert all(q.endswith("?") for q in questions)
    joined = " ".join(questions).lower()
    assert "database" in joined or "load_config" in joined or "connections" in joined


def test_suggest_search_queries(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root))
    queries = suggest_search_queries(index)
    assert queries
    assert all(not q.endswith("?") for q in queries)
    joined = " ".join(queries).lower()
    assert "load" in joined or "config" in joined or "database" in joined or "connect" in joined
    ask = suggest_ask_questions(index)
    assert set(q.lower() for q in queries) != set(q.lower() for q in ask)


def test_summary_exposes_suggestions_and_profile(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root))
    data = summarize_repo(index, root=str(root), use_llm=False).to_dict()
    assert data["suggested_ask_questions"]
    assert data["suggested_search_queries"]
    assert data["project_type"]
    assert "demo" in data["description"].lower()
