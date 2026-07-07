"""Regression tests for cloned GitHub repositories."""

import pytest

from codeseeker.chunking import is_prose_file
from codeseeker.repo import resolve_source
from codeseeker.index import CodeIndex
from codeseeker.summary import summarize_repo


def test_is_prose_file_skips_issue_templates():
    assert not is_prose_file("README.md")
    assert not is_prose_file("readme.rst")
    assert is_prose_file("CONTRIBUTING.md")
    assert is_prose_file("HISTORY.md")


@pytest.mark.integration  # noqa: PT023
def test_requests_github_repo_explain():
    repo = resolve_source("psf/requests", cache_dir="/tmp/codeseeker-test-repos")
    index = CodeIndex.build(repo.local_path, origin=repo.origin, is_remote=True, prefer_faiss="off")
    files = {c.path for c in index.chunks}

    # GitHub noise should not be indexed.
    assert not any(".github/" in f for f in files)
    assert not any(f.endswith("HISTORY.md") for f in files)
    assert any(f.startswith("src/requests/") for f in files)

    summary = summarize_repo(index, root=repo.local_path, use_llm=False)
    desc = summary.description
    components = [c["symbol"] for c in summary.extra["components"]]

    assert summary.name == "requests"
    assert "HTTP" in summary.extra.get("project_type", "") or "HTTP" in str(summary.extra.get("tech_stack"))
    assert ":class:" not in desc
    assert "**" not in desc or "Requests" in desc  # cleaned markdown bold from README ok
    # Should surface real types, not generic get() methods.
    assert "Session" in components
    assert any(s in components for s in ("PreparedRequest", "Response", "Request"))

    ask = summary.extra["suggested_ask_questions"]
    search = summary.extra["suggested_search_queries"]
    assert all(q.endswith("?") for q in ask)
    assert all(not q.endswith("?") for q in search)
    assert any(k in " ".join(search).lower() for k in ("auth", "cookie", "session", "adapter"))

    cookies = index.search("cookies", top_k=1, mode="hybrid")
    assert "cookies" in cookies[0].chunk.path
