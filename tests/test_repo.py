import os

import pytest

from codeseeker.repo import (
    is_remote_source,
    normalize_git_url,
    resolve_source,
)


def test_is_remote_source_detects_urls():
    assert is_remote_source("https://github.com/owner/repo")
    assert is_remote_source("git@github.com:owner/repo.git")
    assert is_remote_source("owner/repo")
    assert is_remote_source("github:owner/repo")


def test_is_remote_source_local_paths(tmp_path):
    assert not is_remote_source(".")
    assert not is_remote_source(str(tmp_path))
    assert not is_remote_source("./relative/path")


def test_normalize_git_url_shorthand():
    assert normalize_git_url("owner/repo") == "https://github.com/owner/repo.git"
    assert normalize_git_url("github:owner/repo") == "https://github.com/owner/repo.git"
    url = "https://github.com/owner/repo.git"
    assert normalize_git_url(url) == url


def test_resolve_source_local(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    src = resolve_source(str(tmp_path))
    assert not src.is_remote
    assert not src.cloned
    assert os.path.abspath(str(tmp_path)) == src.local_path


def test_resolve_source_missing_local():
    with pytest.raises(FileNotFoundError):
        resolve_source("/definitely/not/a/real/path/xyz")
