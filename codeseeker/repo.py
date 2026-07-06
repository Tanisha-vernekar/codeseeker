"""Fetch remote repositories to the local machine before indexing.

``codeseeker`` can index a plain local directory, or it can *pull* a remote
repository (e.g. from GitHub) to a local cache and index that. Supported
source forms:

* an existing local path (returned unchanged)
* a full git URL: ``https://github.com/owner/repo(.git)``, ``git@host:owner/repo``
* a GitHub shorthand: ``owner/repo`` or ``github:owner/repo``

Clones are shallow (``--depth 1``) by default and cached under
``~/.cache/codeseeker/repos`` so repeated indexing is fast.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass

DEFAULT_CACHE_DIR = os.path.expanduser(os.path.join("~", ".cache", "codeseeker", "repos"))

_GIT_URL_RE = re.compile(
    r"^(https?://|git@|ssh://|git://)", re.IGNORECASE
)
_SHORTHAND_RE = re.compile(r"^(?:github:)?([\w.-]+)/([\w.-]+?)(?:\.git)?$")


@dataclass
class RepoSource:
    """Describes where indexable source lives and how it got there."""

    local_path: str
    origin: str  # the original source string
    is_remote: bool
    cloned: bool = False  # True if we just cloned it this run


def is_remote_source(source: str) -> bool:
    """Return ``True`` if ``source`` looks like a remote repo reference."""
    if os.path.exists(source):
        return False
    if _GIT_URL_RE.match(source):
        return True
    # owner/repo shorthand (but not an existing relative path)
    if _SHORTHAND_RE.match(source) and not source.startswith((".", "/", "~")):
        return True
    return False


def normalize_git_url(source: str) -> str:
    """Turn a shorthand like ``owner/repo`` into a full HTTPS git URL."""
    if _GIT_URL_RE.match(source):
        return source
    match = _SHORTHAND_RE.match(source)
    if match:
        owner, repo = match.group(1), match.group(2)
        return f"https://github.com/{owner}/{repo}.git"
    return source


def _cache_path(url: str, cache_dir: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    name = re.sub(r"[^\w.-]+", "_", url.rstrip("/").split("/")[-1]) or "repo"
    name = name[:-4] if name.endswith(".git") else name
    return os.path.join(cache_dir, f"{name}-{digest}")


def _run_git(args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def clone_or_update(
    source: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    depth: int = 1,
    update: bool = False,
    branch: str | None = None,
) -> RepoSource:
    """Clone ``source`` (or refresh an existing clone) and return its location."""
    if not shutil.which("git"):
        raise RuntimeError("git is required to fetch remote repositories but was not found.")

    url = normalize_git_url(source)
    os.makedirs(cache_dir, exist_ok=True)
    dest = _cache_path(url, cache_dir)

    if os.path.isdir(os.path.join(dest, ".git")):
        if update:
            try:
                _run_git(["fetch", "--depth", str(depth), "origin"], cwd=dest)
                _run_git(["reset", "--hard", "origin/HEAD"], cwd=dest)
            except subprocess.CalledProcessError:
                # Fall back to a fresh clone if the update fails.
                shutil.rmtree(dest, ignore_errors=True)
        if os.path.isdir(os.path.join(dest, ".git")):
            return RepoSource(local_path=dest, origin=source, is_remote=True, cloned=False)

    args = ["clone", "--depth", str(depth)]
    if branch:
        args += ["--branch", branch, "--single-branch"]
    args += [url, dest]
    try:
        _run_git(args)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"Failed to clone {url!r}: {detail}") from exc

    return RepoSource(local_path=dest, origin=source, is_remote=True, cloned=True)


def resolve_source(
    source: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    depth: int = 1,
    update: bool = False,
    branch: str | None = None,
) -> RepoSource:
    """Resolve a user-provided ``source`` to a local path, cloning if needed."""
    if not is_remote_source(source):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Path not found: {source}")
        return RepoSource(
            local_path=os.path.abspath(source),
            origin=source,
            is_remote=False,
            cloned=False,
        )
    return clone_or_update(source, cache_dir=cache_dir, depth=depth, update=update, branch=branch)
