"""Unit tests for the campaign engineering self-defense guards.

Hermetic: every git-based guard runs against a throwaway repo built under
``tmp_path`` (``git init``) so the real repo's stash/HEAD are never touched.
"""
from __future__ import annotations

import os
import subprocess

import pytest

from guards import (
    AlreadyRunningError,
    StaleBaseError,
    StashTamperError,
    assert_base_current,
    assert_stash_intact,
    singleton_lock,
    snapshot_stash_list,
)


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A hermetic git repo with one commit on ``main``."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    (root / "f.txt").write_text("base\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return root


# -- (1) stale-base guard --

def test_assert_base_current_passes_on_main_head(repo):
    assert_base_current(str(repo), main_ref="main")  # HEAD == main


def test_assert_base_current_passes_on_descendant_of_main(repo):
    # Branch off main, add a commit: HEAD is a descendant of main -> ok.
    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "g.txt").write_text("more\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feature work")
    assert_base_current(str(repo), main_ref="main")


def test_assert_base_current_raises_when_main_moved_ahead(repo):
    # Pin HEAD to a branch, then advance main past it -> HEAD is stale.
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "stale")
    _git(repo, "checkout", "-q", "main")
    (repo / "h.txt").write_text("newer\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main advances")
    _git(repo, "checkout", "-q", "stale")
    assert _git(repo, "rev-parse", "HEAD") == base
    with pytest.raises(StaleBaseError):
        assert_base_current(str(repo), main_ref="main")


# -- (2) stash-tamper guard --

def _make_stash(root, marker):
    (root / "f.txt").write_text(marker)
    _git(root, "stash", "push", "-q", "-m", marker)


def test_snapshot_stash_list_returns_tuple(repo):
    snap = snapshot_stash_list(str(repo))
    assert isinstance(snap, tuple)
    assert snap == ()  # no stashes yet


def test_assert_stash_intact_passes_when_unchanged(repo):
    _make_stash(repo, "wip-a\n")
    before = snapshot_stash_list(str(repo))
    # No tampering between snapshot and check.
    assert_stash_intact(str(repo), before)


def test_assert_stash_intact_passes_when_stash_grew(repo):
    _make_stash(repo, "wip-a\n")
    before = snapshot_stash_list(str(repo))
    _make_stash(repo, "wip-b\n")  # a new stash is fine; nothing was lost.
    assert_stash_intact(str(repo), before)


def test_assert_stash_intact_raises_when_stash_shrank(repo):
    _make_stash(repo, "wip-a\n")
    before = snapshot_stash_list(str(repo))
    _git(repo, "stash", "drop", "-q")  # the stash-pop trap: an entry vanished.
    with pytest.raises(StashTamperError):
        assert_stash_intact(str(repo), before)


def test_assert_stash_intact_raises_when_stash_changed(repo):
    _make_stash(repo, "wip-a\n")
    before = snapshot_stash_list(str(repo))
    _git(repo, "stash", "drop", "-q")
    _make_stash(repo, "wip-different\n")  # same length, different content.
    with pytest.raises(StashTamperError):
        assert_stash_intact(str(repo), before)


# -- (3) singleton lock --

def test_singleton_lock_acquires_and_releases(tmp_path):
    lockdir = tmp_path / "locks"
    with singleton_lock("scheduler", lockdir=str(lockdir)):
        pass  # acquired
    # Released on exit -> re-acquire must succeed.
    with singleton_lock("scheduler", lockdir=str(lockdir)):
        pass


def test_singleton_lock_double_acquire_raises(tmp_path):
    lockdir = tmp_path / "locks"
    with singleton_lock("scheduler", lockdir=str(lockdir)):
        with pytest.raises(AlreadyRunningError):
            with singleton_lock("scheduler", lockdir=str(lockdir)):
                pass


def test_singleton_lock_distinct_names_coexist(tmp_path):
    lockdir = tmp_path / "locks"
    with singleton_lock("scheduler", lockdir=str(lockdir)):
        with singleton_lock("director", lockdir=str(lockdir)):
            pass  # different lock names do not collide.


def test_singleton_lock_writes_pid(tmp_path):
    lockdir = tmp_path / "locks"
    with singleton_lock("scheduler", lockdir=str(lockdir)):
        pidfile = lockdir / "scheduler.pid"
        assert pidfile.exists()
        assert pidfile.read_text().strip() == str(os.getpid())
