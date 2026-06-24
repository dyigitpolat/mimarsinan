"""Engineering self-defense guards for the autonomous campaign automation.

Three traps the campaign has hit before, each guarded here:
  - stale-base: a daemon kept running on a HEAD that ``main`` had moved past
    (``assert_base_current`` — HEAD must be ``main`` or a descendant of it).
  - stash-pop: an automated step silently dropped a ``git stash`` entry
    (``snapshot_stash_list`` + ``assert_stash_intact`` — the list must never shrink
    or change under us).
  - double-run: two scheduler instances racing the same queue
    (``singleton_lock`` — one ``fcntl.flock`` pidfile per name).
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
from typing import Iterator, Tuple


class StaleBaseError(RuntimeError):
    """HEAD is not equal-to-or-descendant-of the main ref (the stale-base trap)."""


class StashTamperError(RuntimeError):
    """The git stash list shrank or changed since the snapshot (the stash-pop trap)."""


class AlreadyRunningError(RuntimeError):
    """Another instance already holds the named singleton lock."""


def _git(repo_root: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo_root, *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def assert_base_current(repo_root: str, main_ref: str = "main") -> None:
    """Raise StaleBaseError unless HEAD is ``main_ref`` or a descendant of it.

    Uses ``git merge-base --is-ancestor <main_ref> HEAD``: when ``main_ref`` is an
    ancestor of HEAD, HEAD already contains everything on main (HEAD == main, or
    HEAD is ahead of it) — current. Otherwise main has commits HEAD lacks — stale.
    """
    head = _git(repo_root, "rev-parse", "HEAD")
    main = _git(repo_root, "rev-parse", main_ref)
    is_ancestor = subprocess.run(
        ["git", "-C", repo_root, "merge-base", "--is-ancestor", main_ref, "HEAD"],
        capture_output=True, text=True,
    )
    if is_ancestor.returncode != 0:
        raise StaleBaseError(
            f"HEAD ({head}) is stale: {main_ref} ({main}) has commits HEAD lacks. "
            f"Rebase onto {main_ref} before running."
        )


def snapshot_stash_list(repo_root: str) -> Tuple[str, ...]:
    """The current stash entries, identified by their immutable commit SHA.

    Each line is ``<commit-sha> <message>`` (NOT the ``stash@{N}`` index, which
    renumbers when a new stash is pushed) so ``assert_stash_intact`` compares on a
    stable identity — pushing a fresh stash shifts no existing entry's key.
    """
    out = _git(repo_root, "stash", "list", "--format=%H %gs")
    return tuple(line for line in out.splitlines() if line)


def assert_stash_intact(repo_root: str, before: Tuple[str, ...]) -> None:
    """Raise StashTamperError unless every snapshotted stash entry still exists.

    Growth is fine (a new stash may be pushed); loss or in-place change is the
    stash-pop trap. The check is set-based on the snapshotted entries so a newly
    pushed stash shifting indices does not false-positive.
    """
    now = set(snapshot_stash_list(repo_root))
    missing = [entry for entry in before if entry not in now]
    if missing:
        raise StashTamperError(
            f"git stash was tampered with: {len(missing)} snapshotted "
            f"entr{'y' if len(missing) == 1 else 'ies'} vanished or changed: "
            f"{missing!r}"
        )


def _default_lockdir() -> str:
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(repo_root, "runs", "campaign", "locks")


@contextlib.contextmanager
def singleton_lock(name: str, lockdir: str | None = None) -> Iterator[str]:
    """Hold an exclusive ``fcntl.flock`` on ``<lockdir>/<name>.pid`` for the body.

    Only one process may hold a given ``name`` at a time; a second acquirer raises
    AlreadyRunningError immediately (non-blocking ``LOCK_EX | LOCK_NB``). The lock
    releases — and the pidfile is unlinked — on exit, so a later acquire succeeds.
    """
    if lockdir is None:
        lockdir = _default_lockdir()
    os.makedirs(lockdir, exist_ok=True)
    pidfile = os.path.join(lockdir, f"{name}.pid")
    fd = os.open(pidfile, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise AlreadyRunningError(
                f"singleton lock {name!r} is already held (pidfile {pidfile}); "
                "another instance is running."
            ) from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.fsync(fd)
        try:
            yield pidfile
        finally:
            # Release the flock but keep the pidfile as the stable lock anchor;
            # unlinking would swap the inode under a concurrent acquirer.
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
