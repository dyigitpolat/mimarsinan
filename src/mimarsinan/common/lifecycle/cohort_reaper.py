"""The out-of-process half of the exit contract: teardown that survives SIGKILL."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import IO

from mimarsinan.common.lifecycle.owner import (
    DEFAULT_POLL_INTERVAL_S,
    cohort_members,
)
from mimarsinan.common.lifecycle.process_tree import process_is_alive, reap_processes

MODULE = "mimarsinan.common.lifecycle.cohort_reaper"
# <src>, the directory the mimarsinan package lives in, so the reaper can import
# itself without depending on the owner's sys.path edits or working directory.
PACKAGE_ROOT = str(Path(__file__).resolve().parents[3])

TERM_GRACE_S = 5.0
# A holder wedged in an uninterruptible CUDA ioctl absorbs even SIGKILL for a
# while, and the resource tracker reaches EOF only once EVERY holder is gone;
# killing it sooner would destroy the unlink pass that frees the run's semaphores.
RELEASE_GRACE_S = 60.0


def _report(message: str) -> None:
    sys.stderr.write(f"[cohort-reaper] {message}\n")
    sys.stderr.flush()


def reap_when_owner_dies(
    token: str,
    *,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
) -> dict[str, list[int]]:
    """Block until ``token``'s owner is gone, then tear its whole cohort down.

    Runs in its own process precisely so an uncatchable kill of the owner cannot
    take the teardown with it. Membership comes from the inherited cohort mark,
    never from a tree walk: by the time this returns from the wait, the kernel has
    already re-parented everything the owner spawned.
    """
    while process_is_alive(token):
        time.sleep(poll_interval_s)
    outcome = reap_processes(
        lambda: cohort_members(token, exclude=(os.getpid(),)),
        term_grace_s=TERM_GRACE_S,
        release_grace_s=RELEASE_GRACE_S,
    )
    if any(outcome.values()):
        _report(
            f"owner {token} died without running its exit contract; "
            f"terminated={outcome['terminated']} killed={outcome['killed']} "
            f"released={outcome['released']}"
        )
    return outcome


def spawn_cohort_reaper(
    token: str,
    *,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    log_path: str | os.PathLike[str] | None = None,
) -> int:
    """Start the reaper bound to ``token`` and return its pid. Raises if it cannot.

    Its stdio is detached (``log_path`` redirects the diagnostics it would
    otherwise discard): a reaper holding a duplicate of the run's stdout would pin
    every launcher that waits on pipe EOF. Its own session keeps a terminal's
    Ctrl-C from disarming the guard while the run it guards is still shutting down;
    it remains a child, so the owner's ordinary reap still collects it.
    """
    if log_path is None:
        return _spawn(token, poll_interval_s, subprocess.DEVNULL)
    with open(log_path, "ab", buffering=0) as log:
        return _spawn(token, poll_interval_s, log)


def _spawn(token: str, poll_interval_s: float, stderr: "int | IO[bytes]") -> int:
    return subprocess.Popen(
        [sys.executable, "-m", MODULE, token, repr(float(poll_interval_s))],
        cwd=PACKAGE_ROOT,
        env=_reaper_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=stderr,
        close_fds=True,
        start_new_session=True,
    ).pid


def _reaper_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        PACKAGE_ROOT + os.pathsep + existing if existing else PACKAGE_ROOT
    )
    return env


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit(f"usage: python -m {MODULE} <owner-token> [poll-interval-s]")
    poll_interval_s = float(argv[2]) if len(argv) > 2 else DEFAULT_POLL_INTERVAL_S
    reap_when_owner_dies(argv[1], poll_interval_s=poll_interval_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
