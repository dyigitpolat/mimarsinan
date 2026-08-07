"""Die-with-owner: nothing a run spawns may outlive the run that owns it."""

from __future__ import annotations

import functools
import os
import sys
import threading
import time
from typing import Callable, Collection

from mimarsinan.common.env import COHORT_TOKEN_VAR, set_cohort_token
from mimarsinan.common.lifecycle.process_tree import (
    iter_descendants,
    process_environ,
    process_identity,
    process_is_alive,
)

# 128 + SIGKILL: the worker was terminated because its owner already was.
ORPHAN_EXIT_CODE = 137

DEFAULT_POLL_INTERVAL_S = 2.0


def owner_token(pid: int | None = None) -> str | None:
    """Identity of the process that owns work spawned from here (default: self).

    Pass this token to every child. ``os.getppid`` cannot substitute for it: under
    a forkserver start method a worker's parent is the forkserver, whose pid never
    changes when the run dies.
    """
    return process_identity(os.getpid() if pid is None else pid)


def _self_terminate(token: str) -> None:
    sys.stderr.write(
        f"[lifecycle] owner {token} is gone; terminating orphaned worker "
        f"{os.getpid()}\n"
    )
    sys.stderr.flush()
    os._exit(ORPHAN_EXIT_CODE)


def watch_owner(
    token: str,
    *,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    on_owner_death: Callable[[str], None] = _self_terminate,
) -> threading.Thread:
    """Start a daemon thread that ends this process when ``token``'s owner dies."""

    def _poll() -> None:
        while process_is_alive(token):
            time.sleep(poll_interval_s)
        on_owner_death(token)

    thread = threading.Thread(
        target=_poll, name=f"owner-watch[{token}]", daemon=True,
    )
    thread.start()
    return thread


def install_owner_watch(
    token: str | None,
    _worker_id: int | None = None,
    *,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
) -> None:
    """Worker-pool initializer: bind this process's lifetime to ``token``.

    Shaped for ``DataLoader(worker_init_fn=...)`` and pool ``initializer=``, which
    both call it with a worker index. A ``None`` token means the owner was already
    unidentifiable, so nothing is installed.
    """
    if token is None:
        return
    watch_owner(token, poll_interval_s=poll_interval_s)


def owner_bound_initializer(
    token: str | None,
) -> "functools.partial[None] | None":
    """A picklable ``worker_init_fn``/``initializer`` bound to ``token``, or None."""
    if token is None:
        return None
    return functools.partial(install_owner_watch, token)


def stamp_cohort(token: str) -> None:
    """Enrol every process this one goes on to exec into ``token``'s cohort.

    The in-process watch above only binds workers whose framework runs an
    initializer. Forkservers, resource trackers and plain ``Popen`` children never
    do -- but they all inherit an environment, so the mark reaches them all.
    """
    set_cohort_token(token)


def cohort_members(token: str, *, exclude: Collection[int] = ()) -> list[int]:
    """Every live process carrying ``token``'s mark, plus their live descendants.

    Membership is inherited rather than observed, so it survives what a watcher
    cannot: the owner's death (the kernel re-parents its children), a member's own
    ``setsid``/``setpgrp``, and processes born after the last poll. The descendant
    expansion covers the one gap left -- a child handed a scrubbed environment
    still hangs below a marked process.
    """
    mark = f"{COHORT_TOKEN_VAR}={token}".encode()
    skip = set(exclude)
    members: list[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return members
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in skip:
            continue
        if mark in process_environ(pid):
            skip.add(pid)
            members.append(pid)
    for pid in list(members):
        for descendant in iter_descendants(pid):
            if descendant not in skip:
                skip.add(descendant)
                members.append(descendant)
    return members
