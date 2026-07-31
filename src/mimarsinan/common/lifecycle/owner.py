"""Die-with-owner: a spawned worker must not outlive the run that owns it."""

from __future__ import annotations

import functools
import os
import sys
import threading
import time
from typing import Callable

from mimarsinan.common.lifecycle.process_tree import process_identity, process_is_alive

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
