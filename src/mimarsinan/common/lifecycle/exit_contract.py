"""The one process-exit contract: reap the children, then hard-exit. Every path."""

from __future__ import annotations

import os
import signal
import sys
import threading
from typing import Callable, NoReturn

from mimarsinan.common.best_effort import best_effort
from mimarsinan.common.lifecycle.cohort_reaper import spawn_cohort_reaper
from mimarsinan.common.lifecycle.owner import owner_token, stamp_cohort
from mimarsinan.common.lifecycle.process_tree import reap_descendants

# Every catchable way a run is asked to stop: ssh drop and tmux kill-session
# (SIGHUP), Ctrl-C (SIGINT), Ctrl-\ (SIGQUIT), schedulers and wall watchdogs
# (SIGTERM). SIGKILL is uncatchable and is contained by the cohort reaper instead.
TERMINATION_SIGNALS: tuple[signal.Signals, ...] = (
    signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT,
)

_install_lock = threading.Lock()
_installed = False
_exiting = False
_on_terminate: Callable[[int], None] | None = None
_signal_term_grace_s = 2.0


def signal_exit_code(signum: int) -> int:
    """Shell convention: a process killed by signal N exits 128 + N."""
    return 128 + int(signum)


def exit_process(
    code: int,
    *,
    teardown: Callable[[], None] | None = None,
    term_grace_s: float = 5.0,
) -> NoReturn:
    """Reap every descendant, run ``teardown``, then hard-exit with ``code``.

    The reap comes FIRST and the ordering is not the caller's to choose: until it
    runs, every worker still holds a duplicate of this process's stdout/stderr, so
    a launcher waiting on those pipes cannot see EOF and an external SIGKILL in
    that window orphans the whole cohort. ``teardown`` is telemetry-shaped side
    work (draining snapshots, restoring streams); it may fail without changing
    ``code``, and it can never prevent the exit.

    A second entry -- a signal landing while the epilogue runs, say a second
    Ctrl-C -- skips straight to the hard exit rather than restarting the reap.
    """
    global _exiting
    if _exiting:
        os._exit(code)
    _exiting = True
    with best_effort("reap child processes"):
        reap_descendants(term_grace_s=term_grace_s)
    if teardown is not None:
        with best_effort("exit teardown"):
            teardown()
    with best_effort("flush standard streams"):
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(code)


def _handle_termination(signum, _frame) -> NoReturn:
    notice = _on_terminate
    if notice is not None:
        with best_effort(f"exit notice for signal {signum}"):
            notice(int(signum))
    exit_process(signal_exit_code(signum), term_grace_s=_signal_term_grace_s)


def install_exit_contract(
    *,
    on_terminate: Callable[[int], None] | None = None,
    term_grace_s: float = 2.0,
) -> None:
    """Bind this run's whole cohort to its own lifetime, on every termination path.

    Catchable signals route through ``exit_process``. The uncatchable ones (SIGKILL,
    the OOM killer, a hardware fault) cannot run any handler here at all, so the
    same teardown is additionally staked out in a separate process: children are
    marked as they are exec'd, and a reaper sweeps everything carrying the mark once
    this process is gone.

    Call once from ``__main__`` before anything can spawn a child -- an unmarked
    child is one this run can no longer account for. Idempotent, and re-callable to
    refine ``on_terminate`` (a status write, say) once the run's identity is known.
    Signals whose handler this platform refuses are skipped.
    """
    global _installed, _on_terminate, _signal_term_grace_s
    with _install_lock:
        _on_terminate = on_terminate
        _signal_term_grace_s = term_grace_s
        if _installed:
            return
        for signum in TERMINATION_SIGNALS:
            try:
                signal.signal(signum, _handle_termination)
            except (OSError, ValueError):
                continue
        token = owner_token()
        if token is not None:
            stamp_cohort(token)
            spawn_cohort_reaper(token)
        _installed = True
