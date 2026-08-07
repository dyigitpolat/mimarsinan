"""Launch a long-lived child: completion is its exit, never the death of its cohort."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Mapping, Sequence

# How long to let a killed session's members leave before declaring them stuck.
_SESSION_REAP_S = 5.0
# How long to wait for the capture threads once the session is dead.
_DRAIN_S = 5.0


@dataclass(frozen=True)
class ChildResult:
    """Outcome of one child run: its own exit, its output, and its cohort's fate."""

    returncode: int
    timed_out: bool
    stdout: str
    stderr: str
    wall_s: float
    session_pid: int
    session_gone: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def stderr_tail(self, lines: int = 20) -> str:
        return "\n".join(self.stderr.splitlines()[-lines:])


def _drain(stream: IO[str] | None, sink: list[str]) -> threading.Thread:
    def _read() -> None:
        if stream is None:
            return
        while True:
            chunk = stream.read(65536)
            if not chunk:
                return
            sink.append(chunk)

    thread = threading.Thread(target=_read, name="child-capture", daemon=True)
    thread.start()
    return thread


def _signal_session(pgid: int, sig: int) -> bool:
    """Signal the whole session; ``False`` once nothing is left in it."""
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _kill_session(pgid: int) -> bool:
    """SIGTERM then SIGKILL the session; ``True`` when it is verifiably empty."""
    if not _signal_session(pgid, signal.SIGTERM):
        return True
    deadline = time.monotonic() + _SESSION_REAP_S
    while time.monotonic() < deadline:
        if not _signal_session(pgid, 0):
            return True
        time.sleep(0.05)
    _signal_session(pgid, signal.SIGKILL)
    deadline = time.monotonic() + _SESSION_REAP_S
    while time.monotonic() < deadline:
        if not _signal_session(pgid, 0):
            return True
        time.sleep(0.05)
    return not _signal_session(pgid, 0)


def run_child(
    argv: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
) -> ChildResult:
    """Run ``argv`` to completion under an optional wall budget, capturing output.

    Completion is ``wait()`` on the direct child. ``subprocess.run(capture_output=
    True)`` instead completes on EOF of both pipes, which arrives only once every
    process holding a duplicate of the write ends has closed it -- and a run's
    forkserver, resource tracker and dataloader workers all hold one. A single
    leaked descendant therefore pins that caller for its entire budget and then
    reports a timeout for a child that finished long before.

    The child leads its own session, so its whole cohort is killed as a unit on
    the timeout path and verified empty on every path.
    """
    out_chunks: list[str] = []
    err_chunks: list[str] = []
    started = time.monotonic()
    proc = subprocess.Popen(
        list(argv),
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    readers = (_drain(proc.stdout, out_chunks), _drain(proc.stderr, err_chunks))

    timed_out = False
    try:
        returncode = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_session(proc.pid)
        returncode = proc.wait()

    # Nothing may outlive the run that produced it, timeout or not: the session
    # sweep is what makes the capture pipes reach EOF and the readers finish.
    session_gone = _kill_session(proc.pid)
    for reader in readers:
        reader.join(timeout=_DRAIN_S)
    for stream in (proc.stdout, proc.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass

    return ChildResult(
        returncode=returncode,
        timed_out=timed_out,
        stdout="".join(out_chunks),
        stderr="".join(err_chunks),
        wall_s=time.monotonic() - started,
        session_pid=proc.pid,
        session_gone=session_gone,
    )
