"""Process-tree enumeration and reaping so runs exit promptly and orphan-free."""

import os
import signal
import time
from multiprocessing import resource_tracker


def _proc_stat_tail(pid: int) -> list[str] | None:
    """/proc/<pid>/stat fields from field 3 on, or None if the process is gone."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            raw = f.read().decode("ascii", "replace")
    except OSError:
        return None
    # comm may contain spaces/parens; fields resume after the LAST ')'.
    try:
        return raw[raw.rindex(")") + 2:].split()
    except ValueError:
        return None


def _proc_stat_fields(pid: int) -> tuple[int, str] | None:
    """(ppid, state) from /proc/<pid>/stat, or None if the process is gone."""
    tail = _proc_stat_tail(pid)
    if tail is None:
        return None
    try:
        return int(tail[1]), tail[0]
    except (ValueError, IndexError):
        return None


def process_identity(pid: int) -> str | None:
    """``"<pid>:<starttime>"`` — an identity no recycled pid can alias.

    /proc/<pid>/stat field 22 is the process start time in clock ticks since
    boot, so the pair survives pid reuse. ``None`` when the process is gone.
    """
    tail = _proc_stat_tail(pid)
    if tail is None:
        return None
    try:
        return f"{pid}:{int(tail[19])}"
    except (ValueError, IndexError):
        return None


def process_is_alive(identity: str) -> bool:
    """Is the process named by ``identity`` (from ``process_identity``) still running?"""
    pid_text, _, _ = identity.partition(":")
    try:
        pid = int(pid_text)
    except ValueError:
        return False
    return process_identity(pid) == identity


def _child_map() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return children
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        fields = _proc_stat_fields(pid)
        if fields is None:
            continue
        children.setdefault(fields[0], []).append(pid)
    return children


def _is_live(pid: int) -> bool:
    fields = _proc_stat_fields(pid)
    return fields is not None and fields[1] != "Z"


def iter_descendants(root_pid: int) -> list[int]:
    """All live descendant pids of ``root_pid`` (children, grandchildren, ...).

    Zombies are excluded: an exited-but-unwaited child holds no descendants and
    no file descriptors, so it can neither pin a pipe nor be signalled.
    """
    children = _child_map()
    result: list[int] = []
    frontier = list(children.get(root_pid, []))
    while frontier:
        pid = frontier.pop()
        if _is_live(pid):
            result.append(pid)
        frontier.extend(children.get(pid, []))
    return result


def _signal_all(pids: list[int], sig: int) -> list[int]:
    signalled = []
    for pid in pids:
        try:
            os.kill(pid, sig)
            signalled.append(pid)
        except OSError:
            pass
    return signalled


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _is_resource_tracker(pid: int) -> bool:
    """A ``multiprocessing.resource_tracker``: unlinks named semaphores/shm on EOF.

    It sets SIGTERM to SIG_IGN and runs its unlink pass only from the ``finally:``
    of its read loop, so signalling it destroys the cleanup instead of causing it.
    """
    return "multiprocessing.resource_tracker" in _cmdline(pid)


def _await_exit(pids: list[int], grace_s: float) -> list[int]:
    """Poll ``pids`` until all are gone or ``grace_s`` elapses; return the survivors."""
    deadline = time.monotonic() + max(grace_s, 0.0)
    survivors = [pid for pid in pids if _is_live(pid)]
    while survivors and time.monotonic() < deadline:
        time.sleep(0.05)
        survivors = [pid for pid in survivors if _is_live(pid)]
    return survivors


def _close_own_resource_tracker_pipe() -> None:
    """Close this process's write end of the resource-tracker control pipe.

    The tracker exits, and unlinks everything still in its cache, only when EVERY
    write end is closed. This drops ours; the reap has already closed the rest by
    killing the processes that held them.
    """
    tracker = getattr(resource_tracker, "_resource_tracker", None)
    fd = getattr(tracker, "_fd", None)
    if tracker is None or fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass
    tracker._fd = None
    tracker._pid = None


def reap_descendants(
    root_pid: int | None = None,
    *,
    term_grace_s: float = 5.0,
    release_grace_s: float = 2.0,
) -> dict[str, list[int]]:
    """Tear down every descendant of ``root_pid`` (default: this process), leaving
    no orphaned process and no orphaned named OS resource. Never raises.

    Holders (workers, forkservers, anything inheriting our stdio) are SIGTERMed,
    given ``term_grace_s`` to leave, then SIGKILLed. Resource trackers are exempt:
    they cannot die from SIGTERM, and a SIGKILL would destroy the unlink pass.
    They are released instead -- once the holders are gone their control pipes
    reach EOF, so they unlink their caches and exit on their own -- and are
    SIGKILLed only if they outlive ``release_grace_s``.

    Returns {"terminated": [...], "killed": [...], "released": [...]}.
    """
    root = os.getpid() if root_pid is None else root_pid
    trackers = set()
    holders = []
    for pid in iter_descendants(root):
        (trackers.add if _is_resource_tracker(pid) else holders.append)(pid)

    terminated = _signal_all(holders, signal.SIGTERM)
    survivors = _await_exit(terminated, term_grace_s)

    # Re-scan: a TERM'd parent may have re-parented children mid-shutdown.
    rescanned = set(iter_descendants(root))
    trackers |= {pid for pid in rescanned if _is_resource_tracker(pid)}
    stragglers = sorted((set(survivors) | rescanned) - trackers)
    killed = _signal_all([pid for pid in stragglers if _is_live(pid)], signal.SIGKILL)

    if root_pid is None or root_pid == os.getpid():
        _close_own_resource_tracker_pipe()
    released = sorted(pid for pid in trackers if _is_live(pid))
    killed += _signal_all(_await_exit(released, release_grace_s), signal.SIGKILL)
    return {"terminated": terminated, "killed": killed, "released": released}
