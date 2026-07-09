"""Process-tree enumeration and reaping so runs exit promptly and orphan-free."""

import os
import signal
import time


def _proc_stat_fields(pid: int) -> tuple[int, str] | None:
    """(ppid, state) from /proc/<pid>/stat, or None if the process is gone."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            raw = f.read().decode("ascii", "replace")
    except OSError:
        return None
    # comm may contain spaces/parens; fields resume after the LAST ')'.
    try:
        tail = raw[raw.rindex(")") + 2:].split()
        return int(tail[1]), tail[0]
    except (ValueError, IndexError):
        return None


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


def iter_descendants(root_pid: int) -> list[int]:
    """All live descendant pids of ``root_pid`` (children, grandchildren, ...)."""
    children = _child_map()
    result: list[int] = []
    frontier = list(children.get(root_pid, []))
    while frontier:
        pid = frontier.pop()
        result.append(pid)
        frontier.extend(children.get(pid, []))
    return result


def _is_live(pid: int) -> bool:
    fields = _proc_stat_fields(pid)
    return fields is not None and fields[1] != "Z"


def _signal_all(pids: list[int], sig: int) -> list[int]:
    signalled = []
    for pid in pids:
        try:
            os.kill(pid, sig)
            signalled.append(pid)
        except OSError:
            pass
    return signalled


def reap_descendants(
    root_pid: int | None = None, *, term_grace_s: float = 5.0
) -> dict[str, list[int]]:
    """SIGTERM every descendant of ``root_pid`` (default: this process), then
    SIGKILL whatever survives the grace period. Never raises.

    Returns {"terminated": [...], "killed": [...]} for shutdown logging.
    """
    root = os.getpid() if root_pid is None else root_pid
    targets = iter_descendants(root)
    terminated = _signal_all(targets, signal.SIGTERM)

    deadline = time.monotonic() + max(term_grace_s, 0.0)
    survivors = [pid for pid in terminated if _is_live(pid)]
    while survivors and time.monotonic() < deadline:
        time.sleep(0.05)
        survivors = [pid for pid in survivors if _is_live(pid)]

    # A TERM'd parent may have spawned nothing new, but re-scan to catch
    # children that re-parented mid-shutdown.
    stragglers = sorted(set(survivors) | set(iter_descendants(root)))
    killed = _signal_all([pid for pid in stragglers if _is_live(pid)], signal.SIGKILL)
    return {"terminated": terminated, "killed": killed}
