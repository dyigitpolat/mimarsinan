"""Kernel-starttime process identity: makes pid-based liveness PID-reuse safe.

A recorded pid alone cannot prove a recovered run is still alive — the kernel
recycles pids, so ``os.kill(pid, 0)`` may greet an imposter. The pair
(pid, /proc/<pid>/stat field 22 ``starttime``) is unique for the machine's
uptime; recording it at spawn and re-reading it at probe time turns the bare
signal-0 probe into an identity check. When /proc is unavailable (non-Linux),
callers fall back to the bare probe — documented, not silent.
"""

from __future__ import annotations


def parse_stat_starttime(stat: bytes) -> int | None:
    """Field 22 (``starttime``) of one /proc/<pid>/stat payload, or None.

    The comm field (2) is parenthesised and may itself contain spaces and
    parens, so fields are anchored at the LAST ``)``: the tokens after it start
    at field 3, putting ``starttime`` at token index 19.
    """
    try:
        rest = stat[stat.rindex(b")") + 2:].split()
        return int(rest[19])
    except (ValueError, IndexError):
        return None


def read_proc_starttime(pid: int) -> int | None:
    """Kernel start time (clock ticks) of *pid*; None when /proc cannot serve it."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            return parse_stat_starttime(f.read())
    except OSError:
        return None
