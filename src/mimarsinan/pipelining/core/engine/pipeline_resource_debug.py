"""Process resource snapshots for pipeline step debugging."""

from __future__ import annotations

import multiprocessing
import os
import sys

_RESOURCE_DEBUG = os.environ.get("MIMARSINAN_RESOURCE_DEBUG") == "1"


def log_resource_snapshot(tag: str) -> None:
    """One-line stderr snapshot of process resources. No-op unless MIMARSINAN_RESOURCE_DEBUG=1."""
    if not _RESOURCE_DEBUG:
        return
    try:
        pid = os.getpid()
        try:
            fd_count = len(os.listdir(f"/proc/{pid}/fd"))
        except OSError:
            fd_count = -1
        try:
            shm = os.listdir("/dev/shm")
            shm_count = len(shm)
            sem_count = sum(1 for n in shm if n.startswith("sem."))
        except OSError:
            shm_count = -1
            sem_count = -1
        rss_kb = -1
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        break
        except OSError:
            pass
        children = len(multiprocessing.active_children())
        print(
            f"[resource] {tag} pid={pid} rss_kb={rss_kb} "
            f"fd={fd_count} shm={shm_count} sem={sem_count} children={children}",
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        pass
