"""Minimal dynamic GPU leasing for parallel jobs on a shared, changing server.

Two job classes (never hard-code GPU ids; the free set shifts as other users come
and go):

  * ``free`` — profiling / wall-clock work. Needs an *exclusively* free GPU:
    ``mem_free/mem_total >= FREE_FRAC`` AND ``util < UTIL_MAX`` AND not leased by
    this pool. Held exclusively for the job's lifetime so the measurement is
    uncontended.
  * ``fit`` — correctness-only work (not profiling). Any GPU whose free memory
    (minus this pool's outstanding ``fit`` reservations) fits ``need_mb``; util is
    IGNORED (a GPU another user pins at 100% is fine — only correctness matters).

Coordination across all processes / git worktrees on the host is via lease files in
a shared dir (default ``/dev/shm/mim_gpu_leases_<uid>``) guarded by ONE flock, so the
snapshot→pick→claim decision is atomic and two jobs never grab the same free GPU.
External users' usage is observed through ``nvidia-smi`` (their memory/util is in the
snapshot), so a lease only prevents OUR jobs from double-booking. Dead leases (the
owning pid is gone) are pruned on every read — no daemon, no heartbeat.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

FREE_FRAC = 0.80   # >=80% of memory idle => "free" (per the profiling discipline)
UTIL_MAX = 5       # <5% util => "free"
DEFAULT_FIT_MB = 8000


def lease_dir() -> str:
    d = os.environ.get("MIM_GPU_LEASE_DIR")
    if not d:
        base = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"
        d = os.path.join(base, f"mim_gpu_leases_{os.getuid()}")
    os.makedirs(d, exist_ok=True)
    return d


@dataclass(frozen=True)
class GpuStat:
    index: int
    mem_total: int   # MiB
    mem_free: int    # MiB
    util: int        # percent

    @property
    def free_frac(self) -> float:
        return self.mem_free / self.mem_total if self.mem_total else 0.0

    def is_free(self, free_frac: float = FREE_FRAC, util_max: int = UTIL_MAX) -> bool:
        return self.free_frac >= free_frac and self.util < util_max


@dataclass(frozen=True)
class Lease:
    gpu: int
    pid: int
    mode: str
    mb: int
    path: str


def parse_nvidia_smi(text: str) -> List[GpuStat]:
    """Parse ``index,memory.total,memory.free,utilization.gpu`` CSV (noheader,nounits)."""
    stats = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[0].lstrip("-").isdigit():
            continue
        idx, total, free, util = (int(float(p)) for p in parts[:4])
        stats.append(GpuStat(idx, total, free, util))
    return stats


def query_nvidia_smi() -> List[GpuStat]:
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=index,memory.total,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    return parse_nvidia_smi(out)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # alive but owned by another user
    return True


def read_leases(directory: Optional[str] = None,
                pid_alive: Callable[[int], bool] = _pid_alive) -> List[Lease]:
    """Live leases in the dir; stale ones (dead owning pid) are removed as a side effect."""
    directory = directory or lease_dir()
    os.makedirs(directory, exist_ok=True)
    leases: List[Lease] = []
    for name in os.listdir(directory):
        if not name.endswith(".lease"):
            continue
        path = os.path.join(directory, name)
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        if not pid_alive(int(d["pid"])):
            try:
                os.unlink(path)
            except OSError:
                pass
            continue
        leases.append(Lease(int(d["gpu"]), int(d["pid"]), str(d["mode"]),
                            int(d["mb"]), path))
    return leases


def choose(mode: str, need_mb: int, stats: Sequence[GpuStat],
           leases: Sequence[Lease],
           free_frac: float = FREE_FRAC, util_max: int = UTIL_MAX,
           max_per_gpu: Optional[int] = None) -> Optional[int]:
    """Pure pick: the GPU index to claim for ``mode``, or None if none is eligible.

    ``free`` -> an exclusively-free, unleased GPU (load-balanced: most idle memory).
    ``fit``  -> a GPU not held by a ``free`` lease whose ``mem_free`` minus this
    pool's ``fit`` reservations fits ``need_mb`` (worst-fit: most remaining headroom).
    ``max_per_gpu`` caps concurrent ``fit`` leases per GPU so memory packing can't
    oversubscribe a card (compute/CPU thrash, false OOM crashes).
    """
    by_gpu_free = {s.index: any(l.gpu == s.index and l.mode == "free" for l in leases)
                   for s in stats}
    fit_count = {s.index: sum(1 for l in leases
                              if l.gpu == s.index and l.mode == "fit")
                 for s in stats}
    fit_reserved = {s.index: sum(l.mb for l in leases
                                 if l.gpu == s.index and l.mode == "fit")
                    for s in stats}
    held = {s.index: by_gpu_free[s.index] or fit_reserved[s.index] > 0 for s in stats}

    candidates = []
    for s in stats:
        if mode == "free":
            if s.is_free(free_frac, util_max) and not held[s.index]:
                candidates.append((s.mem_free, s.index))
        elif mode == "fit":
            if by_gpu_free[s.index]:
                continue
            if max_per_gpu is not None and fit_count[s.index] >= max_per_gpu:
                continue
            remaining = s.mem_free - fit_reserved[s.index]
            if remaining >= need_mb:
                candidates.append((remaining, s.index))
        else:
            raise ValueError(f"unknown mode {mode!r} (expected 'free' or 'fit')")
    if not candidates:
        return None
    return max(candidates)[1]  # most headroom -> spread load / least contention


@contextmanager
def _flock(directory: str):
    import fcntl
    lock_path = os.path.join(directory, ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def acquire(mode: str, need_mb: int = DEFAULT_FIT_MB,
            directory: Optional[str] = None,
            snapshot: Callable[[], List[GpuStat]] = query_nvidia_smi,
            cmd: str = "", max_per_gpu: Optional[int] = None) -> Optional[Lease]:
    """Atomically claim one GPU for ``mode``; returns the Lease or None if none free."""
    directory = directory or lease_dir()
    os.makedirs(directory, exist_ok=True)
    with _flock(directory):
        stats = snapshot()
        leases = read_leases(directory)
        gpu = choose(mode, need_mb, stats, leases, max_per_gpu=max_per_gpu)
        if gpu is None:
            return None
        path = os.path.join(directory, f"{os.getpid()}_{uuid.uuid4().hex}.lease")
        payload = {"gpu": gpu, "pid": os.getpid(), "mode": mode, "mb": int(need_mb),
                   "ts": time.time(), "cmd": cmd[:500]}
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
        return Lease(gpu, os.getpid(), mode, int(need_mb), path)


def release(lease: Optional[Lease]) -> None:
    if lease is None:
        return
    try:
        os.unlink(lease.path)
    except OSError:
        pass


def acquire_blocking(mode: str, need_mb: int = DEFAULT_FIT_MB,
                     timeout: float = 3600.0, poll: float = 4.0,
                     directory: Optional[str] = None,
                     snapshot: Callable[[], List[GpuStat]] = query_nvidia_smi,
                     cmd: str = "",
                     sleep: Callable[[float], None] = time.sleep,
                     clock: Callable[[], float] = time.monotonic) -> Optional[Lease]:
    """Block until a GPU can be claimed for ``mode`` (so a freed GPU is grabbed
    immediately and never left idle while work waits), or None on timeout."""
    deadline = clock() + timeout
    while True:
        lease = acquire(mode, need_mb, directory, snapshot, cmd)
        if lease is not None:
            return lease
        if clock() >= deadline:
            return None
        sleep(poll)
