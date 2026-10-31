"""CUDA debugging helpers; all entry points are no-ops when CUDA is unavailable or debug mode is off."""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass

from mimarsinan.common.best_effort import best_effort
from mimarsinan.common.env import set_cuda_debug

#: The lifetime reading's name where no CUDA device is visible.
CUDA_UNAVAILABLE = "CUDA_UNAVAILABLE"
LIFETIME_SCOPE = "process lifetime"

# device index -> {"allocated": bytes, "reserved": bytes}: the largest peaks folded in before each reset.
_LIFETIME_PEAKS: dict[int, dict[str, int]] = {}
_LIFETIME_LOCK = threading.Lock()


@dataclass(frozen=True)
class LifetimeCudaPeaks:
    """The process-lifetime CUDA peaks of one device: torch's peak statistics as if no phase had reset them.
    ``absent`` names why there is no reading (``CUDA_UNAVAILABLE``); the byte fields are then ``None``."""

    device: int | None
    max_memory_allocated_bytes: int | None
    max_memory_reserved_bytes: int | None
    absent: str | None = None

    def as_dict(self) -> dict:
        scope = LIFETIME_SCOPE if self.absent is None else f"none ({self.absent})"
        return {"scope": scope, "device": self.device, "absent": self.absent,
                "max_memory_allocated_bytes": self.max_memory_allocated_bytes,
                "max_memory_reserved_bytes": self.max_memory_reserved_bytes}


def _cuda_index(torch, device) -> int:
    """The device index torch's memory statistics use for ``device`` (None: the current device)."""
    if device is None:
        return int(torch.cuda.current_device())
    if isinstance(device, int):
        return device
    d = torch.device(device)
    return int(torch.cuda.current_device()) if d.index is None else int(d.index)


def _fold(torch, index: int) -> dict[str, int]:
    """Fold the running peaks of device ``index`` into its lifetime maxima (under the lock) and return them."""
    running = {"allocated": int(torch.cuda.max_memory_allocated(index)),
               "reserved": int(torch.cuda.max_memory_reserved(index))}
    life = _LIFETIME_PEAKS.setdefault(index, {"allocated": 0, "reserved": 0})
    for key, value in running.items():
        life[key] = max(life[key], value)
    return dict(life)


def reset_cuda_peak_stats(device=None) -> None:
    """Reset torch's peak memory statistics of ``device`` (None: the current device), after folding the running
    peaks into the process-lifetime maxima, so a per-phase reset no longer hides the session's peak. Every reset in
    the toolchain goes through here; a no-op without CUDA."""
    import torch

    if not torch.cuda.is_available():
        return
    index = _cuda_index(torch, device)
    with _LIFETIME_LOCK:
        _fold(torch, index)
        torch.cuda.reset_peak_memory_stats(device)


def lifetime_cuda_peaks(device=None) -> LifetimeCudaPeaks:
    """The process-lifetime peak allocated and reserved bytes of ``device`` (None: the current device): the maxima
    folded in at every reset and the running peak since the last one. Without CUDA, the named empty result."""
    import torch

    if not torch.cuda.is_available():
        return LifetimeCudaPeaks(None, None, None, absent=CUDA_UNAVAILABLE)
    index = _cuda_index(torch, device)
    with _LIFETIME_LOCK:
        life = _fold(torch, index)
    return LifetimeCudaPeaks(index, life["allocated"], life["reserved"])


def enable_cuda_debug() -> None:
    """Set the env vars that force synchronous CUDA kernel launches.

    Must be called before any CUDA context is created, else ``CUDA_LAUNCH_BLOCKING``
    is effectively ignored.
    """
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    set_cuda_debug()


def describe_tensor(t) -> str:
    """One-line summary of a tensor for error reports."""
    import torch

    if not isinstance(t, torch.Tensor):
        return f"<not a tensor: {type(t).__name__}>"
    parts = [
        f"shape={tuple(t.shape)}",
        f"dtype={t.dtype}",
        f"device={t.device}",
    ]
    if t.is_floating_point() and t.numel() > 0:
        finite = torch.isfinite(t)
        if finite.any():
            finite_vals = t[finite]
            parts.append(f"min={finite_vals.min().item():.4g}")
            parts.append(f"max={finite_vals.max().item():.4g}")
        parts.append(f"has_nan={bool(torch.isnan(t).any().item())}")
        parts.append(f"has_inf={bool(torch.isinf(t).any().item())}")
    return ", ".join(parts)


def _rss_mib() -> float:
    """Current resident-set size in MiB, or 0.0 if unavailable."""
    with best_effort("read /proc/self/statm rss"):
        with open("/proc/self/statm") as f:
            rss_pages = int(f.read().split()[1])
        return rss_pages * (os.sysconf("SC_PAGE_SIZE") / (1 << 20))
    return 0.0


@contextlib.contextmanager
def phase_profiler(tag: str, name: str, *, sink=None):
    """Time a block and print peak CPU RSS delta + CUDA peak allocation.

    ``sink`` is an optional callable that receives the formatted line; when
    ``None`` the line is printed. The CUDA peak is the block's own (the reset
    goes through ``reset_cuda_peak_stats``, which keeps the lifetime peak).
    """
    import torch

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        torch.cuda.synchronize()
        reset_cuda_peak_stats()
    t0 = time.perf_counter()
    rss0 = _rss_mib()
    try:
        yield
    finally:
        if cuda_ok:
            with best_effort("cuda synchronize (phase_profiler)"):
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        rss_delta = _rss_mib() - rss0
        if cuda_ok:
            cuda_peak = torch.cuda.max_memory_allocated() / (1 << 20)
            line = (
                f"[{tag}] phase={name} t={elapsed:.2f}s "
                f"rss_delta={rss_delta:+.1f} MiB cuda_peak={cuda_peak:.1f} MiB"
            )
        else:
            line = (
                f"[{tag}] phase={name} t={elapsed:.2f}s "
                f"rss_delta={rss_delta:+.1f} MiB"
            )
        if sink is not None:
            sink(line)
        else:
            print(line)


@contextlib.contextmanager
def cuda_guard(name: str, *, enabled: bool = True):
    """Bracket a block with ``torch.cuda.synchronize()`` so a CUDA assertion raises
    at the boundary (accurate traceback) instead of in a later unrelated kernel.
    """
    import torch

    if not enabled or not torch.cuda.is_available():
        yield
        return

    with best_effort("cuda pre-sync (cuda_guard)"):
        torch.cuda.synchronize()

    try:
        yield
        torch.cuda.synchronize()
    except Exception:
        print(f"[cuda_guard:{name}] exception; CUDA memory summary:", file=sys.stderr)
        with best_effort("cuda memory summary print (cuda_guard)"):
            print(torch.cuda.memory_summary(abbreviated=True), file=sys.stderr)
        traceback.print_exc()
        raise
