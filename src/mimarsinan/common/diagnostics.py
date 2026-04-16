"""CUDA debugging helpers.

Enabled on-demand via ``--debug`` on ``run.py`` or ``cuda_debug: true`` in a
deployment config. All entry points are no-ops when CUDA is unavailable or
debug mode is off, so importing this module has no cost on the hot path.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
import traceback


def enable_cuda_debug() -> None:
    """Set the env vars that force synchronous CUDA kernel launches.

    Must be called before any CUDA context is created: once a kernel has
    launched, ``CUDA_LAUNCH_BLOCKING`` is effectively ignored. ``run.py``
    calls this before importing ``mimarsinan``.
    """
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["MIMARSINAN_CUDA_DEBUG"] = "1"


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
    try:
        with open("/proc/self/statm") as f:
            rss_pages = int(f.read().split()[1])
        return rss_pages * (os.sysconf("SC_PAGE_SIZE") / (1 << 20))
    except Exception:
        return 0.0


@contextlib.contextmanager
def phase_profiler(tag: str, name: str, *, sink=None):
    """Time a block and print peak CPU RSS delta + CUDA peak allocation.

    Intended for instrumenting large pipeline steps so the dominant sub-phase
    is self-identifying. ``tag`` is a prefix shown in the log line (e.g. the
    step name); ``name`` is the sub-phase label. ``sink`` is an optional
    callable that receives the fully-formatted line; when ``None`` the line
    is printed.
    """
    import torch

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    rss0 = _rss_mib()
    try:
        yield
    finally:
        if cuda_ok:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
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
    """Bracket a block of work with ``torch.cuda.synchronize()`` calls.

    When a CUDA assertion trips inside ``name``, the post-sync will raise at
    the boundary rather than later in an unrelated kernel, so tracebacks
    actually point at the offending step. On exception, prints ``name`` and
    a short memory summary before re-raising.
    """
    import torch

    if not enabled or not torch.cuda.is_available():
        yield
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    try:
        yield
        torch.cuda.synchronize()
    except Exception:
        print(f"[cuda_guard:{name}] exception; CUDA memory summary:", file=sys.stderr)
        try:
            print(torch.cuda.memory_summary(abbreviated=True), file=sys.stderr)
        except Exception:
            pass
        traceback.print_exc()
        raise
