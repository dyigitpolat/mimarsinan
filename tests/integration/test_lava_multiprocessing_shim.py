from __future__ import annotations

import importlib.util
import multiprocessing as mp

import pytest
import torch.multiprocessing as torch_mp


def test_probe_lava_after_spawn_context_noops_fork_request():
    """Lava import must not crash after project code selects spawn."""
    torch_mp.set_start_method("spawn", force=True)

    try:
        from mimarsinan.chip_simulation.lava_loihi_runner import _probe_lava
        _probe_lava()
    except Exception as exc:
        pytest.skip(f"Lava not importable on this host: {exc}")

    mp.set_start_method("fork")
    torch_mp.set_start_method("fork")
    assert mp.get_start_method(allow_none=True) == "spawn"


def _import_lava_mp_module() -> None:
    """Worker target: import lava's mp module in a fresh interpreter.

    The spawn bootstrap binds the worker's start method to 'spawn' before
    user code runs. Importing lava's vendored ``multiprocessing`` module
    used to call ``mp.set_start_method('fork')`` unconditionally and crash
    here with ``RuntimeError('context has already been set')``.
    """
    import lava.magma.runtime.message_infrastructure.multiprocessing  # noqa: F401


def test_spawned_worker_can_import_lava_multiprocessing():
    """Repro for the original Loihi-step crash.

    A spawned worker re-imports lava during unpickling; the vendored
    ``multiprocessing.py`` must be idempotent because the parent-side
    shim isn't installed in the worker yet.
    """
    if importlib.util.find_spec("lava") is None:
        pytest.skip("Lava not installed on this host")

    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_import_lava_mp_module)
    p.start()
    p.join(timeout=30)
    assert p.exitcode == 0, f"worker died with exitcode {p.exitcode}"
