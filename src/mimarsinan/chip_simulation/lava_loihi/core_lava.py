"""Single-core Lava LIF execution."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, NamedTuple

import numpy as np
import torch.multiprocessing as torch_mp

from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.lava_loihi.timing import _LAVA_DTYPE

_SUBTRACTIVE_LIF_CLS = None
_LAVA_ACTOR_START_METHOD = "fork"


def _make_set_start_method_idempotent() -> None:
    """No-op ``set_start_method`` when a multiprocessing context is already set."""
    def _patch(module) -> None:
        real_set = module.set_start_method
        if getattr(real_set, "_mimarsinan_lava_safe", False):
            return

        def _safe_set(method, force=False):
            current = module.get_start_method(allow_none=True)
            if force or current is None:
                real_set(method, force=force)

        setattr(_safe_set, "_mimarsinan_lava_safe", True)
        module.set_start_method = _safe_set

    _patch(mp)
    _patch(torch_mp)


def _subtractive_lif_cls():
    """Lazy-import and cache SubtractiveLIFReset."""
    global _SUBTRACTIVE_LIF_CLS
    if _SUBTRACTIVE_LIF_CLS is None:
        _make_set_start_method_idempotent()
        from mimarsinan.chip_simulation.subtractive_lif import SubtractiveLIFReset
        _SUBTRACTIVE_LIF_CLS = SubtractiveLIFReset
    return _SUBTRACTIVE_LIF_CLS


def _require_fork_actors() -> None:
    """Refuse by name a process whose default start method is not ``fork``.

    Lava starts its actors from the default multiprocessing context and runs only under ``fork``: under ``spawn`` its
    SharedMemoryManager is never bound (``'SharedMemoryManager' object has no attribute 'shutdown'``). The wave pool's
    spawn workers set ``fork`` for their own process (``core_worker``), never for the caller.
    """
    established = mp.get_start_method(allow_none=True)
    if established not in (None, _LAVA_ACTOR_START_METHOD):
        raise RuntimeError(
            f"Lava executor refused: this process's default multiprocessing start method is {established!r}; "
            f"Lava's runtime starts its actors from the default context and runs only under "
            f"{_LAVA_ACTOR_START_METHOD!r}. Run the Lava executor in a process whose default start method is "
            f"{_LAVA_ACTOR_START_METHOD!r} (the wave pool's spawn workers set it for their own process)."
        )


class _LavaExecutorApi(NamedTuple):
    run_steps: Any
    sim_cfg: Any
    dense: Any
    source: Any
    sink: Any
    subtractive_lif: Any


def _lava_executor_api() -> _LavaExecutorApi:
    """The Lava classes the executor builds a core graph from, imported behind the start-method guard.

    lava's message infrastructure calls ``multiprocessing.set_start_method('fork')`` on its first import, which raises
    ``context has already been set`` wherever another caller established the context (an xdist worker, a pipeline
    after a pool), so the guard (``_subtractive_lif_cls``) precedes every lava import of the executor.
    """
    subtractive_lif = _subtractive_lif_cls()
    _require_fork_actors()
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi2SimCfg
    from lava.proc.dense.process import Dense
    from lava.proc.io.sink import RingBuffer as Sink
    from lava.proc.io.source import RingBuffer as Source
    return _LavaExecutorApi(RunSteps, Loihi2SimCfg, Dense, Source, Sink, subtractive_lif)


def _release_unstarted_runtime(process: Any) -> None:
    """Kill the actors of a Lava runtime that raised before it started.

    Lava's ``stop()`` joins every actor, but only a started runtime sends them STOP: after a failed start (a thread
    that cannot start under a full task cap, a port that does not open) the actors wait for a command forever and the
    join never returns, so the error that stopped the run would turn into a hang. A started runtime keeps Lava's stop.
    """
    runtime = getattr(process, "runtime", None)
    if runtime is None or getattr(runtime, "_is_started", False):
        return
    infrastructure = getattr(runtime, "_messaging_infrastructure", None)
    for actor in getattr(infrastructure, "actors", None) or ():
        if actor.is_alive():
            actor.kill()


class LavaCoreMixin:
    """Host contract: attributes below are provided by ``LavaLoihiRunner.__init__``."""

    T: int
    thresholding_mode: str
    _behavior: NeuralBehaviorConfig

    def _run_core_lava(
        self,
        *,
        weights: np.ndarray,
        threshold: float,
        hardware_bias: np.ndarray | None,
        input_spikes: np.ndarray,
    ) -> np.ndarray:
        """Run one hard core's Dense + SubtractiveLIFReset on Lava."""
        lava = _lava_executor_api()

        n_out, n_in = weights.shape
        total_cycles = input_spikes.shape[1]
        assert total_cycles % self.T == 0
        N = total_cycles // self.T
        T = self.T

        PAD_HEAD = 2
        TAIL = 3
        PIPELINE_DELAY = 1

        pad_head_block = np.zeros((n_in, PAD_HEAD), dtype=np.float32)
        warmup = input_spikes[:, :T]
        tail_block = np.zeros((n_in, TAIL), dtype=np.float32)
        data = np.concatenate([pad_head_block, warmup, input_spikes, tail_block], axis=1).astype(_LAVA_DTYPE)
        total_steps = data.shape[1]

        reset_offset = (PAD_HEAD + T + PIPELINE_DELAY + 1) % T

        if hardware_bias is None:
            bias_mant = np.zeros((n_out,), dtype=_LAVA_DTYPE)
        else:
            bias_mant = np.asarray(hardware_bias, dtype=_LAVA_DTYPE).reshape(-1)

        src = lava.source(data=data)
        dense = lava.dense(weights=weights.astype(_LAVA_DTYPE))
        lif = lava.subtractive_lif(
            shape=(n_out,),
            du=1,
            dv=0,
            vth=float(threshold),
            bias_mant=bias_mant,
            reset_interval=T,
            reset_offset=reset_offset,
            thresholding_mode=self.thresholding_mode,
            zero_reset=self._behavior.lava_zero_reset(),
        )
        sink = lava.sink(shape=(n_out,), buffer=total_steps)

        src.s_out.connect(dense.s_in)
        dense.a_out.connect(lif.a_in)
        lif.s_out.connect(sink.a_in)

        completed = False
        try:
            lif.run(
                condition=lava.run_steps(num_steps=total_steps),
                run_cfg=lava.sim_cfg(select_tag="floating_pt"),
            )
            raw = sink.data.get()
            completed = True
        finally:
            if not completed:
                _release_unstarted_runtime(lif)
            lif.stop()

        start = PAD_HEAD + T + PIPELINE_DELAY
        return np.asarray(raw[:, start : start + N * T], dtype=_LAVA_DTYPE)
