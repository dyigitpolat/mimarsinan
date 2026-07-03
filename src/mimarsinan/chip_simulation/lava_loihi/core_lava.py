"""Single-core Lava LIF execution."""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import torch.multiprocessing as torch_mp

from mimarsinan.chip_simulation.lava_loihi.timing import _LAVA_DTYPE

_SUBTRACTIVE_LIF_CLS = None


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

        _safe_set._mimarsinan_lava_safe = True
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


class LavaCoreMixin:
    def _run_core_lava(
        self,
        *,
        weights: np.ndarray,
        threshold: float,
        hardware_bias: np.ndarray | None,
        input_spikes: np.ndarray,
    ) -> np.ndarray:
        """Run one hard core's Dense + SubtractiveLIFReset on Lava."""
        from lava.magma.core.run_conditions import RunSteps
        from lava.magma.core.run_configs import Loihi2SimCfg
        from lava.proc.dense.process import Dense
        from lava.proc.io.sink import RingBuffer as Sink
        from lava.proc.io.source import RingBuffer as Source

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

        SubLIF = _subtractive_lif_cls()
        src = Source(data=data)
        dense = Dense(weights=weights.astype(_LAVA_DTYPE))
        lif = SubLIF(
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
        sink = Sink(shape=(n_out,), buffer=total_steps)

        src.s_out.connect(dense.s_in)
        dense.a_out.connect(lif.a_in)
        lif.s_out.connect(sink.a_in)

        try:
            lif.run(
                condition=RunSteps(num_steps=total_steps),
                run_cfg=Loihi2SimCfg(select_tag="floating_pt"),
            )
            raw = sink.data.get()
        finally:
            lif.stop()

        start = PAD_HEAD + T + PIPELINE_DELAY
        return np.asarray(raw[:, start : start + N * T], dtype=_LAVA_DTYPE)
