"""Subtractive-reset LIF process + float model for Lava."""

# Must stay a top-level module because Lava's compiler discovers ProcessModels by scanning
# the Process class's __module__, so closures/dynamic class bodies cannot be imported back.

from __future__ import annotations

from typing import cast

import numpy as np

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.lif.models import AbstractPyLifModelFloat
from lava.proc.lif.process import LIFReset


class SubtractiveLIFReset(LIFReset):
    """LIFReset variant with subtractive-reset semantics, matching nevresim ``firing_mode='Default'``.

    ``thresholding_mode`` picks the firing comparator (``"<"`` strict / ``"<="`` inclusive)
    and must mirror what ``SpikingHybridCoreFlow`` / nevresim use for the same run.
    """

    def __init__(
        self,
        *,
        active_start=None,
        active_length=None,
        sample_start=None,
        thresholding_mode: str = "<=",
        zero_reset: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if active_start is None:
            active_start = self.proc_params["reset_offset"]
        if active_length is None:
            active_length = self.proc_params["reset_interval"]
        if sample_start is None:
            sample_start = active_start
        if thresholding_mode not in ("<", "<="):
            raise ValueError(
                f"thresholding_mode must be '<' or '<='; got {thresholding_mode!r}"
            )
        self.proc_params["active_start"] = int(active_start)
        self.proc_params["active_length"] = int(active_length)
        self.proc_params["sample_start"] = int(sample_start)
        self.proc_params["thresholding_mode"] = str(thresholding_mode)
        self.proc_params["zero_reset"] = bool(zero_reset)


@implements(proc=SubtractiveLIFReset, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PySubtractiveLIFResetModelFloat(AbstractPyLifModelFloat):
    # Lava's builder swaps these LavaPyType descriptors for concrete ports/values at compile time.
    s_out: PyOutPort = cast(PyOutPort, LavaPyType(PyOutPort.VEC_DENSE, float))
    vth: float = cast(float, LavaPyType(float, float))

    def __init__(self, proc_params):
        super().__init__(proc_params)
        assert proc_params is not None, "Lava constructs process models with concrete proc_params"
        self.reset_interval = int(proc_params["reset_interval"])
        self.reset_offset = int(proc_params["reset_offset"]) % self.reset_interval
        self.active_start = int(proc_params.get("active_start", self.reset_offset)) % self.reset_interval
        self.active_length = int(proc_params.get("active_length", self.reset_interval))
        self.sample_start = int(proc_params.get("sample_start", self.active_start)) % self.reset_interval
        self._thresholding_mode = str(proc_params.get("thresholding_mode", "<="))
        self._zero_reset = bool(proc_params.get("zero_reset", False))
        self._last_spike = np.zeros_like(self.v, dtype=float)

    def spiking_activation(self):
        if self._thresholding_mode == "<=":
            return self.v >= self.vth
        return self.v > self.vth

    def reset_voltage(self, spike_vector: np.ndarray):
        if self._zero_reset:
            self.v[spike_vector] = 0.0
        else:
            self.v[spike_vector] -= self.vth

    def run_spk(self):
        a_in = cast(np.ndarray, self.a_in.recv())
        phase = self.time_step % self.reset_interval
        if phase == self.sample_start:
            self.u *= 0
            self.v *= 0
            self._last_spike = np.zeros_like(self.v, dtype=float)
        if phase == self.reset_offset:
            self.u *= 0
            self.v *= 0
            self._last_spike = np.zeros_like(self.v, dtype=float)
        active_delta = (phase - self.active_start) % self.reset_interval
        if active_delta >= self.active_length:
            self.s_out.send(self._last_spike.copy())
            return
        self.subthr_dynamics(activation_in=a_in)
        s_out = self.spiking_activation()
        self.reset_voltage(spike_vector=s_out)
        self._last_spike = s_out.astype(float)
        self.s_out.send(self._last_spike.copy())
