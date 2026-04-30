"""Subtractive-reset LIF process + float model for Lava.

Must live in its own module: Lava's compiler discovers ``ProcessModel``
implementations by scanning the ``__module__`` of the ``Process`` class
(see ``ProcGroupDiGraphs._find_proc_models``).  Closures and dynamic class
bodies inside a function cannot be imported back by Lava, so this file is
kept at top level and imported lazily by :mod:`lava_loihi_runner` once the
Lava submodule has been verified to load on the current interpreter.

Semantics
---------
``du = 1``, ``dv = 0`` produce pure integrate-and-fire: current equals the
synaptic input for that cycle (no persistence), voltage accumulates across
cycles (no leak).  ``spiking_activation`` fires when ``v > vth`` to match
HCM/nevresim's ``thresholding_mode='<'`` (strict threshold).  ``reset_voltage``
subtracts ``vth`` (subtractive reset) so above-threshold charge carries
into the next cycle — again matching nevresim ``firing_mode='Default'``.

A periodic state reset is applied every ``reset_interval`` timesteps so
many input samples can be processed sequentially by a single Lava graph;
this amortises graph-compile overhead across samples.
"""

from __future__ import annotations

import numpy as np

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.lif.models import AbstractPyLifModelFloat
from lava.proc.lif.process import LIFReset


class SubtractiveLIFReset(LIFReset):
    """LIFReset variant with subtractive-reset semantics.

    Matches nevresim ``firing_mode='Default'`` exactly and the subtractive
    SpikingJelly ``IFNode(v_reset=None)`` used during training.

    ``thresholding_mode`` controls the firing comparator and must mirror
    what ``SpikingHybridCoreFlow`` / nevresim use for the same run:

    * ``"<"``  → strict (``self.v > self.vth``)  — matches HCM ``ops['<']``
      (``torch.lt(thresh, memb)``).
    * ``"<="`` → inclusive (``self.v >= self.vth``) — matches HCM
      ``ops['<=']`` (``torch.le(thresh, memb)``).

    Hard-coding the comparator in the model class previously discarded
    the pipeline config and silently broke parity in any run that used
    inclusive thresholding.
    """

    def __init__(
        self,
        *,
        active_start=None,
        active_length=None,
        sample_start=None,
        thresholding_mode: str = "<",
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


@implements(proc=SubtractiveLIFReset, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PySubtractiveLIFResetModelFloat(AbstractPyLifModelFloat):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.reset_interval = int(proc_params["reset_interval"])
        self.reset_offset = int(proc_params["reset_offset"]) % self.reset_interval
        self.active_start = int(proc_params.get("active_start", self.reset_offset)) % self.reset_interval
        self.active_length = int(proc_params.get("active_length", self.reset_interval))
        self.sample_start = int(proc_params.get("sample_start", self.active_start)) % self.reset_interval
        self._thresholding_mode = str(proc_params.get("thresholding_mode", "<"))
        self._last_spike = np.zeros_like(self.v, dtype=float)

    def spiking_activation(self):
        if self._thresholding_mode == "<=":
            return self.v >= self.vth
        return self.v > self.vth

    def reset_voltage(self, spike_vector: np.ndarray):
        # Subtractive reset — preserve residual charge above threshold.
        self.v[spike_vector] -= self.vth

    def run_spk(self):
        a_in = self.a_in.recv()
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
