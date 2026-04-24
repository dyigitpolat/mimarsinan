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
cycles (no leak).  ``spiking_activation`` fires when ``v >= vth`` to match
nevresim's ``thresholding_mode='<'`` (strict threshold).  ``reset_voltage``
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
    """


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

    def spiking_activation(self):
        return self.v >= self.vth

    def reset_voltage(self, spike_vector: np.ndarray):
        # Subtractive reset — preserve residual charge above threshold.
        self.v[spike_vector] -= self.vth

    def run_spk(self):
        a_in = self.a_in.recv()
        if (self.time_step % self.reset_interval) == self.reset_offset:
            self.u *= 0
            self.v *= 0
        self.subthr_dynamics(activation_in=a_in)
        s_out = self.spiking_activation()
        self.reset_voltage(spike_vector=s_out)
        self.s_out.send(s_out.astype(float))
