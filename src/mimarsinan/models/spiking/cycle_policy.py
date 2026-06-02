"""Per-cycle neuron policy: the swappable fire/reset behavior of the pipelined
(cascaded) spiking executor.

LIF and cascaded TTFS share the same per-cycle executor skeleton (cycle loop,
latency-window gating, span fill, ``count/simulation_length`` decode); they differ
only in the per-core integrate-and-fire step:

- **LIF** integrates inputs, fires whenever the membrane crosses threshold, and
  resets (subtractive / zero) — can fire every cycle.
- **cascaded TTFS** integrates latched inputs and fires **at most once**, latching
  its output high for the rest of the window; no reset (greedy: later input ignored).

Mirrors nevresim's ``fire_policy`` decomposition on the Python side so the executor
stays schedule-agnostic.
"""

from __future__ import annotations

from typing import Dict

import torch

from mimarsinan.models.spiking.lif_core_step import lif_core_contribute_and_fire
from mimarsinan.models.spiking.ttfs_cycle_step import ttfs_cycle_contribute_and_fire

NeuronState = Dict[str, torch.Tensor]


class LIFCyclePolicy:
    """Multi-spike integrate-and-fire with Default/Novena reset."""

    # Mirror nevresim's ``FirePolicy`` static flags so the shared executor stays
    # schedule-agnostic. LIF: cores active only inside their latency window;
    # always-on (bias) axons inject one spike per input cycle.
    latency_gated = True
    always_on_every_cycle = False
    single_spike_io = False

    def __init__(self, firing_mode: str):
        self.firing_mode = str(firing_mode)

    def make_state(self, batch_size: int, n_neurons: int, device, dtype) -> NeuronState:
        return {"memb": torch.zeros(batch_size, n_neurons, device=device, dtype=dtype)}

    def step(self, state, weight, inp, threshold, *, hw_bias, thresholding_mode,
             output_dtype=None) -> torch.Tensor:
        return lif_core_contribute_and_fire(
            state["memb"], weight, inp, threshold,
            hw_bias=hw_bias, thresholding_mode=thresholding_mode,
            firing_mode=self.firing_mode, output_dtype=output_dtype,
        )


class TTFSGreedyCyclePolicy:
    """Cascaded TTFS: single-spike, fire-once integrate-and-fire (no reset).

    Hardware-faithful: each neuron emits exactly one spike (the firing-cycle
    transition); the integration is a **ramp** reconstructed from single-spike
    arrivals via a persistent ``ramp_current``. Mirrors nevresim's single-spike
    ``SpikingCompute`` neuron and the SANA-FE single-spike soma + ramping dendrite.
    """

    # Latency-gated like LIF: each core is active only in its own window
    # [lat, lat+T) — so its bias/ramp start at its reference time (no premature
    # firing) and it fires within the window where the decode reads it. The TTFS
    # value is decoded per-source as (src_lat + T - fire)/T; full-window
    # accumulation would overcount shallow sources by (chip_latency - src_lat)/T
    # and saturate everything when latency >> T. ``single_spike_io`` selects the
    # single-spike (one spike per neuron on the wire) + consumer-side ramp path.
    latency_gated = True
    always_on_every_cycle = True
    single_spike_io = True

    def make_state(self, batch_size: int, n_neurons: int, device, dtype) -> NeuronState:
        return {
            "memb": torch.zeros(batch_size, n_neurons, device=device, dtype=dtype),
            "ramp_current": torch.zeros(batch_size, n_neurons, device=device, dtype=dtype),
            "has_fired": torch.zeros(batch_size, n_neurons, device=device, dtype=torch.bool),
        }

    def step(self, state, weight, inp, threshold, *, hw_bias, thresholding_mode,
             output_dtype=None) -> torch.Tensor:
        return ttfs_cycle_contribute_and_fire(
            state["memb"], state["ramp_current"], weight, inp, threshold,
            state["has_fired"],
            hw_bias=hw_bias, thresholding_mode=thresholding_mode, output_dtype=output_dtype,
        )


def cycle_neuron_policy(spiking_mode: str, schedule: str, firing_mode: str):
    """Build the per-cycle neuron policy for the pipelined cascade executor."""
    from mimarsinan.chip_simulation.spiking_semantics import is_cascaded_ttfs

    if is_cascaded_ttfs(spiking_mode, schedule):
        return TTFSGreedyCyclePolicy()
    return LIFCyclePolicy(firing_mode)
