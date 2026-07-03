"""Per-cycle neuron policy: swappable fire/reset behavior for the pipelined cascade executor."""

from __future__ import annotations

from typing import Dict

import torch

from mimarsinan.chip_simulation.spiking_semantics import is_cascaded_ttfs

from mimarsinan.models.spiking.lif_core_step import lif_core_contribute_and_fire
from mimarsinan.models.spiking.ttfs_cycle_step import ttfs_cycle_contribute_and_fire

NeuronState = Dict[str, torch.Tensor]


class LIFCyclePolicy:
    """Multi-spike integrate-and-fire with Default/Novena reset."""

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

    Each neuron emits one spike; integration is a ramp reconstructed from
    single-spike arrivals via a persistent ``ramp_current``.
    """

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
    if is_cascaded_ttfs(spiking_mode, schedule):
        return TTFSGreedyCyclePolicy()
    return LIFCyclePolicy(firing_mode)
