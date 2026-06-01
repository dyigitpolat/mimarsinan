"""Per-cycle neuron policy: the swappable fire/reset behavior of a cycle-based
spiking executor.

Both LIF and TTFS-cycle modes share the same per-cycle executor skeleton (cycle
loop, latency-window gating, span fill, output windowing, ``count/simulation_length``
decode). They differ only in the per-core integrate-and-fire step:

- **LIF** integrates inputs, fires whenever the membrane crosses threshold, and
  resets (subtractive / zero); it can fire every cycle.
- **TTFS-cycle** integrates inputs and fires **at most once**, latching its output
  high for the rest of the window; there is no reset.

This mirrors nevresim's ``fire_policy`` / ``reset_policy`` decomposition on the
Python side so the executor stays mode-agnostic.
"""

from __future__ import annotations

from typing import Dict

import torch

from mimarsinan.chip_simulation.spiking_semantics import is_ttfs_cycle_based
from mimarsinan.models.spiking.lif_core_step import lif_core_contribute_and_fire
from mimarsinan.models.spiking.ttfs_cycle_step import ttfs_cycle_contribute_and_fire

NeuronState = Dict[str, torch.Tensor]


class LIFCyclePolicy:
    """Multi-spike integrate-and-fire with Default/Novena reset."""

    def __init__(self, firing_mode: str):
        self.firing_mode = str(firing_mode)

    def make_state(self, batch_size: int, n_neurons: int, device, dtype) -> NeuronState:
        return {"memb": torch.zeros(batch_size, n_neurons, device=device, dtype=dtype)}

    def step(
        self,
        state: NeuronState,
        weight: torch.Tensor,
        inp: torch.Tensor,
        threshold: torch.Tensor,
        *,
        hw_bias: torch.Tensor | None,
        thresholding_mode: str,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return lif_core_contribute_and_fire(
            state["memb"],
            weight,
            inp,
            threshold,
            hw_bias=hw_bias,
            thresholding_mode=thresholding_mode,
            firing_mode=self.firing_mode,
            output_dtype=output_dtype,
        )


class TTFSCyclePolicy:
    """Fire-once, latched-output integrate-and-fire (no reset)."""

    def make_state(self, batch_size: int, n_neurons: int, device, dtype) -> NeuronState:
        return {
            "memb": torch.zeros(batch_size, n_neurons, device=device, dtype=dtype),
            "has_fired": torch.zeros(batch_size, n_neurons, device=device, dtype=torch.bool),
        }

    def step(
        self,
        state: NeuronState,
        weight: torch.Tensor,
        inp: torch.Tensor,
        threshold: torch.Tensor,
        *,
        hw_bias: torch.Tensor | None,
        thresholding_mode: str,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return ttfs_cycle_contribute_and_fire(
            state["memb"],
            weight,
            inp,
            threshold,
            state["has_fired"],
            hw_bias=hw_bias,
            thresholding_mode=thresholding_mode,
            output_dtype=output_dtype,
        )


def cycle_neuron_policy(spiking_mode: str, firing_mode: str):
    """Build the per-cycle neuron policy for a spiking mode."""
    if is_ttfs_cycle_based(spiking_mode):
        return TTFSCyclePolicy()
    return LIFCyclePolicy(firing_mode)
