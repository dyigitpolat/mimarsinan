"""Per-cycle neuron policy: swappable fire/reset behavior for the pipelined cascade executor."""

from __future__ import annotations

from typing import Dict

import torch

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.chip_simulation.spiking_semantics import is_cascaded_ttfs

from mimarsinan.models.spiking.serial.policy import serial_policy_for
from mimarsinan.models.spiking.lif_core_step import (
    lif_core_advance,
    lif_core_contribute_and_fire,
)
from mimarsinan.models.spiking.ttfs_cycle_step import ttfs_cycle_contribute_and_fire

NeuronState = Dict[str, torch.Tensor]


def precharge_lif_states(neuron_states, thresholds, membrane_init: float) -> None:
    """[calculus sec.15.11] window-start membrane guard: pre-charge each LIF
    membrane by ``membrane_init * theta`` (per-neuron threshold units)."""
    if not membrane_init:
        return
    for state, threshold in zip(neuron_states, thresholds):
        state["memb"] += float(membrane_init) * threshold


class LIFCyclePolicy:
    """Multi-spike integrate-and-fire with Default/Novena reset.

    ``integer_lattice`` arms the exact chip-lattice membrane projection
    (integer-chip cells: true membranes are half-integer multiples in chip
    units; float noise must never decide a threshold tie)."""

    latency_gated = True
    always_on_every_cycle = False
    single_spike_io = False

    _CHIP_LATTICE_SCALE = 2.0  # half-integer quantum covers half-step init

    # The packed executor asks the policy how to be driven; only the
    # event-serial policy wants the per-axon tensor instead of the charge.
    serial = False

    def __init__(self, firing_mode: str, integer_lattice: bool = False, *,
                 membrane_bounds: "tuple[float, float] | None" = None,
                 membrane_rail_assert: bool = False):
        self.firing_mode = str(firing_mode)
        self.integer_lattice = bool(integer_lattice)
        # The declared register interval of a saturating membrane; None is the
        # default point's unbounded accumulator, byte-identical.
        self.membrane_bounds = membrane_bounds
        # Whether a rail is a FAILURE rather than this register's physics.
        self.membrane_rail_assert = bool(membrane_rail_assert)

    def _lattice_scale(self) -> float | None:
        return self._CHIP_LATTICE_SCALE if self.integer_lattice else None

    def make_state(self, batch_size: int, n_neurons: int, device, dtype) -> NeuronState:
        return {"memb": torch.zeros(batch_size, n_neurons, device=device, dtype=dtype)}

    def step(self, state, weight, inp, threshold, *, hw_bias, thresholding_mode,
             output_dtype=None) -> torch.Tensor:
        return lif_core_contribute_and_fire(
            state["memb"], weight, inp, threshold,
            hw_bias=hw_bias, thresholding_mode=thresholding_mode,
            firing_mode=self.firing_mode, output_dtype=output_dtype,
            lattice_scale=self._lattice_scale(),
            membrane_bounds=self.membrane_bounds,
            membrane_rail_assert=self.membrane_rail_assert,
        )

    def advance(self, state, contribution, threshold, *, thresholding_mode,
                output_dtype=None) -> torch.Tensor:
        """Elementwise cycle on a precomputed contribution — the SAME physics
        as ``step`` with the charge layout owned by the caller (packed path)."""
        return lif_core_advance(
            state["memb"], contribution, threshold,
            thresholding_mode=thresholding_mode,
            firing_mode=self.firing_mode, output_dtype=output_dtype,
            lattice_scale=self._lattice_scale(),
            membrane_bounds=self.membrane_bounds,
            membrane_rail_assert=self.membrane_rail_assert,
        )


class TTFSGreedyCyclePolicy:
    """Cascaded TTFS: single-spike, fire-once integrate-and-fire (no reset).

    Each neuron emits one spike; integration is a ramp reconstructed from
    single-spike arrivals via a persistent ``ramp_current``.
    """

    latency_gated = True
    always_on_every_cycle = True
    single_spike_io = True
    serial = False

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


def cycle_neuron_policy(
    spiking_mode: str, schedule: str, firing_mode: str,
    integer_lattice: bool = False, *, soma_law: SomaLaw,
):
    """Build the per-cycle neuron policy for the pipelined cascade executor.

    ``soma_law`` is the resolved point (keyword-only, default-free: a silently
    defaulted firing granularity is another chip's physics). At the DEFAULT
    point this returns exactly the object today's code returned — the serial
    class is constructed only when the point declares ``per_event``.
    """
    if is_cascaded_ttfs(spiking_mode, schedule):
        return TTFSGreedyCyclePolicy()
    serial = serial_policy_for(soma_law)
    if serial is not None:
        return serial
    return LIFCyclePolicy(
        firing_mode, integer_lattice=integer_lattice,
        membrane_bounds=soma_law.membrane_bounds,
        membrane_rail_assert=soma_law.asserts_no_saturation,
    )
