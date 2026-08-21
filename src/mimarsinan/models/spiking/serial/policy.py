"""``SerialLIFCyclePolicy``: the per-cycle policy surface over the serial fold.

Same three-method protocol the pipelined cascade executor already speaks
(``make_state``/``step``/``advance``) plus ``advance_events`` — the fourth
method the packed executor needs because a serial fold cannot start from a
pre-reduced contribution. ``advance`` therefore REFUSES: reducing the charge
before the compare is precisely the cycle-atomic assumption the point denies.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.models.spiking.serial.fold import lif_serial_fold
from mimarsinan.models.spiking.serial.refusals import (
    CycleAtomicRefusalError,
    SerialFoldUnsupportedError,
)

NeuronState = Dict[str, torch.Tensor]


class SerialLIFCyclePolicy:
    """Event-serial integrate-and-fire: >=0 spikes per neuron per cycle.

    The latency grid is per-CYCLE and the fold is intra-cycle, so latency
    gating, the local-time input re-alignment and the output-span windows are
    untouched: only what a cycle DOES changes, never when a core runs.
    """

    latency_gated = True
    always_on_every_cycle = False
    single_spike_io = False
    # The packed executor reads this to hand over the per-axon tensor instead
    # of the reduced charge; every other policy leaves it False.
    serial = True

    def __init__(self, soma_law: SomaLaw) -> None:
        self.soma_law = soma_law
        # Mirrors LIFCyclePolicy's attribute for the shared read sites.
        self.firing_mode = soma_law.firing_mode

    def _check_thresholding(self, thresholding_mode: str) -> None:
        if thresholding_mode != self.soma_law.thresholding_mode:
            raise SerialFoldUnsupportedError(
                f"the executor passed thresholding_mode="
                f"{thresholding_mode!r} while the resolved soma law declares "
                f"{self.soma_law.thresholding_mode!r}: the comparator is part "
                f"of the point and must not be re-decided at the call site."
            )

    def make_state(self, batch_size: int, n_neurons: int, device, dtype) -> NeuronState:
        return {"memb": torch.zeros(batch_size, n_neurons, device=device, dtype=dtype)}

    def step(self, state, weight, inp, threshold, *, hw_bias, thresholding_mode,
             output_dtype=None) -> torch.Tensor:
        """Per-core reference loop: ``inp`` is ALREADY the axon-ordered
        multiplicity vector, so path B reaches the fold with no call-site
        change beyond this policy."""
        self._check_thresholding(thresholding_mode)
        return lif_serial_fold(
            state["memb"], weight, inp, threshold,
            soma_law=self.soma_law, hw_bias=hw_bias, output_dtype=output_dtype,
        )

    def advance(self, state, contribution, threshold, *, thresholding_mode,
                output_dtype=None) -> torch.Tensor:
        del state, contribution, threshold, thresholding_mode, output_dtype
        raise CycleAtomicRefusalError(
            "advance() on a pre-reduced contribution is refused under "
            "firing_granularity='per_event': summing every axon's charge "
            "before ONE compare is the cycle-atomic law itself, and the "
            "per-axon identity it discards is exactly what decides how many "
            "spikes this cycle emits. Call advance_events() with the "
            "per-axon multiplicity tensor instead."
        )

    def advance_events(self, state, weight, events, threshold, *, hw_bias,
                       thresholding_mode, output_dtype=None) -> torch.Tensor:
        """Packed executor: the grouped ``(B, G, A)`` per-axon tensor the
        stage-flat gather already materializes, folded against ``(G, N, A)``
        weights. Returns the group's counts flattened back to ``(B, G*N)``."""
        self._check_thresholding(thresholding_mode)
        memb = state["memb"]
        lead = tuple(weight.shape[:-2])
        n_neurons = int(weight.shape[-2])
        view = memb.view(memb.shape[0], *lead, n_neurons) if lead else memb
        counts = lif_serial_fold(
            view, weight, events, threshold,
            soma_law=self.soma_law, hw_bias=hw_bias, output_dtype=output_dtype,
        )
        return counts.reshape(counts.shape[0], -1) if lead else counts


def serial_policy_for(soma_law: Optional[SomaLaw]) -> Optional[SerialLIFCyclePolicy]:
    """The serial policy when the point demands it, else ``None``.

    ``None`` is the byte-identical answer: the caller keeps whatever policy it
    builds today, and no default-point configuration constructs this class.
    """
    if soma_law is None or not soma_law.is_per_event:
        return None
    return SerialLIFCyclePolicy(soma_law)
