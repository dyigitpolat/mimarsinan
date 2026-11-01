"""[U-SC2 B1] The NF's per-cycle deployed fold: a float-mode core's threshold decided in the deployed core's arithmetic.

The deployed core integrates its effective weights (float32 values) times its 0/1 axon events in float64, where the
sum of every cycle's charges is exact, so a membrane can meet the threshold exactly (U-SC's DenseNet-BC fixture, seed 1,
sample 7: a cumulative membrane of exactly 2.0, final_units/usc2/probes/out/tie_margin_seed1.json). The NF integrated the
same charge in float32 through the perceptron's own layers (the raw weight on the gauged train, the normalization, the
division by theta) and held 2 - eps: one count apart. Inside the measurement plane (the parity twins, the certificates),
a float-mode hop under the per-cycle soma law therefore decides its threshold through the deployed fold: the mapper's
own unfold (``Mapper.serial_slot_unfold``, read, never re-derived), ``get_effective_weight`` and
``get_effective_bias`` in float64, the window-start pre-charge, and the deployed fire-and-reset kernel
(``lif_fire_and_reset``) at the deployed threshold. It decides only where the twin's own fused pre-activation agrees
with that decomposition to within the reduction residual (``decomposition_tolerance``, the event-serial slot's
contract): a departure beyond it is a gauge defect, not a tie, so the hop is re-run in the twin's own arithmetic and the
departure still shows as counts. Outside the measurement plane (training, tuning-plane telemetry), and on a hop with a
lattice (weight quantization: the lattice snap decides its ties exactly), a declared membrane register, a comparator
other than the point's, or no 0/1 event train, nothing is armed.
"""

from __future__ import annotations

from typing import Optional

import torch

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.models.nn.activations.lif_serial import decomposition_tolerance
from mimarsinan.models.nn.lif_kernels import in_measurement_plane, lif_fire_and_reset
from mimarsinan.spiking.cycle_trains import is_event_train
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

__all__ = ["CycleFoldSlot", "cycle_fold_applies"]

#: Float64 slot-table elements one decomposition holds at once (the unfold's (batch, cores, slots) table, in batch rows).
_TABLE_ELEMENTS = 1 << 24


def cycle_fold_applies(soma_law: SomaLaw, lif, dep_events: list) -> bool:
    """Whether this hop decides its threshold through the deployed fold (module docstring)."""
    return (in_measurement_plane() and not soma_law.is_per_event and soma_law.membrane_bounds is None
            and getattr(lif.if_node, "lattice_scale", None) is None
            and lif.thresholding_mode == soma_law.thresholding_mode
            and len(dep_events) == 1 and is_event_train(dep_events[0]))


class CycleFoldSlot:
    """One perceptron's armed per-cycle fold (the LIF's ``_serial_fold`` slot), in the deployed core's units (theta 1).

    ``agrees`` turns False at the first cycle whose decomposition departs from the twin's own pre-activation beyond
    the reduction residual; the hop's runner then re-runs the window in the twin's own arithmetic."""

    def __init__(self, *, perceptron, lif, unfold) -> None:
        transformer = PerceptronTransformer()
        self.weight = transformer.get_effective_weight(perceptron).detach().to(torch.float64)
        bias = transformer.get_effective_bias(perceptron)
        self.bias = None if bias is None else bias.detach().to(torch.float64)
        self.unfold = unfold
        self.membrane_init = float(getattr(lif, "membrane_init", 0.0))
        self.thresholding_mode = lif.thresholding_mode
        self.firing_mode = lif.firing_mode
        self.events: Optional[torch.Tensor] = None
        self.membrane: Optional[torch.Tensor] = None
        self.counts: list = []
        self.agrees = True

    def feed(self, events: torch.Tensor) -> None:
        """This cycle's upstream 0/1 events, in the NF's feature order."""
        self.events = events.reshape(events.shape[0], -1).to(self.weight.dtype)

    def _decompose(self, flat: torch.Tensor) -> torch.Tensor:
        """``(B, *cores, neurons)``: the mapper's slot tables against the effective weight and bias, float64, in batch
        chunks (a convolution's table is cores x slots per sample)."""
        weight = self.weight.to(flat.device)
        bias = None if self.bias is None else self.bias.to(flat.device)
        per_row = max(1, int(self.unfold.n_cores) * int(self.unfold.n_slots))
        rows = max(1, _TABLE_ELEMENTS // per_row)
        return torch.cat([torch.nn.functional.linear(self.unfold.unfold_events(part), weight, bias)
                          for part in flat.split(rows)])

    def run_cycle(self, x: torch.Tensor, safe_scale) -> torch.Tensor:
        """Fold one cycle and return the per-neuron spike (unscaled), in ``x``'s layout."""
        if self.events is None:
            raise RuntimeError("the armed cycle fold was driven without this cycle's events; feed() precedes forward")
        own = x / safe_scale
        decomposed = self._decompose(self.events)
        charge = self.unfold.group_major(own)
        if tuple(charge.shape) != tuple(decomposed.shape) or not float(
                (charge.to(decomposed.dtype) - decomposed).abs().max()) <= decomposition_tolerance(
                    charge, decomposed, theta=1.0, n_slots=int(self.weight.shape[1])):
            self.agrees = False
            return torch.zeros_like(x)
        if self.membrane is None:
            # The deployed window-start pre-charge (``precharge_lif_states``): membrane_init x theta, theta 1.
            self.membrane = torch.full_like(decomposed, self.membrane_init)
        self.membrane += decomposed
        fired = lif_fire_and_reset(self.membrane, torch.ones((), dtype=decomposed.dtype, device=decomposed.device),
                                   thresholding_mode=self.thresholding_mode, firing_mode=self.firing_mode,
                                   output_dtype=decomposed.dtype)
        self.events = None
        spikes = self.unfold.restore(fired.to(x.dtype), x)
        self.counts.append(spikes.detach())
        # Straight-through, as the event-serial slot: forward byte-identical, backward the normalized pre-activation.
        return spikes + (own - own.detach())
