"""The NF's event-serial slot: the training twin drives the DEPLOYED fold.

``nn.Linear`` erases axon identity before the neuron sees the charge, so the NF
cannot reach an event-serial law through the fused pre-activation. The slot
carries the decomposition the mapper itself uses — ``get_effective_weight``,
the same columns in the same order — and hands it to ``lif_serial_fold``, the
one kernel both torch executors run. WHICH upstream cells fill each core's
table is the mapper's own unfold (``Mapper.serial_slot_unfold``), read here and
never re-derived: a fully-connected hop maps to one whole-input core, a
convolution to one core per output position over a shared weight bank. The
fused pre-activation is still computed and is checked against the decomposition
every cycle, refusing by name: that comparison IS the NF-order == mapper-order
contract, checked rather than believed (and a raise, never an assert — the
contract must survive ``-O``).
"""

from __future__ import annotations

from typing import Optional

import torch

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.mapping.platform.event_order import canonical_slot_order
from mimarsinan.mapping.platform.slot_unfold import (
    SerialSlotUnfold,
    WholeInputSlotUnfold,
)
from mimarsinan.models.spiking.serial import (
    SerialDecompositionMismatchError,
    SerialFoldUnsupportedError,
    lif_serial_fold,
)

_DECOMPOSITION_ATOL = 1e-6
# The two sides are the SAME sum reduced in two orders, so the residual scales
# with the summed charge AND with the coarser of the two dtypes' resolution:
# a 256-slot hop measures 1.6e-7 relative in float32 and 7e-4 under the
# trainer's autocast, while a wrong slot order moves the sum by O(1) relative.
# An absolute-only bound reads either as a slot-order defect.
_DECOMPOSITION_REDUCTION_SLACK = 4.0


class SerialFoldSlot:
    """One perceptron's armed per-event fold, in the deployed core's units.

    ``theta`` scales the normalized (theta = 1) NF charge into the same units
    the deployed core integrates in, so the declared membrane bounds and
    lattice quantum — which are register facts, not NF facts — apply verbatim.
    """

    def __init__(self, *, soma_law: SomaLaw, weight: torch.Tensor,
                 bias: Optional[torch.Tensor], theta: float,
                 membrane_init: float,
                 unfold: Optional[SerialSlotUnfold] = None) -> None:
        if weight.dim() != 2:
            raise SerialFoldUnsupportedError(
                f"the NF event-serial twin decomposes a rank-2 effective "
                f"weight (out, in); this perceptron's is rank {weight.dim()} "
                f"{tuple(weight.shape)}. Fan-in structure belongs to the "
                f"mapper's unfold (``serial_slot_unfold``), never to the "
                f"weight's rank."
            )
        n_slots = int(weight.shape[1])
        self.unfold: SerialSlotUnfold = (
            WholeInputSlotUnfold(n_slots) if unfold is None else unfold
        )
        if int(self.unfold.n_slots) != n_slots:
            raise SerialFoldUnsupportedError(
                f"the mapper's unfold fills {int(self.unfold.n_slots)}-slot "
                f"tables but this perceptron's effective weight has {n_slots} "
                f"columns: the twin and the deployment disagree about the "
                f"receptive field."
            )
        self.soma_law = soma_law
        self.theta = float(theta)
        self.weight = weight * self.theta
        self.bias = None if bias is None else bias * self.theta
        self.membrane_init = float(membrane_init)
        self.events: Optional[torch.Tensor] = None
        self.membrane: Optional[torch.Tensor] = None
        # (T, B, out) per-cycle emission counts, for the raster-level gate.
        self.counts: list = []

    def feed(self, events: torch.Tensor) -> None:
        """This cycle's upstream multiplicities, tiled into the mapper's slot tables."""
        flat = events.reshape(events.shape[0], -1)
        cells = int(flat.shape[1])
        if cells != int(self.unfold.source_size):
            raise SerialFoldUnsupportedError(
                f"the NF feature order carries {cells} slots but the mapper's "
                f"unfold consumes {int(self.unfold.source_size)} of them into "
                f"{int(self.unfold.n_cores)} core(s) of "
                f"{int(self.unfold.n_slots)} slots: the twin and the "
                f"deployment disagree about the canonical slot order."
            )
        n_slots = int(self.unfold.n_slots)
        if list(canonical_slot_order(n_slots)) != list(range(n_slots)):
            raise SerialDecompositionMismatchError(
                f"the NF twin feeds each core's gathered table straight "
                f"through as the slot order, but canonical_slot_order("
                f"{n_slots}) is not ascending: the mapper folds a DIFFERENT "
                f"order than this twin, and the two counts would diverge "
                f"silently."
            )
        self.events = self.unfold.unfold_events(flat)

    def _decomposition_tolerance(
        self, charge: torch.Tensor, decomposed: torch.Tensor,
    ) -> float:
        """The reduction residual two orderings of one sum may differ by."""
        eps = max(
            float(torch.finfo(charge.dtype).eps),
            float(torch.finfo(decomposed.dtype).eps),
        )
        magnitude = max(
            float(charge.abs().max()), float(decomposed.abs().max()), 1.0,
        )
        depth = float(int(self.weight.shape[1])) ** 0.5
        return (
            _DECOMPOSITION_ATOL * max(self.theta, 1.0)
            + _DECOMPOSITION_REDUCTION_SLACK * depth * eps * magnitude
        )

    def run_cycle(self, x: torch.Tensor, safe_scale) -> torch.Tensor:
        """Fold one cycle and return the per-neuron COUNT (unscaled)."""
        events = self.events
        if events is None:
            raise SerialFoldUnsupportedError(
                "the armed serial slot was driven without this cycle's event "
                "multiplicities; feed() must precede every forward."
            )
        charge = self.unfold.group_major(x / safe_scale) * self.theta
        decomposed = torch.nn.functional.linear(
            events.to(self.weight.dtype), self.weight, self.bias)
        max_error = float((charge - decomposed).abs().max())
        tolerated = self._decomposition_tolerance(charge, decomposed)
        if not max_error <= tolerated:
            # An `assert` would vanish under `python -O`, and this comparison
            # IS the NF-order == mapper-order contract of the whole twin.
            raise SerialDecompositionMismatchError(
                f"the NF event decomposition disagrees with the fused "
                f"pre-activation by {max_error} (max tolerated {tolerated}): "
                f"the twin's feature order or input scale is not the mapper's, "
                f"so this fold would run a DIFFERENT event order than the "
                f"deployment it is supposed to mirror."
            )
        if self.membrane is None:
            # One membrane per (core, neuron): a tiled hop's cores integrate
            # independently on the chip, and so must the twin.
            self.membrane = torch.full(
                tuple(events.shape[:-1]) + (int(self.weight.shape[0]),),
                self.membrane_init * self.theta,
                dtype=self.weight.dtype, device=self.weight.device,
            )
        counts = lif_serial_fold(
            self.membrane, self.weight, events.to(self.weight.dtype),
            torch.as_tensor(self.theta, dtype=self.weight.dtype,
                            device=self.weight.device),
            soma_law=self.soma_law, hw_bias=self.bias,
        )
        self.events = None
        typed = self.unfold.restore(counts.to(x.dtype), x)
        self.counts.append(typed.detach())
        # Straight-through: the fold is a hard integer count with no gradient,
        # and every theta of charge buys exactly one spike, so the normalized
        # pre-activation IS the count's first-order surrogate. Forward is
        # BYTE-identical (the residual is exactly zero without autograd); the
        # backward path is the one the endpoint stages train through.
        surrogate = self.unfold.restore(charge, x) / self.theta
        return typed + (surrogate - surrogate.detach())


def arm_serial_fold(lif, slot: SerialFoldSlot) -> None:
    """Install the fold on a LIF activation for the duration of one hop."""
    lif._serial_fold = slot


def disarm_serial_fold(lif) -> None:
    lif._serial_fold = None


def serial_fold_theta(lif) -> float:
    """The deployed core's threshold in the NF's own units.

    Without weight quantization the mapper emits ``theta = 1`` and the NF's
    normalized membrane IS the core's; with it armed, the integer-lattice
    scale already records ``2*theta_int`` (the existing NF<->HCM unit
    contract), so the same read serves both.
    """
    lattice_scale = getattr(lif.if_node, "lattice_scale", None)
    return 1.0 if not lattice_scale else float(lattice_scale) / 2.0
