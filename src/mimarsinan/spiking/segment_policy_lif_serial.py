"""The streamed LIF hop of the NF walk, per-cycle — including its ODIN twin."""

from __future__ import annotations

from typing import List, Optional

import torch

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.models.nn.activations.lif_serial import (
    SerialFoldSlot,
    arm_serial_fold,
    disarm_serial_fold,
    serial_fold_theta,
)
from mimarsinan.models.spiking.serial import SerialFoldUnsupportedError
from mimarsinan.spiking.spike_trains import uniform_spike_train
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)


def run_streamed_lif_cycles(
    policy, *, node, perceptron, lif, forward_node, dep_trains: List[torch.Tensor],
    dep_events: List[Optional[torch.Tensor]], T: int, scale,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """One streamed hop: T per-cycle forwards. Returns ``(train, events)``.

    ``events`` is the hop's per-cycle emission MULTIPLICITY train under the
    per-event law (the currency the next hop folds), and ``None`` under the
    per-cycle law, where the train's own 0/1 values already are it.
    """
    slot = _arm_serial_slot(policy.soma_law, node, perceptron, lif, dep_events)
    events_in = dep_events[0] if slot is not None else None
    # set_cycle_accurate resets the node: one window, one membrane.
    lif.set_cycle_accurate(True)
    try:
        outs = []
        for t in range(T):
            if slot is not None and events_in is not None:
                slot.feed(events_in[t])
            outs.append(forward_node(node, [dt[t] for dt in dep_trains]))
        train = torch.stack(outs, dim=0)
        events = (
            torch.stack(slot.counts, dim=0) if slot is not None else None
        )
    finally:
        lif.set_cycle_accurate(False)
        if slot is not None:
            disarm_serial_fold(lif)
    if policy.retime:
        # STE re-encode: forward = the deployed uniform train of the window
        # count; backward = the raw cascade's per-cycle surrogate path (a hard
        # re-encode severs every hop's gradient).
        retimed = uniform_spike_train(
            (train / scale).mean(dim=0).clamp(0.0, 1.0).detach(), T,
            phase_dither=policy.phase_dither,
        ) * scale
        train = retimed.detach() + (train - train.detach())
    return train, events


def _slot_unfold(node, upstream_events: torch.Tensor, n_slots: int):
    """The MAPPER's statement of how this hop tiles its upstream cells.

    Read, never re-derived: the twin folding an unfold of its own invention is
    exactly the failure the decomposition check exists to catch.
    """
    resolve = getattr(node, "serial_slot_unfold", None)
    if resolve is None:
        raise SerialFoldUnsupportedError(
            f"{type(node).__name__} declares no serial_slot_unfold, so the NF "
            f"event-serial twin has no statement of how the mapping fills this "
            f"hop's slot tables and would have to guess one."
        )
    return resolve(
        input_shape=tuple(int(d) for d in upstream_events.shape[2:]),
        n_slots=int(n_slots),
    )


def _arm_serial_slot(
    soma_law: SomaLaw, node, perceptron, lif, dep_events,
) -> Optional[SerialFoldSlot]:
    """Arm the per-event fold on this hop, or refuse by name."""
    if not soma_law.is_per_event:
        return None
    if len(dep_events) != 1 or dep_events[0] is None:
        raise SerialFoldUnsupportedError(
            f"the NF event-serial twin needs exactly ONE upstream event train "
            f"(got {len(dep_events)}, present="
            f"{[e is not None for e in dep_events]}): a multi-source hop's "
            f"axon slots are the mapper's concatenation order, which this "
            f"twin does not reproduce — folding a guessed order would report "
            f"a different physics as the deployed number."
        )
    if lif.thresholding_mode != soma_law.thresholding_mode:
        raise SerialFoldUnsupportedError(
            f"the NF activation compares with "
            f"{lif.thresholding_mode!r} while the resolved point declares "
            f"{soma_law.thresholding_mode!r}: the comparator is part of the "
            f"point and the twins must share it exactly."
        )
    transformer = PerceptronTransformer()
    weight = transformer.get_effective_weight(perceptron).detach()
    slot = SerialFoldSlot(
        soma_law=soma_law,
        weight=weight,
        bias=transformer.get_effective_bias(perceptron).detach(),
        theta=serial_fold_theta(lif),
        membrane_init=float(getattr(lif, "membrane_init", 0.0)),
        unfold=_slot_unfold(node, dep_events[0], int(weight.shape[1])),
    )
    arm_serial_fold(lif, slot)
    return slot
