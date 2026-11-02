"""The streamed LIF hop of the NF walk, per-cycle — including its ODIN twin."""

from __future__ import annotations

from typing import List, Optional

import torch

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.models.nn.activations.lif_serial import (
    SerialFoldSlot,
    arm_serial_fold,
    disarm_serial_fold,
    serial_fold_integer_lattice,
    serial_fold_theta,
)
from mimarsinan.models.nn.lif_kernels import in_measurement_plane
from mimarsinan.models.spiking.serial import SerialFoldUnsupportedError
from mimarsinan.spiking.cycle_trains import CycleStack, CycleTrain, encode_train, streams
from mimarsinan.spiking.segment_policy_lif_fold import CycleFoldSlot, cycle_fold_applies
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)


def run_streamed_lif_cycles(
    policy, *, node, perceptron, lif, forward_node, dep_trains: List[torch.Tensor | CycleTrain],
    dep_events: List[Optional[torch.Tensor | CycleTrain]], T: int, scale, keep_events: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor | CycleTrain]]:
    """One streamed hop: T per-cycle forwards. Returns ``(train, events)``.

    ``events`` is the hop's per-cycle emission MULTIPLICITY train under the
    per-event law (the currency the next hop folds); under the per-cycle law
    the 0/1 train a measurement-plane fold or the re-encode emitted (the
    currency the next hop's fold reads, U-SC2), else ``None``. [U-SC3] With
    ``keep_events`` False (no later node of the segment reads this hop) no
    event train is kept: ``None``.
    """
    slot = _arm_serial_slot(policy.soma_law, node, perceptron, lif, dep_events)
    fold = None if slot is not None else _arm_cycle_fold(policy.soma_law, node, perceptron, lif, dep_events)
    train, events = _run_window(node, lif, forward_node, dep_trains, dep_events, T,
                                slot if slot is not None else fold, keep_events)
    if fold is not None and not fold.agrees:
        # The twin's own pre-activation departs from the deployed decomposition: a gauge defect, not a tie, so the
        # window runs in the twin's own arithmetic and the departure shows as counts.
        train, events = _run_window(node, lif, forward_node, dep_trains, dep_events, T, None, keep_events)
    if policy.retime:
        # STE re-encode: forward = the deployed uniform train of the window
        # count; backward = the raw cascade's per-cycle surrogate path (a hard
        # re-encode severs every hop's gradient).
        rate = (train / scale).mean(dim=0).clamp(0.0, 1.0).detach()
        reencoded = encode_train(
            rate, T, phase_dither=policy.phase_dither,
            channels=policy.dither_keys(node, rate) if policy.phase_dither else None,  # [U-DOOR] its IR's keys
        )
        if streams() and (not isinstance(scale, torch.Tensor) or scale.dim() < train.dim()):
            # [U-SC3] the same elementwise arithmetic, cycle by cycle into the window's own train (cycle_trains)
            for t in range(T):
                train[t] = (reencoded[t] * scale).detach() + (train[t] - train[t].detach())
        else:
            if not isinstance(reencoded, torch.Tensor):
                reencoded = reencoded.materialize()
            train = (reencoded * scale).detach() + (train - train.detach())
        if not policy.soma_law.is_per_event and in_measurement_plane():
            events = reencoded  # the 0/1 train the next hop's measurement-plane fold reads
    return train, (events if keep_events else None)


def _run_window(node, lif, forward_node, dep_trains, dep_events, T: int, armed, keep_events: bool = True):
    """The hop's T cycles, through the armed fold when one is (``(train, the fold's emitted events or None)``); the
    fold's per-cycle emissions are kept only for a reader (``keep_events``)."""
    events_in = dep_events[0] if armed is not None else None
    # set_cycle_accurate resets the node: one window, one membrane.
    lif.set_cycle_accurate(True)
    if armed is not None:
        arm_serial_fold(lif, armed)
    try:
        window = CycleStack(T)  # [U-SC3] one train, written cycle by cycle where the walk streams
        for t in range(T):
            if armed is not None and events_in is not None:
                armed.feed(events_in[t])
            window.add(forward_node(node, [dt[t] for dt in dep_trains]))
            if armed is not None and not keep_events:
                armed.counts.clear()  # [U-SC3] no reader: the emissions are not held
        train = window.stacked()
        del window
        events = torch.stack(armed.counts, dim=0) if armed is not None and armed.counts else None
    finally:
        lif.set_cycle_accurate(False)
        if armed is not None:
            disarm_serial_fold(lif)
    return train, events


def _slot_unfold(node, upstream_events: torch.Tensor | CycleTrain, n_slots: int):
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
    """The per-event fold for this hop (``_run_window`` arms it), or refuse by name."""
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
        integer_lattice=serial_fold_integer_lattice(lif),
    )
    return slot


def _arm_cycle_fold(soma_law: SomaLaw, node, perceptron, lif, dep_events) -> Optional[CycleFoldSlot]:
    """[U-SC2 B1] The per-cycle deployed fold on this hop (``segment_policy_lif_fold``), or None where it does not apply
    or the mapper's unfold does not tile the hop's input (the twin keeps its own arithmetic)."""
    if not cycle_fold_applies(soma_law, lif, dep_events):
        return None
    n_slots = int(PerceptronTransformer().get_effective_weight(perceptron).shape[1])
    try:
        unfold = _slot_unfold(node, dep_events[0], n_slots)
    except (SerialFoldUnsupportedError, ValueError):
        return None
    cells = 1
    for d in dep_events[0].shape[2:]:
        cells *= int(d)
    if int(unfold.n_slots) != n_slots or int(unfold.source_size) != cells:
        return None
    return CycleFoldSlot(perceptron=perceptron, lif=lif, unfold=unfold)
