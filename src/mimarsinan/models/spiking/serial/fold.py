"""``lif_serial_fold``: THE event-serial soma law, defined once (plan §2.2).

Four implementations must agree bit-for-bit — the NF twin, the HCM reference
loop, the HCM packed executor, and (P3+) nevresim and the RTL. This module is
the torch half's single home: both torch executors and the NF decomposition
call THIS function, so a fold-order question has exactly one answer.

    for a in canonical_slot_order(A):        # ascending slots
        repeat e[a] times:                   # occurrences of one slot ADJACENT
            m := sat(m + w[a])               # saturating_unsigned clamps here
            if compare(theta, m): emit; reset(m)
    fold the bias event at the declared tail position
    return the per-neuron COUNT emitted this cycle
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from mimarsinan.chip_simulation.soma_law import BIAS_SLOT_TAIL, SomaLaw
from mimarsinan.mapping.platform.event_order import canonical_slot_order
from mimarsinan.models.nn.lif_kernels import (
    in_measurement_plane,
    lif_fire_and_reset,
    snap_membrane_to_lattice,
)
from mimarsinan.models.spiking.serial.refusals import (
    EmissionBoundExceededError,
    SerialFoldUnsupportedError,
)

EMISSION_COUNT_CEILING = 127
"""The count currency's ceiling. A window/cycle count travels as one signed
8-bit event count (nevresim's ``spike_t``), and the exporter prices the wire
from the same number, so 127 is the loudest bound EVERY implementation shares
(plan §2.2). It is asserted, never clamped: a silent saturation here is the
exact failure the count currency exists to make impossible."""


def require_serial_law(soma_law: SomaLaw) -> None:
    """The fold IS the per-event law; a per-cycle point must not reach it."""
    if not soma_law.is_per_event:
        raise SerialFoldUnsupportedError(
            f"lif_serial_fold executes the event-serial soma law, but the "
            f"resolved point declares firing_granularity="
            f"{soma_law.firing_granularity!r}: the per-cycle law integrates "
            f"the whole contribution before ONE compare and is executed by "
            f"LIFCyclePolicy. Dispatch through cycle_neuron_policy."
        )
    if soma_law.bias_slot != BIAS_SLOT_TAIL:
        raise SerialFoldUnsupportedError(
            f"lif_serial_fold implements bias_slot={BIAS_SLOT_TAIL!r} only; "
            f"the point declares bias_slot={soma_law.bias_slot!r}. A head or "
            f"multi-row bias placement changes the event order and is new "
            f"machinery, not a parameter."
        )


def require_event_counts(events: torch.Tensor) -> None:
    """Events cross the executor seam as PER-SLOT COUNTS, never a raw order.

    Adjacency is count-changing (§2.3), so an arbitrary arrival sequence has
    to be normalized by ``mapping.platform.event_order`` before it reaches a
    kernel. A fractional or negative entry is not a multiplicity at all — it
    is a rate or a sign error, and folding it would silently invent physics.
    """
    if not bool(torch.all(events >= 0)):
        raise SerialFoldUnsupportedError(
            "lif_serial_fold requires NON-NEGATIVE event multiplicities per "
            "slot; a negative entry is a sign folded into the wire instead of "
            "the weight."
        )
    if not torch.equal(events, torch.round(events)):
        raise SerialFoldUnsupportedError(
            "lif_serial_fold requires INTEGER event multiplicities per slot; "
            "a fractional entry is a RATE, and the per-event law has no "
            "fractional occurrence to fold. Normalize the stream through "
            "mapping.platform.event_order first."
        )


def _apply_event(
    memb: torch.Tensor,
    counts: torch.Tensor,
    delta: torch.Tensor,
    threshold: torch.Tensor,
    *,
    soma_law: SomaLaw,
    bounds: Optional[Tuple[float, float]],
    lattice_scale: Optional[float],
) -> None:
    """One event occurrence: charge, saturate, snap, compare, reset, count."""
    memb += delta
    if bounds is not None:
        memb.clamp_(bounds[0], bounds[1])
    if lattice_scale is not None and in_measurement_plane():
        snap_membrane_to_lattice(memb, lattice_scale)
    counts += lif_fire_and_reset(
        memb, threshold,
        thresholding_mode=soma_law.thresholding_mode,
        firing_mode=soma_law.firing_mode,
        output_dtype=memb.dtype,
    )


def lif_serial_fold(
    memb: torch.Tensor,
    weight: torch.Tensor,
    events: torch.Tensor,
    threshold: torch.Tensor,
    *,
    soma_law: SomaLaw,
    hw_bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Fold one cycle's events for every neuron in the layout.

    ``memb``   ``(B, *lead, N)`` — the membrane, updated IN PLACE.
    ``weight`` ``(*lead, N, A)`` — signed logical weight per (neuron, slot).
    ``events`` ``(B, *lead, A)`` — per-slot NON-NEGATIVE INTEGER multiplicities.
    Returns    ``(B, *lead, N)`` — the per-neuron COUNT emitted this cycle.

    ``lead`` is empty for the per-core reference loop and ``(G,)`` for the
    packed executor's bucket group; the fold is otherwise shape-agnostic, so
    both torch paths execute the SAME arithmetic in the same order.

    A masked add carries the occurrence: lanes with ``e[a] <= k`` receive a
    ZERO-magnitude event, which is a no-op for the compare too because every
    fire-path leaves ``m < theta`` and the window starts at ``V0*theta <
    theta`` (§1.2). That is the row-pair lemma, and it is what lets one
    vectorized pass be the serial fold.
    """
    require_serial_law(soma_law)
    require_event_counts(events)
    n_slots = int(weight.shape[-1])
    if int(events.shape[-1]) != n_slots:
        raise SerialFoldUnsupportedError(
            f"event slot count {int(events.shape[-1])} does not match the "
            f"weight's {n_slots} slots: the wire and the core disagree about "
            f"the canonical slot order."
        )
    bounds = soma_law.membrane_bounds
    quantum = soma_law.membrane_lattice_quantum
    lattice_scale = None if quantum is None else 1.0 / float(quantum)
    counts = torch.zeros_like(memb)

    # ONE host sync for the whole cycle: the per-slot occurrence depth.
    batch_axes = tuple(range(events.dim() - 1))
    depths = (
        events.amax(dim=batch_axes).to(torch.int64).tolist()
        if events.numel() else [0] * n_slots
    )
    for slot in canonical_slot_order(n_slots):
        depth = int(depths[slot])
        if depth <= 0:
            continue
        slot_weight = weight[..., slot]
        slot_events = events[..., slot].unsqueeze(-1)
        for occurrence in range(depth):
            _apply_event(
                memb, counts,
                slot_weight * (slot_events > occurrence).to(memb.dtype),
                threshold,
                soma_law=soma_law, bounds=bounds, lattice_scale=lattice_scale,
            )
    if hw_bias is not None:
        # The always-on (parameter-encoded) bias row sits at the TAIL of the
        # canonical order (SomaLaw.bias_slot); it arrives exactly once.
        _apply_event(
            memb, counts, hw_bias, threshold,
            soma_law=soma_law, bounds=bounds, lattice_scale=lattice_scale,
        )

    if counts.numel() and float(counts.max()) > EMISSION_COUNT_CEILING:
        raise EmissionBoundExceededError(
            f"a neuron emitted {int(counts.max())} spikes in ONE cycle, above "
            f"the count-currency ceiling {EMISSION_COUNT_CEILING}: the wire, "
            f"the records and the exported image all carry a count that no "
            f"longer fits. This is refused, NEVER clamped — a clamp would "
            f"report a number the deployment cannot produce. Lower the fan-in "
            f"or raise theta (plan §2.2's propagated emission bound)."
        )
    if output_dtype is not None and output_dtype != counts.dtype:
        return counts.to(output_dtype)
    return counts
