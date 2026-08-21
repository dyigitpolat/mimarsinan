"""The canonical event order: slot order, bias tail, row pairs, and adjacency — one home."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def canonical_slot_order(n_slots: int) -> range:
    """Axon slots are delivered in ascending index order, 0..n_slots-1."""
    if n_slots < 0:
        raise ValueError(f"slot count must be non-negative, got {n_slots}")
    return range(n_slots)


def bias_tail_slots(n_slots: int, n_bias_rows: int) -> range:
    """Bias (always-on) rows occupy the tail of the slot order (SomaLaw.bias_slot='tail')."""
    if n_bias_rows < 0 or n_bias_rows > n_slots:
        raise ValueError(
            f"bias rows must fit the slot order: {n_bias_rows} rows in {n_slots} slots"
        )
    return range(n_slots - n_bias_rows, n_slots)


# Row-pair contract (per_axon weight sign): logical slot a occupies physical rows
# (2a, 2a+1) = (excitatory, inhibitory); at most one member holds a nonzero
# magnitude per (slot, neuron), so a signed-once fold over slots equals the
# physical exc-then-inh delivery, provided the membrane is below theta on entry
# to every zero-magnitude event (guaranteed after any event, and at window start
# by the V0*theta < theta constraint).


def excitatory_row(slot: int) -> int:
    if slot < 0:
        raise ValueError(f"slot must be non-negative, got {slot}")
    return 2 * slot


def inhibitory_row(slot: int) -> int:
    if slot < 0:
        raise ValueError(f"slot must be non-negative, got {slot}")
    return 2 * slot + 1


def is_inhibitory_row(row: int) -> bool:
    if row < 0:
        raise ValueError(f"row must be non-negative, got {row}")
    return row % 2 == 1


def logical_slot_of_row(row: int) -> int:
    if row < 0:
        raise ValueError(f"row must be non-negative, got {row}")
    return row // 2


# Adjacency contract: the per-event fold is order-dependent, and adjacency of one
# slot's multiplicity changes counts (theta=5, w=[+3,-3], e=[2,1]: adjacent fires
# once, interleaved fires zero times). The wire therefore never carries raw event
# order: producers normalize to per-slot counts and consumers drain ascending
# with each slot's multiplicity adjacent. The v1 host router, the testbench
# driver, the torch folds, and nevresim's per-axon gather all implement this one
# contract.


def normalize_event_counts(
    events: Iterable[tuple[int, int]], n_slots: int
) -> list[int]:
    """Fold an arbitrary (slot, multiplicity) stream into per-slot counts, loud on bad input."""
    counts = [0] * len(canonical_slot_order(n_slots))
    for slot, multiplicity in events:
        if slot < 0 or slot >= n_slots:
            raise ValueError(f"slot {slot} outside the {n_slots}-slot order")
        if multiplicity < 0:
            raise ValueError(
                f"multiplicity must be non-negative, got {multiplicity} at slot {slot}"
            )
        counts[slot] += multiplicity
    return counts


def drain_events(counts: Sequence[int]) -> list[tuple[int, int]]:
    """Emit (slot, multiplicity) ascending, one entry per active slot — the delivery order."""
    return [
        (slot, int(counts[slot]))
        for slot in canonical_slot_order(len(counts))
        if counts[slot] > 0
    ]
