"""What a placement costs, and why: bank affinity as the packer's stated preference."""

from __future__ import annotations

from typing import Any

RESIDENT_BANKS_ATTR = "resident_bank_ids"


def resident_bank_ids(hard_core: Any) -> frozenset:
    """Weight banks already programmed into this hardware core."""
    return frozenset(getattr(hard_core, RESIDENT_BANKS_ATTR, ()) or ())


def note_resident_bank(hard_core: Any, softcore: Any) -> None:
    """Record the bank a softcore brings, so later placements can reuse it."""
    bank_id = getattr(softcore, "weight_bank_id", None)
    if bank_id is None:
        return
    banks = getattr(hard_core, RESIDENT_BANKS_ATTR, None)
    if banks is None:
        banks = set()
        setattr(hard_core, RESIDENT_BANKS_ATTR, banks)
    banks.add(bank_id)


def bank_affinity_cost(softcore: Any, hard_core: Any) -> int:
    """0 when this core already hosts the softcore's bank, else the weight area it introduces.

    What this models, precisely: clustering the instances of one bank onto the same physical core
    is the PRECONDITION for `try_bank_clustered_passes` to keep that bank resident across passes,
    which is where `weight_programming` credits reuse (`schedule_weights_resident`).

    What it does NOT model: an intra-pass saving. Block-diagonal placement writes each instance
    into its own sub-region, so `weight_programming` charges every placement its area even when
    two instances share a bank and a core. This is an affinity preference, not a discount.

    Deliberately not a bank special case in the packer: it is a cost, and bank clustering falls
    out of minimizing it.
    """
    bank_id = getattr(softcore, "weight_bank_id", None)
    if bank_id is not None and bank_id in resident_bank_ids(hard_core):
        return 0
    return int(softcore.get_input_count()) * int(softcore.get_output_count())


def placement_cost(softcore: Any, hard_core: Any, *, remaining_capacity: int) -> tuple[int, int]:
    """Lexicographic: keep banks clustered first, then fit as tightly as possible.

    Ordering affinity first is what preserves cross-pass weight residency once residency classes
    stop coinciding with weight banks. Further terms (routing, energy) extend the tuple without
    the packer learning anything about them.
    """
    return (bank_affinity_cost(softcore, hard_core), int(remaining_capacity))
