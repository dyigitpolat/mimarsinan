"""What a placement costs: the quantity the packer minimizes IS the quantity we report."""

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


def programming_delta(softcore: Any, hard_core: Any) -> int:
    """Extra weight parameters that must be programmed if ``softcore`` joins ``hard_core``.

    Zero when the softcore's bank is already resident there -- which is exactly the reuse the
    bank-clustered scheduler exists to create, and exactly what ``weight_programming`` reports.
    An owned matrix is always programmed, so it always costs its own area.

    This is deliberately not a bank special case in the packer: it is the cost of the placement,
    and bank affinity falls out of minimizing it.
    """
    bank_id = getattr(softcore, "weight_bank_id", None)
    if bank_id is not None and bank_id in resident_bank_ids(hard_core):
        return 0
    return int(softcore.get_input_count()) * int(softcore.get_output_count())


def placement_cost(softcore: Any, hard_core: Any, *, remaining_capacity: int) -> tuple[int, int]:
    """Lexicographic placement cost: program as little as possible, then fit as tightly as possible.

    Ordering programming cost first is what preserves weight reuse once residency classes stop
    coinciding with weight banks. Further terms (routing, energy) extend the tuple without the
    packer learning anything about them.
    """
    return (programming_delta(softcore, hard_core), int(remaining_capacity))
