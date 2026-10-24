"""Producer writes of dynamic operand banks (N2 design §2.6): the payload cells each pass writes per inference,
on both planes; the writes each physical core takes are the period census's one slot rule
(``write_census._slot_writes``), which places operand regions as it places static tiles.

**Declared deployment model.** A pass writes every operand region it holds once per sample: in the pass-major
order of a batch the next sample overwrites the payload, so a region is rewritten in every pass that holds it,
resident or not. Each streamed instance of a bank is placed as its own region holding the whole bank, so a bank
of L instances is written L times per inference on the program the packer builds (``placed_operand_cells``).
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple


def operand_cells_by_core(hard_core_mapping: Any, banks: Mapping) -> List[int]:
    """Per hard core of one built segment, the cells of its bound-bank regions (0 for a static core)."""
    return [sum(int(p["axons"]) * int(p["neurons"]) for p in placements
                if p.get("weight_bank_id") is not None and int(p["weight_bank_id"]) in banks)
            for placements in hard_core_mapping.soft_core_placements_per_hard_core]


def candidate_operand_cells(softcores: Sequence[Any], placements: Sequence[Tuple[int, int]], cores: Sequence[int],
                            dynamic_banks) -> List[int]:
    """Per hard core of one planned (pass, segment), the area of the operand soft cores placed on it, each soft
    core counted once per core (a fragment repeats its origin)."""
    seen = {(softcore, hardcore) for softcore, hardcore in placements
            if getattr(softcores[softcore], "bank_id", None) is not None
            and int(softcores[softcore].bank_id) in dynamic_banks}
    return [sum(int(softcores[s].area) for s, h in seen if h == core) for core in cores]


def placed_operand_cells(banks: int, instances: int, bank_cells: int) -> int:
    """The cells a product's operand regions take per inference on the packed program: every streamed instance
    holds its whole bank (the deployment model above)."""
    return int(banks) * int(instances) * int(bank_cells)
