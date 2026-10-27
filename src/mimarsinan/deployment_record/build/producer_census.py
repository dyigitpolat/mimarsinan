"""Producer writes of dynamic operand banks (N2 design §2.6): the payload cells each pass writes per inference,
on both planes; the writes each physical core takes are the period census's one slot rule
(``mapping/support/core_slots.place``), which places operand regions as it places static tiles.

**Declared deployment model.** A pass writes every operand region it holds once per sample: in the pass-major
order of a batch the next sample overwrites the payload, so a region is rewritten in every pass that holds it,
resident or not. Each streamed instance of a bank is placed as its own region holding the whole bank, so a bank
of L instances is written L times per inference on the program the packer builds (``placed_operand_cells``).
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from mimarsinan.deployment_record.build.write_census import WriteCensusError
from mimarsinan.mapping.ir.dynamic_bank import operand_cells_by_core as operand_cells_by_core  # one rule (F3)


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


def operand_cells_per_tile(operand: Sequence[int], tiles: Sequence[Sequence[Any]]) -> Tuple[int, ...]:
    """Per physical tile of a pass's cores (``tiles`` per core: a fused core's members), its operand cells. A core's
    operand regions sit on its one tile: no fused core takes the operand role (``canonical.fused_core_roles``), and a
    fused core holding them would leave each region's tile unstated, which is refused by name."""
    fused = [i for i, (cells, t) in enumerate(zip(operand, tiles)) if cells and len(t) > 1]
    if fused:
        raise WriteCensusError(f"cores {fused} hold operand regions and are fused of several physical tiles, so the "
                               "tile of each region is not stated (no fused core takes the operand role)")
    return tuple(c for cells, t in zip(operand, tiles) for c in (int(cells),) + (0,) * (len(t) - 1))
