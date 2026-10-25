"""The physical cores of a period census: each stated core's declared tiles, the pins' fit, and each core's writes.

A stated core is one declared core, or a fused core that occupies its components (axons summed over cores of one
declared type, ``canonical_fuse_hardcores``), the rule the residency planner reads (``core_type_tiles``). The runner
loads each execution stage's tiles of a type from that type's first core not pinned by a residency mark (R3), so
core j of a type takes one write from every pass that loads more than j tiles of the type (``SlotWrites``). Pins the
builder fits only by placing a tile on a larger core (its dominance matching, ``residency_rules.check_pinned_fit``)
leave a pass too few cores of its own type: the census is still stated, with no rewrite placed on a physical core.
[N2] Operand regions are placed by the same slot rule: a slot takes one producer write per inference from each pass
whose tile there holds them, its static writes count only the passes whose tile there holds static cells, and a kept
core with them is its own core.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import AbstractSet, Dict, Mapping, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.packing.residency_tiles import core_type_tiles
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.mapping.support.residency_rules import dominance_matching

#: A physical core type, ``(axons, neurons)``: the geometry a core is declared and built at.
CoreType = Tuple[int, int]
#: Per stated core of a pass, the declared tiles it occupies.
CoreTiles = Tuple[Tuple[CoreType, ...], ...]


@dataclass(frozen=True)
class SlotWrites:
    """One core type's static writes per period per physical core.

    The runner loads each pass's tiles from the first core of their type (``fragments.py`` R3), so the j-th tile of a
    type that a pass rewrites sits on that type's j-th core not pinned by a residency mark: core j takes one write per
    period of every pass whose j-th rewritten tile of the type holds static cells (N2: a tile of operand regions only
    takes producer writes, ``place``).
    """

    core_type: CoreType
    writes: Tuple[int, ...]


def kind_name(kind: CoreType) -> str:
    return f"{kind[0]} x {kind[1]}"


def stated_tiles(kind: CoreType, capacity: Mapping[CoreType, int]) -> Optional[Tuple[CoreType, ...]]:
    """The declared tiles one stated core occupies, or None when it is no declared core and no fused one."""
    if kind in capacity:
        return (kind,)
    try:
        return core_type_tiles(kind[0], kind[1], [LayoutHardCoreType(a, n, c) for (a, n), c in capacity.items()])
    except ResidencyPlanError:
        return None


def _tally(tiles: CoreTiles, cores: Sequence[int]) -> Counter:
    return Counter(t for i in cores for t in tiles[i])


#: Per core type, per core holding operand regions, (static writes per period, producer writes per inference) (N2).
ProducerSlots = Tuple[Tuple[CoreType, Tuple[Tuple[int, int], ...]], ...]


def place(resident: Sequence[bool], marks: Sequence[Sequence[int]], tiles: Sequence[CoreTiles],
          placed: Sequence[Sequence[int]], capacity: Mapping[CoreType, int], *,
          written: Optional[Sequence[Sequence[int]]] = None, static: Optional[Sequence[AbstractSet[int]]] = None,
          producers: Optional[Sequence[AbstractSet[int]]] = None,
          ) -> Tuple[Optional[Tuple[SlotWrites, ...]], Optional[ProducerSlots], Optional[str], Optional[str]]:
    """(per core type the static writes of each free core, or None; per core type the (static, producer) writes of
    each core holding operand regions [N2], or None; why no load is placed on its own type; why the pins do not fit at
    all). Every pass loads its ``placed`` cores beside every pass's pins; a pass's own type that holds them gives the
    slots, the dominance matching on the declared pool decides whether they fit at all. The slot rule counts the
    ``written`` cores (default ``placed``): per pass, ``static`` names the cores holding static cells (default all)
    and ``producers`` those holding operand regions (default none); a kept core with operand regions is its own core,
    one producer write per inference and no static write."""
    pinned: Counter = Counter()
    for whole, kept, cores in zip(resident, marks, tiles):
        if not whole:
            pinned += _tally(cores, [i for i in kept if i < len(cores)])
    per_pass = [_tally(cores, loads) for cores, loads in zip(tiles, placed)]
    short = [(kind, pinned[kind], max((c[kind] for c in per_pass), default=0)) for kind in sorted(capacity)
             if max((c[kind] for c in per_pass), default=0) > capacity[kind] - pinned[kind]]
    if short:
        kind, held, widest = short[0]
        own = (f"the residency marks pin {held} of the {capacity[kind]} cores of type {kind_name(kind)}, leaving "
               f"{capacity[kind] - held}, and a pass loads {widest} of its cores onto them")
        for loads in per_pass:
            need = loads + pinned
            if dominance_matching(need, capacity) < sum(need.values()):
                return None, None, None, own + "; no core of at least each tile's size holds the rest"
        return None, None, (f"{own}; the pass fits with tiles on cores at least its size (the builder's dominance "
                            "matching), so no rewritten core is placed on a physical core"), None
    slots, pairs = _slots(tiles, placed if written is None else written, capacity, static, producers)
    return slots, pairs, None, None


def _slots(tiles: Sequence[CoreTiles], written: Sequence[Sequence[int]], capacity: Mapping[CoreType, int],
           static: Optional[Sequence[AbstractSet[int]]], producers: Optional[Sequence[AbstractSet[int]]],
           ) -> Tuple[Tuple[SlotWrites, ...], ProducerSlots]:
    """The one slot rule (R3): the j-th written tile of a type sits on that type's j-th free core."""
    static_n: Dict[CoreType, Counter] = {}
    producer_n: Dict[CoreType, Counter] = {}
    widest: Counter = Counter()
    dedicated: set = set()
    for position, (cores, loads) in enumerate(zip(tiles, written)):
        held = frozenset() if producers is None else frozenset(producers[position])
        rank: Counter = Counter()
        for i in loads:
            for kind in cores[i]:
                j = rank[kind]
                rank[kind] += 1
                if static is None or i in static[position]:
                    static_n.setdefault(kind, Counter())[j] += 1
                if i in held:
                    producer_n.setdefault(kind, Counter())[j] += 1
        widest |= rank
        dedicated |= {kind for i in held - set(loads) if i < len(cores) for kind in cores[i]}
    out = tuple(SlotWrites(kind, tuple(static_n.get(kind, Counter())[j] for j in range(widest[kind])))
                for kind in sorted(capacity) if widest[kind])
    pairs = {kind: {(static_n.get(kind, Counter())[j], n) for j, n in producer_n.get(kind, Counter()).items()}
             | ({(0, 1)} if kind in dedicated else set()) for kind in sorted(set(producer_n) | dedicated)}
    return out, tuple((kind, tuple(sorted(slots))) for kind, slots in pairs.items())
