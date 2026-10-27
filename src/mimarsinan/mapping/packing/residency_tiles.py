"""[E85] The cores a set of layout specs packs into, the declared physical tiles each packed core occupies, the
tile groups a layer splits into across passes, each spec's layer, [F2] the program order of a program's tile
groups, and [F3] the cells of the dynamic operand-bank regions each packed core holds."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from mimarsinan.mapping.latency.core_graph import condensed_order
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutPackingResult, LayoutSoftCoreSpec
from mimarsinan.mapping.packing.canonical import OPERAND_ROLE
from mimarsinan.mapping.platform.mapping_structure import PoolAdmissionError, ResidencyPlanError

#: One packed core: its occupied cells (used axons x used neurons, the rectangle the record's ``params_bytes``
#: prices) and the physical tiles it occupies (a fused core counts its components).
PackedCore = Tuple[int, int]
#: The declared physical cores one packed core occupies, (axons, neurons) per component.
CoreTiles = Tuple[Tuple[int, int], ...]
#: [W03] One tile group: the indices (into the packed specs) that share packed cores, and those cores, in order.
TileGroup = Tuple[Tuple[int, ...], Tuple[PackedCore, ...]]
#: [F2] One unit of a program: its segment, its layer, its spec indices and its packed cores, then (F3) per core
#: further facts in core order (its operand-bank cells).
ProgramUnit = Tuple[Any, ...]


def declared_tiles(axons: int, neurons: int, declared: Iterable[Tuple[int, int]]) -> Optional[CoreTiles]:
    """The declared physical cores one core of geometry ``axons`` x ``neurons`` occupies: itself when the geometry is
    declared, else its unique decomposition into cores of one declared type (a fused core sums the axons of cores
    of one type, ``canonical_fuse_hardcores``); None when the declared types do not determine it."""
    kinds = {(int(a), int(n)) for a, n in declared}
    if (int(axons), int(neurons)) in kinds:
        return ((int(axons), int(neurons)),)
    fits = sorted(a for a, n in kinds if n == int(neurons) and 0 < a and int(axons) % a == 0)
    return ((fits[0], int(neurons)),) * (int(axons) // fits[0]) if len(fits) == 1 else None


def core_type_tiles(axons: int, neurons: int, core_types: Sequence[LayoutHardCoreType],
                    components: Optional[Sequence[int]] = None) -> Tuple[Tuple[int, int], ...]:
    """The declared physical cores one packed core occupies: itself, or a fused core's components, read from
    ``components`` or decomposed uniquely (``declared_tiles``)."""
    if components:
        return tuple((int(a), int(neurons)) for a in components)
    declared = sorted({(int(t.max_axons), int(t.max_neurons)) for t in core_types})
    tiles = declared_tiles(axons, neurons, declared)
    if tiles is None:
        ways = sum(1 for a, n in declared if n == int(neurons) and 0 < a and int(axons) % a == 0)
        raise ResidencyPlanError(
            f"residency_plan: a fused {axons}x{neurons} core decomposes into the declared core types "
            f"{declared} in {ways} ways, so its physical tiles are not determined")
    return tiles


def tile_extents(used_axons: int, used_neurons: int, tiles: CoreTiles) -> Tuple[Tuple[int, int], ...]:
    """Per physical tile of one core, its used (rows, columns): a fused core's used rows fill its members in order
    (its axons are its members' axons concatenated), every member holding the core's used columns."""
    out, left = [], int(used_axons)
    for axons, _neurons in tiles:
        rows = min(int(axons), max(0, left))
        left -= rows
        out.append((rows, int(used_neurons)))
    return tuple(out)


def tile_rows(rows: int, tiles: CoreTiles) -> Tuple[int, ...]:
    """A count of one core's used rows (the rows its per-row sign register signs, CL-SG) per physical tile: the rows
    fill its members in order, as ``tile_extents`` fills them, and each member counts its own."""
    return tuple(r for r, _ in tile_extents(int(rows), 0, tiles))


def expand_to_tiles(cores: Sequence[Tuple[int, Optional[Tuple[int, int]], CoreTiles]], marks: Iterable[int]):
    """A pass's per-core census as one entry per physical tile: ``cores`` per core (cells, used (rows, columns) or
    None, its tiles), ``marks`` the kept core indices. Returns per tile (cells, kept tile indices, extents or None,
    tile types); a core's cells split by ``tile_extents`` where its extent is stated, else on its first tile."""
    kept = frozenset(int(m) for m in marks)
    cells: List[int] = []
    extents: List[Tuple[int, int]] = []
    types: List[Tuple[int, int]] = []
    tile_marks: List[int] = []
    for index, (core_cells, extent, tiles) in enumerate(cores):
        split = tile_extents(extent[0], extent[1], tiles) if extent is not None else None
        for j, tile in enumerate(tiles):
            if index in kept:
                tile_marks.append(len(cells))
            cells.append(split[j][0] * split[j][1] if split is not None else (int(core_cells) if j == 0 else 0))
            types.append((int(tile[0]), int(tile[1])))
            if split is not None:
                extents.append(split[j])
    stated = all(extent is not None for _, extent, _ in cores)
    return tuple(cells), frozenset(tile_marks), tuple(extents) if stated else None, tuple(types)


def _pack(specs: Sequence[LayoutSoftCoreSpec], core_types: Sequence[LayoutHardCoreType], *, allow_coalescing: bool,
          allow_splitting: bool, bounded: bool, placements: bool = False) -> LayoutPackingResult:
    bound = sum(
        math.ceil(s.input_count / max(1, min(t.max_axons for t in core_types)))
        * math.ceil(s.output_count / max(1, min(t.max_neurons for t in core_types))) + 1
        for s in specs
    )
    pack = pack_layout(
        softcores=list(specs),
        core_types=list(core_types) if bounded else [LayoutHardCoreType(t.max_axons, t.max_neurons, bound)
                                                     for t in core_types],
        allow_neuron_splitting=allow_splitting, allow_coalescing=allow_coalescing, collect_placements=placements,
    )
    if not pack.feasible:  # on the declared counts (``bounded``) a refusal of the pool's admission [W03]
        raise (PoolAdmissionError if bounded else ResidencyPlanError)(
            f"residency_plan: the cores of {specs[0].name!r} pack on no declared core type "
            f"{'on the declared pool ' if bounded else ''}({pack.error})")
    return pack


def _cores(pack: LayoutPackingResult, core_types: Sequence[LayoutHardCoreType],
           specs: Sequence[LayoutSoftCoreSpec] = ()) -> Tuple[Tuple[PackedCore, ...], Tuple[CoreTiles, ...]]:
    """Per packed core its occupied cells (used axons, less the operand axons of the specs placed on it [F2], as
    ``CoreOccupancy`` counts a built core, times used neurons) and tiles."""
    snaps = tuple(pack.used_core_snapshots or ())
    tiles = tuple(core_type_tiles(snap.axons_per_core, snap.neurons_per_core, core_types) for snap in snaps)
    port = [0] * len(snaps)
    for origin, core in (pack.placements or ()) if any(s.operand_axons for s in specs) else ():
        port[int(core)] += int(specs[int(origin)].operand_axons)
    return tuple(((int(s.used_axons) - p) * int(s.used_neurons), len(t)) for s, t, p in zip(snaps, tiles, port)), tiles


def _operand(specs: Sequence[LayoutSoftCoreSpec]) -> bool:
    return any(getattr(s, "core_role", None) == OPERAND_ROLE for s in specs)


def producer_cells(pack: LayoutPackingResult, specs: Sequence[LayoutSoftCoreSpec]) -> Tuple[int, ...]:
    """[F3] Per packed core, the cells of the dynamic operand-bank regions it holds (each region's axons x neurons,
    the regions of the banks of ``specs`` whose role is the operand's), as ``dynamic_bank.operand_cells_by_core``
    counts a built core: their payload is written every inference. () where no spec holds an operand bank."""
    if not _operand(specs):
        return ()
    banks = {s.bank_id for s in specs if getattr(s, "core_role", None) == OPERAND_ROLE}
    if None in banks:
        raise ResidencyPlanError("residency_plan: an operand bank's soft core states no bank id, so the cores holding "
                                 "its payload are not determined")
    return tuple(sum(int(a) * int(n) for bank, a, n, *_ in regions if bank in banks)
                 for regions in pack.core_regions or ())


def pack_specs(specs: Sequence[LayoutSoftCoreSpec], core_types: Sequence[LayoutHardCoreType], *,
               allow_coalescing: bool, allow_splitting: bool,
               bounded: bool = False) -> Tuple[Tuple[PackedCore, ...], Tuple[CoreTiles, ...], Tuple[int, ...]]:
    """Every core ``specs`` pack into on the declared core types with the layout packer (``pack_layout``): per core
    its ``PackedCore``, its physical tiles and [F3] its operand-bank cells (``producer_cells``, () without operand
    banks). The pool is unbounded (a layer packed alone on its best type), or ``bounded`` to the declared counts (a
    phase, packed as ``collect_noc_fragments`` packs a pass)."""
    pack = _pack(specs, core_types, allow_coalescing=allow_coalescing, allow_splitting=allow_splitting,
                 bounded=bounded, placements=any(s.operand_axons for s in specs) or _operand(specs))
    return (*_cores(pack, core_types, specs), producer_cells(pack, specs))


def packed_cores(specs: Sequence[LayoutSoftCoreSpec], core_types: Sequence[LayoutHardCoreType], *,
                 allow_coalescing: bool, allow_splitting: bool, bounded: bool = False) -> Tuple[PackedCore, ...]:
    """``pack_specs``' cores only."""
    return pack_specs(specs, core_types, allow_coalescing=allow_coalescing, allow_splitting=allow_splitting,
                      bounded=bounded)[0]


def unit_tile_count(specs: Sequence[LayoutSoftCoreSpec], core_types: Sequence[LayoutHardCoreType], *,
                    allow_coalescing: bool, allow_splitting: bool) -> int:
    """Physical cores one layer's specs pack into alone on the declared core types (the pool unbounded)."""
    return sum(t for _, t in packed_cores(specs, core_types, allow_coalescing=allow_coalescing,
                                          allow_splitting=allow_splitting))


def tile_groups(specs: Sequence[LayoutSoftCoreSpec], core_types: Sequence[LayoutHardCoreType], *,
                allow_coalescing: bool, allow_splitting: bool, bonds: Sequence[Tuple[int, int]] = (),
                producer: bool = False) -> Tuple[Tuple[Any, ...], ...]:
    """[W03] The finest units a layer's ``specs`` split into across passes: packed alone (an unbounded pool), the
    specs that share a packed core, or that a ``bond`` ties (a dependency, a coalescing group), form one group with
    every core they occupy; groups in order of their first spec, so they partition the layer's packed tiles. Each
    group is (spec indices, cores), with ``producer`` [F3] (spec indices, cores, each core's operand-bank cells)."""
    pack = _pack(specs, core_types, allow_coalescing=allow_coalescing, allow_splitting=allow_splitting,
                 bounded=False, placements=True)
    parent = list(range(len(specs)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    holder: Dict[int, int] = {}
    for origin, core in pack.placements or ():
        first = holder.setdefault(int(core), int(origin))
        parent[root(int(origin))] = root(first)
    for a, b in bonds:
        parent[root(int(a))] = root(int(b))
    members: Dict[int, List[int]] = {}
    for i in range(len(specs)):
        members.setdefault(root(i), []).append(i)
    held: Dict[int, set] = {}
    for origin, core in pack.placements or ():
        held.setdefault(root(int(origin)), set()).add(int(core))
    cores, _ = _cores(pack, core_types, specs)
    cells = (producer_cells(pack, specs) or (0,) * len(cores)) if producer else ()
    return tuple((tuple(group), tuple(cores[c] for c in sorted(held.get(key, ()))),
                  *((tuple(cells[c] for c in sorted(held.get(key, ()))),) if producer else ()))
                 for key, group in sorted(members.items(), key=lambda item: item[1][0]))


def layer_of_each_spec(specs: Sequence[LayoutSoftCoreSpec], nodes: Sequence[Any]) -> List[Any]:
    """Layer identity per spec: its ``perceptron_index``, else the layer it feeds (its consumer of the lowest index)."""
    consumers: Dict[int, List[int]] = {}
    for node in nodes:
        for source in node.sources:
            consumers.setdefault(int(source), []).append(int(node.core_id))
    layer: List[Any] = [spec.perceptron_index for spec in specs]
    for index in range(len(specs) - 1, -1, -1):
        if layer[index] is None:
            fed = [layer[c] for c in sorted(consumers.get(index, ())) if layer[c] is not None]
            layer[index] = fed[0] if fed else None
    missing = [specs[i].name or i for i, value in enumerate(layer) if value is None]
    if missing:
        raise ResidencyPlanError(
            f"residency_plan needs a layer identity for every core: {missing[:5]} carry no perceptron_index and "
            f"feed no core that does")
    return layer


def program_order(groups: Sequence[ProgramUnit], nodes: Sequence[Any]) -> List[ProgramUnit]:
    """[F2] A program's tile groups (``groups`` in layer order: by segment, then each layer's first spec) in a program
    order over the cut nodes' reads (``core_id`` and ``sources`` spec indices, within a segment): the layers of each
    segment ranked by a topological order of the layer graph, ties to layer order; the groups in a topological order
    of the group graph, ties to (segment, layer rank, layer-order position); a strongly connected set of groups one
    unit, its specs in index order, its cores (and each per-core fact after them) in group order. Wherever layer
    order is a program order, ``groups`` unchanged (each order is then its key order,
    ``core_graph.topological_order``)."""
    unit_of = {int(spec): u for u, group in enumerate(groups) for spec in group[2]}
    reads: List[set] = [set() for _ in groups]
    for node in nodes:
        v = unit_of[int(node.core_id)]
        reads[v].update(unit_of[int(s)] for s in node.sources
                        if int(s) in unit_of and unit_of[int(s)] != v and groups[unit_of[int(s)]][0] == groups[v][0])
    layer_of: Dict[Tuple[int, Any], int] = {}
    for group in groups:
        layer_of.setdefault((group[0], group[1]), len(layer_of))
    layer = [layer_of[(group[0], group[1])] for group in groups]
    layer_reads: List[set] = [set() for _ in layer_of]
    for v, read in enumerate(reads):
        layer_reads[layer[v]].update(layer[u] for u in read if layer[u] != layer[v])
    rank = {m: r for r, members in enumerate(condensed_order([sorted(r) for r in layer_reads], key=int))
            for m in members}
    out: List[ProgramUnit] = []
    for members in condensed_order([sorted(r) for r in reads], key=lambda u: (groups[u][0], rank[layer[u]], u)):
        first = groups[members[0]]
        out.append(first if len(members) == 1 else (
            first[0], first[1], tuple(sorted(s for u in members for s in groups[u][2])),
            *(tuple(fact for u in members for fact in groups[u][k]) for k in range(3, len(first)))))
    return out
