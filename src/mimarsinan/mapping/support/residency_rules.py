"""[E85] The refusal rules of an executed residency plan: one predicate per rule, applied on both planes.

The builder (IR cores) and the shape-only plane (the layout walk's specs and wire census) plan one composition
(``schedule/residency_composition.py``) and refuse by name through the same functions here, over the same facts:

- ``ProgramGraph``: the program's nodes in program order with their producers. On the builder it is the IR
  (``of_ir``); on the shape-only plane the layout walk (``of_walk``), whose node ids are allocated in the order the IR
  walk creates its nodes and whose neural producers are the cells its wire census materializes, so both planes read
  one graph and name one node by one id (on ``IRMapping`` and on the deferred ``LayoutIRMapping`` walk alike); the
  plane's spec segments are tied to it (``require_depth_segments``).
- ``require_segmentation_parity``: the key plans the depth segments (a neural core's segment rises across a host
  dependency, ``compute_segment_ids``: the specs' ``segment_id``, which ``require_depth_segments`` ties to the graph),
  admitted only where every neural-to-neural edge keeps its class in the reference segmentation
  (``partition_ir_graph``: runs of consecutive neural cores between host ops, which the identity and U1 builders
  execute).
- ``check_pinned_fit``: every pass with the other passes' pins fits the declared pool (each plane on its own packing).
- ``unscheduled_pack_holds`` [fix_b]: the fit rule with scheduling allowed or not; a program whose unscheduled pack
  holds on the declared pool is built so and pinned (each plane on its own packer); ``fit_rule_applies`` [fix_c]:
  under U1 and every declared plan but the stateless reload baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from mimarsinan.mapping.ir import IRSource, NeuralCore
from mimarsinan.mapping.ir.dynamic_bank import payload_sources
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.layout.segmentation import compute_segment_ids
from mimarsinan.mapping.platform.mapping_structure import PoolAdmissionError, ResidencyPlanError, residency_declaration
from mimarsinan.mapping.support.period_census import STEP_STATELESS
from mimarsinan.mapping.packing.residency_tiles import CoreTiles, core_type_tiles


@dataclass(frozen=True)
class ProgramGraph:
    """A program's nodes in program order: per node its id, its spec index on the plane (None: a host op) and the
    ids of the nodes it reads (raw input, always-on and off cells excluded)."""

    ids: Tuple[int, ...]
    specs: Tuple[Optional[int], ...]
    producers: Tuple[FrozenSet[int], ...]

    @classmethod
    def of_ir(cls, ir_graph: Any, spec_of: Mapping[int, int]) -> "ProgramGraph":
        """The IR's program (``ir_graph.nodes``, the order ``partition_ir_graph`` reads), ``spec_of`` a neural
        core's id to its spec index on the builder's plane; [F3] an operand bank's instance also reads the host op
        writing its payload (``dynamic_bank.payload_sources``)."""
        nodes, payload = list(ir_graph.nodes), payload_sources(ir_graph)
        return cls(
            ids=tuple(int(n.id) for n in nodes),
            specs=tuple(int(spec_of[int(n.id)]) if isinstance(n, NeuralCore) else None for n in nodes),
            producers=tuple(frozenset(int(s.node_id) for s in np.asarray(n.input_sources, dtype=object).ravel()
                                      if isinstance(s, IRSource) and s.node_id >= 0)
                            | frozenset([payload[int(n.id)]] if int(n.id) in payload else []) for n in nodes),
        )

    @classmethod
    def of_walk(cls, mapping: Any) -> "ProgramGraph":
        """The layout walk's program (``LayoutIRMapping`` or ``IRMapping`` after ``map``, the wire census collected),
        its spec indices those of ``mapping.layout_softcores``, as ``census_of_walk`` reads the walk's records.

        A neural node's producers are the nodes of its wire-census record (``record_emission_census``: the distinct
        source cells it reads, materialized), the IR's on either walk. The recorded node-id sets are not on the deferred
        walk: a ``LayoutSourceView`` keeps every producer of a view it slices (tier-0 t0_01: IR node 33 reads [1], its
        recorded set is [1, ..., 12]). A host op's producers are the recorded set, the walk holding no census for it:
        the IR's on ``IRMapping``; on the deferred walk a host op reading a slice of a multi-producer view records a
        superset (t0_30: 96 of 99 host ops, their producers at one depth). [F3] An operand bank's instance also reads
        the host op that writes its payload (the walk's ``dynamic_bank_bindings``, by its spec's bank), as the walk's
        spec segments count it. They feed only the depth segments, which ``require_depth_segments`` ties to the plane's
        spec segments. A walk without the census refuses by name."""
        if not getattr(mapping, "collect_wire_census", False):
            raise ResidencyPlanError(
                "residency_plan: the program graph reads the layout walk's wire census (a neural node's producers are "
                "the cells it materializes; a deferred view's recorded node ids are a superset); walk with "
                "collect_wire_census=True")
        neural: Mapping[int, bool] = mapping._node_is_neural
        to_spec: Mapping[int, int] = mapping._node_id_to_softcore_idx
        reads: Mapping[int, Set[int]] = mapping._node_input_node_ids
        cells: Mapping[int, Tuple[Mapping[int, int], int, int]] = mapping._node_wire_census
        ids = sorted(int(i) for i in neural)
        bindings = getattr(mapping, "dynamic_bank_bindings", None) or {}
        writer = {int(bank): int(binding.source_node_id) for bank, binding in bindings.items()}
        specs = getattr(mapping, "layout_softcores", None) or ()
        banks = {i: getattr(specs[int(to_spec[i])], "bank_id", None) for i in ids
                 if writer and neural[i] and int(to_spec[i]) < len(specs)}
        payload = {i: writer[int(b)] for i, b in banks.items() if b is not None and int(b) in writer}
        return cls(ids=tuple(ids), specs=tuple(int(to_spec[i]) if neural[i] else None for i in ids),
                   producers=tuple(frozenset(int(p) for p in (cells[i][0] if neural[i] else reads.get(i, ())))
                                   | frozenset([payload[i]] if i in payload else []) for i in ids))

    def node_of_spec(self) -> Dict[int, int]:
        return {int(s): i for i, s in zip(self.ids, self.specs) if s is not None}

    def require_specs(self, count: int) -> None:
        """The graph is the program of the plane's ``count`` specs, each exactly once; refuses by name otherwise."""
        held = sorted(int(s) for s in self.specs if s is not None)
        if held != list(range(int(count))):
            raise ResidencyPlanError(
                f"residency_plan: the program graph holds spec indices {held[:8]}{'...' if len(held) > 8 else ''} "
                f"({len(held)} neural nodes), not each of the plane's {count} specs once; it is not this plane's walk")

    def require_depth_segments(self, segments: Sequence[int]) -> None:
        """The plane's segment of each spec groups the specs as the graph's depth segments do, so the composition
        plans the segments the rules check; refuses by name otherwise."""
        depth, node_of = self.depth_segments(), self.node_of_spec()
        pairs = sorted({(int(segments[i]), depth[node_of[i]]) for i in range(len(segments))})
        if len(pairs) == len({a for a, _ in pairs}) == len({b for _, b in pairs}):
            return
        raise ResidencyPlanError(
            f"residency_plan: the plane's spec segments are not its program graph's depth segments "
            f"(compute_segment_ids over the graph's producers); (spec segment, depth segment) pairs {pairs[:8]}; the "
            f"composition would plan segments the refusal rules never checked")

    def reference_runs(self) -> Dict[int, int]:
        """Per neural node, its run in the reference segmentation: the count of host ops before it in program
        order (two nodes share a ``partition_ir_graph`` segment iff they share a run)."""
        run, out = 0, {}
        for node, spec in zip(self.ids, self.specs):
            if spec is None:
                run += 1
            else:
                out[node] = run
        return out

    def depth_segments(self) -> Dict[int, int]:
        """Per neural node, its depth segment (``compute_segment_ids``, the shape-only specs' ``segment_id``)."""
        neural = {i: s is not None for i, s in zip(self.ids, self.specs)}
        depth = compute_segment_ids({i: set(p) for i, p in zip(self.ids, self.producers)}, neural)
        return {i: int(depth[i]) for i in self.ids if neural[i]}

    def spec_runs(self, count: int) -> Tuple[int, ...]:
        """[U1 W9] Per spec index ``0..count-1``, its run in the reference segmentation (``reference_runs``)."""
        reference, node_of = self.reference_runs(), self.node_of_spec()
        missing = [i for i in range(int(count)) if i not in node_of]
        if missing:
            raise ValueError(f"the program graph holds no node for spec(s) {missing[:8]}; it is not this plane's walk")
        return tuple(int(reference[node_of[i]]) for i in range(int(count)))


def spec_runs(walk: Any) -> Tuple[int, ...]:
    """[U1 W9] Per spec of a layout walk (``LayoutIRMapping`` after ``collect_layout_softcores``, ``IRMapping`` after
    ``map``), its IR-run segment: the host ops before its node in the walk's program order. Two specs share a builder
    segment (``partition_ir_graph``) iff they share a run, so a plane given these runs plans the builder's segments
    (``plan_program_passes(runs=)``). Reads only the walk's node records: no wire census needed."""
    neural: Mapping[int, bool] = walk._node_is_neural
    to_spec: Mapping[int, int] = walk._node_id_to_softcore_idx
    ids = sorted(int(i) for i in neural)
    order = ProgramGraph(ids=tuple(ids), specs=tuple(int(to_spec[i]) if neural[i] else None for i in ids),
                         producers=tuple(frozenset() for _ in ids))
    return order.spec_runs(len(walk.layout_softcores))


def require_segmentation_parity(graph: ProgramGraph) -> None:
    """The depth segments are admitted only where every neural-to-neural edge keeps its class in the reference
    segmentation: streamed inside one segment in both, or a segment boundary in both. Then every core reads the same
    producers through the same kind of boundary, at the same latency within its segment, and the latency-gated
    executor runs each core in its own window, so a core that moves computes what it computed. An edge that changes
    class changes the function (a raster streamed against a transcode at a host boundary): refused by name."""
    ref, depth = graph.reference_runs(), graph.depth_segments()
    changed = sorted((p, c) for c, sources in zip(graph.ids, graph.producers) if c in ref for p in sources
                     if p in ref and (ref[p] == ref[c]) != (depth[p] == depth[c]))
    if not changed:
        return
    first: Dict[int, int] = {}
    for node in graph.ids:
        if node in depth:
            first.setdefault(depth[node], node)
    moved = sorted(n for n in depth if ref[n] != ref[first[depth[n]]])
    kind = {True: "streamed", False: "a segment boundary"}
    edges = "; ".join(f"{p}->{c} ({kind[ref[p] == ref[c]]} in the reference, "
                      f"{kind[depth[p] == depth[c]]} re-segmented)" for p, c in changed[:8])
    raise ResidencyPlanError(
        f"residency_plan plans the shape-only plane's segments (a neural core's segment rises across a host "
        f"dependency); on this graph re-segmenting the reference segments (partition_ir_graph, which the identity "
        f"and U1 builders execute) moves cores {moved[:16]}, and {len(changed)} neural-to-neural edge(s) change "
        f"class: {edges}. A raster streamed inside a segment and a transcode at a segment boundary are different "
        f"functions, so the key is refused rather than deploying a different function")


def declared_pool(core_types: Sequence[LayoutHardCoreType]) -> Dict[Tuple[int, int], int]:
    """The declared physical cores per (axons, neurons) geometry."""
    pool: Dict[Tuple[int, int], int] = {}
    for t in core_types:
        key = (int(t.max_axons), int(t.max_neurons))
        pool[key] = pool.get(key, 0) + int(t.count)
    return pool


def check_pinned_fit(pass_core_types: Sequence[Sequence[CoreTiles]], pins: Sequence[FrozenSet[int]],
                     type_counts: Mapping[Any, int]) -> None:
    """Every pass's physical cores plus every other pass's pinned ones fit the declared pool, a tile on a core at
    least its size (an exact dominance matching); a violation refuses by name. The check is physical, not the
    planning budget C: under heterogeneous core types C is the type-blind 0.8 x pool (``effective_core_budget``), whose
    discount is the room a phase packed on the declared counts uses (t0_11 at x0.6: F 50 <= C 57, packed 60 of 72
    cores), and under one type the pool is C. A pass may pack into more or fewer tiles than its layers' unit count:
    that difference is recorded per pass, never refused."""
    pinned_by_type: List[Dict[Any, int]] = []
    for cores, pin in zip(pass_core_types, pins):
        tally: Dict[Any, int] = {}
        for index in pin:
            for t in cores[index]:
                tally[t] = tally.get(t, 0) + 1
        pinned_by_type.append(tally)
    for q, cores in enumerate(pass_core_types):
        need: Dict[Any, int] = {}
        for types in cores:
            for t in types:
                need[t] = need.get(t, 0) + 1
        for p, tally in enumerate(pinned_by_type):
            if p != q:
                for t, n in tally.items():
                    need[t] = need.get(t, 0) + n
        placed = dominance_matching(need, type_counts)
        if placed < sum(need.values()):
            raise PoolAdmissionError(
                f"residency_plan: pass {q} with the other passes' pinned cores needs {dict(sorted(need.items()))} "
                f"(axons, neurons) tiles; the declared pool {dict(sorted(type_counts.items()))} holds {placed} of "
                f"them, a tile on a core at least its size in both dimensions")


def fit_pins(pass_core_types: Sequence[Sequence[CoreTiles]], whole: Sequence[bool],
             type_counts: Mapping[Any, int]) -> Optional[Tuple[FrozenSet[int], ...]]:
    """[W03] The floor at a fit (r = 0, the pinned rule then pins every tile): every core of every pass, when the
    cores of the passes that load their own fit the declared pool side by side (a dominance matching), else None. A
    wholly resident pass (``whole``, U1's bank-clustered passes after the first) runs on cores already loaded."""
    need: Dict[Any, int] = {}
    for cores, resident in zip(pass_core_types, whole):
        if not resident:
            for types in cores:
                for t in types:
                    need[t] = need.get(t, 0) + 1
    if dominance_matching(need, type_counts) < sum(need.values()):
        return None
    return tuple(frozenset(range(len(cores))) for cores in pass_core_types)


def passes_pinned(packs: Sequence[Any], whole: Sequence[bool], core_types: Sequence[LayoutHardCoreType]) -> bool:
    """[fix_b] Whether the fit rule pins planned passes (``fit_pins`` over their packs' tiles, a fused core's members
    as its snapshot states them, ``LayoutCoreSnapshot.component_axons``; undetermined tiles: no)."""
    try:
        tiles = [[core_type_tiles(c.axons_per_core, c.neurons_per_core, core_types,
                                  getattr(c, "component_axons", None)) for c in p.used_core_snapshots or ()]
                 for p in packs]
    except ResidencyPlanError:
        return False
    return fit_pins(tiles, list(whole), declared_pool(core_types)) is not None


def unscheduled_pack_holds(pack: Any, core_types: Sequence[LayoutHardCoreType]) -> bool:
    """[fix_b] The fit rule on the shape-only plane: the program's unscheduled pack (one pack of every spec on the
    declared cores, ``pack_layout``) is feasible and its cores' tiles fit the declared pool. Where U1's passes do not fit
    side by side, that pack is the program with every core pinned, as the builder builds its single pool
    (``hybrid_build_fit``): a chip that holds the program programs it once (K27). Each plane judges its own pack."""
    return bool(getattr(pack, "feasible", False)) and passes_pinned([pack], [False], core_types)


def fit_rule_applies(residency_plan: Optional[Mapping[str, Any]]) -> bool:
    """[fix_c] Whether the fit rule judges a program planned under ``residency_plan``: U1 (no plan) and every declared
    plan (the floor, a step index) but the stateless reload baseline, which pins no core by definition (r = C, every
    core rewritten each period; plan section 9.3), on both planes."""
    return residency_plan is None or residency_declaration(residency_plan)[0] != STEP_STATELESS


def dominance_matching(need: Mapping[Tuple[int, int], int], pool: Mapping[Tuple[int, int], int]) -> int:
    """Most tiles of ``need`` placeable on distinct cores of ``pool``, a tile on a core at least its size in both
    dimensions: a max flow over (tile geometry -> dominating core geometry), exact."""
    kinds, types = sorted(need), sorted(pool)
    residual: Dict[Tuple[Any, Any], int] = {}
    for k in kinds:
        residual[("s", ("k", k))] = int(need[k])
        for t in types:
            if t[0] >= k[0] and t[1] >= k[1]:
                residual[(("k", k), ("t", t))] = int(need[k])
    for t in types:
        residual[(("t", t), "z")] = int(pool[t])
    flow = 0
    while True:
        parent: Dict[Any, Any] = {"s": None}
        frontier = ["s"]
        while frontier and "z" not in parent:
            node = frontier.pop(0)
            for (a, b), cap in residual.items():
                if cap > 0 and a == node and b not in parent:
                    parent[b] = a
                    frontier.append(b)
        if "z" not in parent:
            return flow
        path, node = [], "z"
        while parent[node] is not None:
            path.append((parent[node], node))
            node = parent[node]
        push = min(residual[e] for e in path)
        for a, b in path:
            residual[(a, b)] -= push
            residual[(b, a)] = residual.get((b, a), 0) + push
        flow += push
