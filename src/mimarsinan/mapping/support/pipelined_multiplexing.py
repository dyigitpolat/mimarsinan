"""[U-F4] Pipelined position multiplexing: under position multiplexing, levels share passes as a wavefront.

A declared cost-only decision of the pass planner (``pipelined_multiplexing``). Consecutive levels of a segment form a
group while one instance of every copy key of the group packs on the declared cores (the packer's own feasibility); a
group's first pass loads that layout (every key's copy) and every later pass of the group runs on it. An instance runs
in the first pass after (i) every instance it reads and (ii) the previous instance of its own copy (one copy computes
one instance per pass); the fragments of one coalescing group run together. Every read is then of a value an EARLIER
pass produced (pass transparency), so a pass's cores are independent and the program computes what the one-level-per-
pass program computes. A level that holds owned cores, or whose keys do not pack even alone, keeps W02's own runs as
a group of its own. Undeclared, the planner composes W02's passes byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Dict, List, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.platform.mapping_structure import PositionMultiplexingError
from mimarsinan.mapping.support.position_multiplexing import CopyKey, copy_keys, multiplexed_segment_passes

__all__ = ["WavefrontGroup", "WavefrontPlan", "census_producers", "pipelined_segment_passes", "refuse_layer_copies",
           "refuse_without_wire_census", "segment_producers"]


@dataclass(frozen=True)
class WavefrontGroup:
    """One group of a segment: its levels, its passes ``[first_pass, end_pass)``, whether it is a wavefront (else
    W02's runs), its layout's copy keys in order and, per key, the segment position of its first instance."""

    levels: Tuple[int, ...]
    first_pass: int
    end_pass: int
    pipelined: bool
    keys: Tuple[CopyKey, ...]
    template: Tuple[int, ...]


@dataclass(frozen=True)
class WavefrontPlan:
    """One segment's composed passes: per pass its specs and whether it programs its weights (a wavefront group's
    first pass, a W02 run's first pass), the groups, per spec (segment position: ``specs``) its pass and copy key."""

    pass_lists: Tuple[Tuple[LayoutSoftCoreSpec, ...], ...]
    run_heads: Tuple[bool, ...]
    groups: Tuple[WavefrontGroup, ...]
    pass_of: Tuple[int, ...]
    keys: Tuple[Optional[CopyKey], ...]
    feasible: bool
    specs: Tuple[LayoutSoftCoreSpec, ...] = ()

    def group_of_pass(self) -> Tuple[int, ...]:
        return tuple(g for g, group in enumerate(self.groups) for _ in range(group.first_pass, group.end_pass))


def segment_producers(producers: Optional[Sequence[Collection[int]]], members: Sequence[int]
                      ) -> Optional[List[frozenset]]:
    """The producers of a segment's specs (``members``: their program indices) as segment positions, the reads of
    specs outside the segment dropped (an earlier segment's values arrive as segment input); None without any."""
    if producers is None:
        return None
    position = {int(i): k for k, i in enumerate(members)}
    return [frozenset(position[int(p)] for p in producers[int(i)] if int(p) in position) for i in members]


def census_producers(census, count: int) -> Optional[List[frozenset]]:
    """Per spec (program index), the specs it reads on the layout walk's wire census (``pair_wires``); None without
    one."""
    if census is None:
        return None
    reads: List[set] = [set() for _ in range(int(count))]
    for producer, consumer in census.pair_wires:
        reads[int(consumer)].add(int(producer))
    return [frozenset(r) for r in reads]


def refuse_without_wire_census(caller: str, pipelined: Optional[bool]) -> None:
    """A caller that holds no wire census cannot order a wavefront: a declared ``pipelined_multiplexing`` refuses
    there by name."""
    if pipelined:
        raise PositionMultiplexingError(
            f"pipelined_multiplexing orders each pass by the instances it reads, and {caller} holds no wire census "
            "of the layout walk to read them from")


def refuse_layer_copies(layer_replication: Optional[bool], wavefront_group_cores: Optional[int]) -> None:
    """[thru:k] The layer copies' keys reach the one planner on both planes; until its copies are planned (item W1 of
    PL/thru/design.md), a declared ``layer_replication`` true refuses by name, as does a group limit without it."""
    if wavefront_group_cores is not None and not layer_replication:
        raise PositionMultiplexingError("wavefront_group_cores bounds the cores one copy of a wavefront group takes so "
                                        "that the rest hold copies: it needs layer_replication")
    if layer_replication:
        raise PositionMultiplexingError("layer_replication: copies of a wavefront group's banks on its free cores are "
                                        "not planned on this tree yet, so the declaration is refused")


def _packs(specs: Sequence[LayoutSoftCoreSpec], core_types, allow_coalescing: bool, allow_splitting: bool) -> bool:
    return bool(specs) and pack_layout(softcores=list(specs), core_types=core_types,
                                       allow_neuron_splitting=allow_splitting,
                                       allow_coalescing=allow_coalescing).feasible


def _level_plan(specs, keys, by_level, levels, core_types, allow_coalescing, allow_splitting
                ) -> List[Tuple[bool, List[int]]]:
    """``(wavefront, levels)`` per group: consecutive levels while one copy of each of their keys packs together."""
    first: Dict[CopyKey, int] = {}
    for i, key in enumerate(keys):
        if key is not None and key not in first:
            first[key] = i

    def level_keys(level) -> List[CopyKey]:
        return list(dict.fromkeys(keys[i] for i in by_level[level] if keys[i] is not None))

    plan: List[Tuple[bool, List[int]]] = []
    current: List[int] = []
    current_keys: List[CopyKey] = []
    for level in levels:
        mine = level_keys(level)
        alone = all(keys[i] is not None for i in by_level[level]) and _packs(
            [specs[first[k]] for k in mine], core_types, allow_coalescing, allow_splitting)
        if not alone:
            if current:
                plan.append((True, current))
            plan.append((False, [level]))
            current, current_keys = [], []
            continue
        if current and not _packs([specs[first[k]] for k in current_keys + mine], core_types, allow_coalescing,
                                  allow_splitting):
            plan.append((True, current))
            current, current_keys = [], []
        current.append(level)
        current_keys += mine
    if current:
        plan.append((True, current))
    return plan


def _wavefront(members: Sequence[int], keys, producers, start: int, coalescing, label: str) -> Dict[int, int]:
    """Per member its pass: the first after its producers inside the group and its copy's previous instance, the
    members of one coalescing group together; a producer not yet composed refuses by name."""
    member_set = set(members)
    units: Dict[object, List[int]] = {}
    for i in members:
        group = coalescing[i] if coalescing is not None and coalescing[i] is not None else ("own", i)
        units.setdefault(group, []).append(i)
    t: Dict[int, int] = {}
    last: Dict[CopyKey, int] = {}
    for unit in units.values():
        ready = start
        for i in unit:
            for p in producers[i]:
                if p not in member_set:
                    continue
                if p not in t:
                    raise PositionMultiplexingError(
                        f"pipelined_multiplexing: in {label}, spec {i} reads spec {p}, which is composed after it; "
                        "pass transparency needs every read from an earlier pass")
                ready = max(ready, t[p] + 1)
            if keys[i] in last:
                ready = max(ready, last[keys[i]] + 1)
        for i in unit:
            t[i] = ready
            last[keys[i]] = ready
    return t


def pipelined_segment_passes(
    specs: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    *,
    producers: Sequence[frozenset],
    budget: int,
    allow_coalescing: bool,
    allow_splitting: bool,
    max_schedule_passes: int,
    shares: Optional[Sequence[bool]] = None,
    coalescing_group_ids: Optional[Sequence[Optional[int]]] = None,
    label: str = "segment",
) -> WavefrontPlan:
    """The segment's groups in level order, each a wavefront (or W02's runs), the passes numbered across groups."""
    keys = copy_keys(specs, shares)
    level_of = [int(s.latency_tag or 0) for s in specs]
    levels = sorted(set(level_of))
    by_level = {lv: [i for i in range(len(specs)) if level_of[i] == lv] for lv in levels}
    pass_lists: List[List[int]] = []
    heads: List[bool] = []
    groups: List[WavefrontGroup] = []
    pass_of: Dict[int, int] = {}
    feasible = True
    for wavefront, group_levels in _level_plan(specs, keys, by_level, levels, core_types, allow_coalescing,
                                               allow_splitting):
        start = len(pass_lists)
        members = [i for lv in group_levels for i in by_level[lv]]
        if not wavefront:
            sub = [specs[i] for i in members]
            mux = multiplexed_segment_passes(
                sub, core_types, budget=budget, allow_coalescing=allow_coalescing, allow_splitting=allow_splitting,
                max_schedule_passes=1 << 62, shares=None if shares is None else [shares[i] for i in members],
                coalescing_group_ids=None if coalescing_group_ids is None
                else [coalescing_group_ids[i] for i in members], label=label)
            feasible = feasible and mux.feasible
            local = {id(specs[i]): i for i in members}
            for chunk, head in zip(mux.pass_lists, mux.run_heads):
                pass_lists.append([local[id(s)] for s in chunk])
                heads.append(bool(head))
            groups.append(WavefrontGroup(tuple(group_levels), start, len(pass_lists), False, (), ()))
            continue
        t = _wavefront(members, keys, producers, start, coalescing_group_ids, label)
        end = max(t.values()) + 1
        pass_lists.extend([] for _ in range(start, end))
        heads.extend(p == start for p in range(start, end))
        for i in sorted(members, key=lambda j: (t[j], j)):
            pass_lists[t[i]].append(i)
        order = list(dict.fromkeys(keys[i] for i in members))
        first = {k: next(i for i in members if keys[i] == k) for k in order}
        groups.append(WavefrontGroup(tuple(group_levels), start, end, True, tuple(order),  # type: ignore[arg-type]
                                     tuple(first[k] for k in order)))
    for q, chunk in enumerate(pass_lists):
        for i in chunk:
            pass_of[i] = q
    for c in range(len(specs)):
        for p in producers[c]:
            if pass_of[p] >= pass_of[c]:
                raise PositionMultiplexingError(
                    f"pipelined_multiplexing: in {label}, spec {c} (pass {pass_of[c]}) reads spec {p} of pass "
                    f"{pass_of[p]}; pass transparency needs every read from an earlier pass")
    if len(pass_lists) > int(max_schedule_passes):
        raise PositionMultiplexingError(
            f"position_multiplexing: {label} needs {len(pass_lists)} passes (pipelined: levels sharing passes as a "
            f"wavefront), above max_schedule_passes {int(max_schedule_passes)}; declare a larger pass budget")
    return WavefrontPlan(
        pass_lists=tuple(tuple(specs[i] for i in chunk) for chunk in pass_lists), run_heads=tuple(heads),
        groups=tuple(groups), pass_of=tuple(pass_of[i] for i in range(len(specs))), keys=tuple(keys),
        feasible=feasible, specs=tuple(specs))
