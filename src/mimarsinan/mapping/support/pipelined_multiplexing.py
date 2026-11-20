"""[U-F4] Pipelined position multiplexing: under position multiplexing, levels share passes as a wavefront.

A declared cost-only decision of the pass planner (``pipelined_multiplexing``). Consecutive levels of a segment form a
group while one instance of every copy key of the group packs on the declared cores (the packer's own feasibility); a
group's first pass loads that layout (every key's copy) and every later pass of the group runs on it. An instance runs
in the first pass after (i) every instance it reads and (ii) the previous instance of its own copy (one copy computes
one instance per pass); the fragments of one coalescing group run together. Every read is then of a value an EARLIER
pass produced (pass transparency), so a pass's cores are independent and the program computes what the one-level-per-
pass program computes. A level that holds owned cores, or whose keys do not pack even alone, keeps W02's own runs as
a group of its own. Undeclared, the planner composes W02's passes byte for byte.

[thru:w1] Layer copies (declared ``layer_replication``): the cores a group's layout leaves free take more copies of
its keys, one at a time, each to the key with the most instances per copy, while the layout packs on the declared
cores; an instance runs on its key's copy that is free earliest. ``wavefront_group_cores`` bounds the cores (physical
tiles) one copy of a group's keys may take, so a group closes earlier and leaves more cores to copies; a bound below
one copy of a level is refused by name. Undeclared or false, one copy per key, byte for byte.

[thru:w2] The batch pipeline (declared ``batch_pipeline_slots`` B): a run of the program serves B inputs. The instances
of input 1, then input 2, and so on share each wavefront group's passes, each in the first pass after its own input's
producers, on its key's copy that is free earliest (a copy computes one instance per pass, of whichever input), so a
group's start-up passes are paid once per run. [thru:w5] A pass outside every wavefront group (a W02 group's, or a
segment's outside the wavefront) holds by the group rule as a group of one pass (``slot_turns``): it serves the run's
inputs as slot stages under one load, input slot 0 to B-1 in consecutive passes (its one copy computes one instance per
pass), the first with the pass's own load and every other resident on it. The pass budget is counted per run. The chip
states how many inputs' carried values it holds at once (``carry_input_slots``): none stated, or fewer than B, refuses B
by name (``refuse_batch_slots``). Undeclared, one input per run, byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Collection, Dict, List, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.platform.mapping_structure import PositionMultiplexingError, refuse_batch_slots
from mimarsinan.mapping.support.position_multiplexing import CopyKey, copy_keys, multiplexed_segment_passes

__all__ = ["WavefrontGroup", "WavefrontPlan", "census_producers", "layout_specs", "pipelined_segment_passes",
           "refuse_layer_copies", "refuse_without_wire_census", "segment_producers", "slot_turns"]


@dataclass(frozen=True)
class WavefrontGroup:
    """One group of a segment: its levels, its passes ``[first_pass, end_pass)``, whether it is a wavefront (else
    W02's runs), its layout's copy keys in order and, per key, the segment position of its first instance. [thru:w1]
    Under a declared ``layer_replication``, per key its copies (the layout holds them key by key, then copy by copy);
    None undeclared (one copy per key)."""

    levels: Tuple[int, ...]
    first_pass: int
    end_pass: int
    pipelined: bool
    keys: Tuple[CopyKey, ...]
    template: Tuple[int, ...]
    copies: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class WavefrontPlan:
    """One segment's composed passes: per pass its specs and whether it programs its weights (a wavefront group's
    first pass, a W02 run's first pass), the groups, per spec (segment position: ``specs``) its pass and copy key.
    [thru:w1] Under a declared ``layer_replication``, per spec the copy of its key it runs on (0 outside every
    wavefront group); None undeclared."""

    pass_lists: Tuple[Tuple[LayoutSoftCoreSpec, ...], ...]
    run_heads: Tuple[bool, ...]
    groups: Tuple[WavefrontGroup, ...]
    pass_of: Tuple[int, ...]
    keys: Tuple[Optional[CopyKey], ...]
    feasible: bool
    specs: Tuple[LayoutSoftCoreSpec, ...] = ()
    copy_of: Optional[Tuple[int, ...]] = None
    #: [thru:w2] Under a declared ``batch_pipeline_slots``: B, and per pass the input slot of each of its entries
    #: (``pass_lists[q][k]`` serves input slot ``entry_slots[q][k]``; a spec appears once per input) and, under layer
    #: copies, the copy of its key each entry runs on; ``pass_of`` and ``copy_of`` then state input slot 0's. None
    #: undeclared (one input per run).
    slots: Optional[int] = None
    entry_slots: Optional[Tuple[Tuple[int, ...], ...]] = None
    entry_copies: Optional[Tuple[Tuple[int, ...], ...]] = None

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


def refuse_layer_copies(layer_replication: Optional[bool], wavefront_group_cores: Optional[int],
                        pipelined_multiplexing: Optional[bool] = None) -> None:
    """[thru:k, thru:w1] The layer copies' keys reach the one planner on both planes, which plans the copies of a
    declared ``layer_replication`` in every wavefront group; a declaration the planner cannot serve refuses by name: a
    group limit without ``layer_replication``, and ``layer_replication`` without the wavefront it copies."""
    if wavefront_group_cores is not None and not layer_replication:
        raise PositionMultiplexingError("wavefront_group_cores bounds the cores one copy of a wavefront group takes so "
                                        "that the rest hold copies: it needs layer_replication")
    if layer_replication and not pipelined_multiplexing:
        raise PositionMultiplexingError("layer_replication copies the banks of a wavefront group of "
                                        "pipelined_multiplexing; without pipelined_multiplexing there is no group")


def layout_specs(specs: Sequence[LayoutSoftCoreSpec], template: Sequence[int],
                 copies: Optional[Sequence[int]]) -> List[LayoutSoftCoreSpec]:
    """[thru:w1] A group's layout as the packer places it: per key its template spec, once per copy (key by key, then
    copy by copy), every copy past the first its own equal spec (the packer tracks specs by identity)."""
    if copies is None:
        return [specs[i] for i in template]
    return [specs[i] if c == 0 else replace(specs[i]) for i, n in zip(template, copies) for c in range(int(n))]


def _packs(specs: Sequence[LayoutSoftCoreSpec], core_types, allow_coalescing: bool, allow_splitting: bool,
           limit: Optional[int] = None) -> bool:
    """Whether ``specs`` pack on the declared cores; [thru:w1] with ``limit``, on at most that many of them (a fused
    core counts its physical tiles)."""
    if not specs:
        return False
    packed = pack_layout(softcores=list(specs), core_types=core_types, allow_neuron_splitting=allow_splitting,
                         allow_coalescing=allow_coalescing)
    return packed.feasible and (limit is None or sum(c.tiles for c in packed.used_core_snapshots or ()) <= limit)


def _level_plan(specs, keys, by_level, levels, core_types, allow_coalescing, allow_splitting, limit=None, label=""
                ) -> List[Tuple[bool, List[int]]]:
    """``(wavefront, levels)`` per group: consecutive levels while one copy of each of their keys packs together
    (within ``limit`` cores, where a group limit is declared)."""
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
        if limit is not None and not _packs([specs[first[k]] for k in mine], core_types, allow_coalescing,
                                            allow_splitting, limit):
            raise PositionMultiplexingError(
                f"layer_replication: wavefront_group_cores {int(limit)} is below one copy of level {level}'s banks in "
                f"{label}; declare a group limit of at least the cores one copy of every level takes")
        if current and not _packs([specs[first[k]] for k in current_keys + mine], core_types, allow_coalescing,
                                  allow_splitting, limit):
            plan.append((True, current))
            current, current_keys = [], []
        current.append(level)
        current_keys += mine
    if current:
        plan.append((True, current))
    return plan


def _replicate(specs, keys, members, order: Sequence[CopyKey], template: Sequence[int], core_types,
               allow_coalescing: bool, allow_splitting: bool) -> Tuple[int, ...]:
    """[thru:w1] Per key of ``order``, its copies: one each, then one more at a time to the open key with the most
    instances per copy (ties to the earlier key) while the layout packs on the declared cores; a key closes when it
    holds a copy per instance or its next copy does not pack."""
    count = {k: 0 for k in order}
    for i in members:
        count[keys[i]] += 1
    copies = {k: 1 for k in order}
    open_ = [k for k in order if count[k] > 1]
    while open_:
        best = open_[0]
        for k in open_[1:]:
            if count[k] * copies[best] > count[best] * copies[k]:
                best = k
        trial = {**copies, best: copies[best] + 1}
        if _packs(layout_specs(specs, template, [trial[k] for k in order]), core_types, allow_coalescing,
                  allow_splitting):
            copies = trial
            if copies[best] < count[best]:
                continue
        open_.remove(best)
    return tuple(copies[k] for k in order)


Entry = Tuple[int, int]   # [thru:w2] (input slot, member): one instance of one input


def slot_turns(heads: Sequence[bool], slots: int) -> List[Tuple[int, int, bool]]:
    """[thru:w5] The group rule for the passes outside every wavefront group, each as a group of one pass: under a
    declared ``batch_pipeline_slots`` B every planned pass (``heads``: per pass, whether it loads its weights) serves the
    run's B inputs in B consecutive passes, input slot 0 to B-1 (its one copy computes one instance per pass), slot 0's
    with the pass's own load and every other slot's resident on the layout it loads. Every producer of such a pass ran,
    for every slot, in an earlier pass (an earlier group's, or an earlier planned pass's turns), so each slot's reads are
    of its own earlier values. Per run pass, ``(planned pass, input slot, whether it loads)``."""
    return [(q, b, bool(head) and b == 0) for q, head in enumerate(heads) for b in range(int(slots))]


def _wavefront(members: Sequence[int], keys, producers, start: int, coalescing, label: str,
               copies: Optional[Dict[CopyKey, int]] = None, inputs: int = 1
               ) -> Tuple[Dict[Entry, int], Dict[Entry, int]]:
    """Per entry (input slot, member) its pass and its copy: the first pass after its producers of its own input inside
    the group and after its copy's previous instance, on its key's copy free earliest ([thru:w1] one copy per key
    undeclared), the members of one coalescing group together; a producer not yet composed refuses by name. [thru:w2]
    Input 0's members first, then input 1's, and so on (``inputs``: one undeclared); the copies are every input's."""
    member_set = set(members)
    units: Dict[object, List[int]] = {}
    for i in members:
        group = coalescing[i] if coalescing is not None and coalescing[i] is not None else ("own", i)
        units.setdefault(group, []).append(i)
    t: Dict[Entry, int] = {}
    copy_of: Dict[Entry, int] = {}
    free: Dict[CopyKey, List[int]] = {}   # per key, per copy: the first pass it is free in
    for b in range(int(inputs)):
        for unit in units.values():
            ready = start
            chosen: List[Tuple[int, int]] = []
            for i in unit:
                for p in producers[i]:
                    if p not in member_set:
                        continue
                    if (b, p) not in t:
                        raise PositionMultiplexingError(
                            f"pipelined_multiplexing: in {label}, spec {i} reads spec {p}, which is composed after "
                            "it; pass transparency needs every read from an earlier pass")
                    ready = max(ready, t[(b, p)] + 1)
                copy_free = free.setdefault(keys[i], [start] * (1 if copies is None else int(copies[keys[i]])))
                taken = {c for j, c in chosen if keys[j] == keys[i]}
                c = min((c for c in range(len(copy_free)) if c not in taken), key=lambda c: (copy_free[c], c),
                        default=0)
                ready = max(ready, copy_free[c])
                chosen.append((i, c))
            for i, c in chosen:
                t[(b, i)] = ready
                copy_of[(b, i)] = c
                free[keys[i]][c] = ready + 1
    return t, copy_of


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
    layer_replication: Optional[bool] = None,
    wavefront_group_cores: Optional[int] = None,
    batch_pipeline_slots: Optional[int] = None,
    carry_input_slots: Optional[int] = None,
) -> WavefrontPlan:
    """The segment's groups in level order, each a wavefront (or W02's runs), the passes numbered across groups.
    [thru:w1] Under ``layer_replication`` each wavefront group's free cores hold copies of its keys
    (``wavefront_group_cores``: the cores one copy of a group may take). [thru:w2] Under ``batch_pipeline_slots`` B
    the run serves B inputs: they share each wavefront group's passes, and [thru:w5] each pass of a W02 group serves
    them in turn under one load (``slot_turns``)."""
    refuse_layer_copies(layer_replication, wavefront_group_cores, True)
    refuse_batch_slots(batch_pipeline_slots, carry_input_slots, True)
    inputs = 1 if batch_pipeline_slots is None else int(batch_pipeline_slots)
    keys = copy_keys(specs, shares)
    level_of = [int(s.latency_tag or 0) for s in specs]
    levels = sorted(set(level_of))
    by_level = {lv: [i for i in range(len(specs)) if level_of[i] == lv] for lv in levels}
    pass_lists: List[List[Entry]] = []   # per pass, its entries (input slot, spec position)
    heads: List[bool] = []
    groups: List[WavefrontGroup] = []
    pass_of: Dict[Entry, int] = {}
    copy_of: Dict[Entry, int] = {}
    feasible = True
    for wavefront, group_levels in _level_plan(specs, keys, by_level, levels, core_types, allow_coalescing,
                                               allow_splitting, wavefront_group_cores if layer_replication else None,
                                               label):
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
            for q, b, head in slot_turns(mux.run_heads, inputs):  # [thru:w5] each W02 pass serves the slots in turn
                pass_lists.append([(b, local[id(s)]) for s in mux.pass_lists[q]])
                heads.append(head)
            groups.append(WavefrontGroup(tuple(group_levels), start, len(pass_lists), False, (), ()))
            continue
        order = list(dict.fromkeys(keys[i] for i in members))
        first = {k: next(i for i in members if keys[i] == k) for k in order}
        template = tuple(first[k] for k in order)
        copies = _replicate(specs, keys, members, order, template, core_types, allow_coalescing,
                            allow_splitting) if layer_replication else None
        t, copied = _wavefront(members, keys, producers, start, coalescing_group_ids, label,
                               None if copies is None else dict(zip(order, copies)), inputs)
        copy_of.update(copied)
        end = max(t.values()) + 1
        pass_lists.extend([] for _ in range(start, end))
        heads.extend(p == start for p in range(start, end))
        for entry in sorted(t, key=lambda e: (t[e], e)):   # a pass lists input 0's specs, then input 1's, ...
            pass_lists[t[entry]].append(entry)
        groups.append(WavefrontGroup(tuple(group_levels), start, end, True, tuple(order),  # type: ignore[arg-type]
                                     template, copies))
    for q, chunk in enumerate(pass_lists):
        for entry in chunk:
            pass_of[entry] = q
    for b in range(inputs):
        for c in range(len(specs)):
            for p in producers[c]:
                if pass_of[(b, p)] >= pass_of[(b, c)]:
                    raise PositionMultiplexingError(
                        f"pipelined_multiplexing: in {label}, spec {c} (pass {pass_of[(b, c)]}) reads spec {p} of pass "
                        f"{pass_of[(b, p)]}{'' if batch_pipeline_slots is None else f' (input slot {b})'}; pass "
                        "transparency needs every read from an earlier pass")
    if len(pass_lists) > int(max_schedule_passes):
        raise PositionMultiplexingError(
            f"position_multiplexing: {label} needs {len(pass_lists)} passes (pipelined: levels sharing passes as a "
            f"wavefront), above max_schedule_passes {int(max_schedule_passes)}; declare a larger pass budget")
    slotted = batch_pipeline_slots is not None
    return WavefrontPlan(
        pass_lists=tuple(tuple(specs[i] for _b, i in chunk) for chunk in pass_lists), run_heads=tuple(heads),
        groups=tuple(groups), pass_of=tuple(pass_of[(0, i)] for i in range(len(specs))), keys=tuple(keys),
        feasible=feasible, specs=tuple(specs),
        copy_of=tuple(copy_of.get((0, i), 0) for i in range(len(specs))) if layer_replication else None,
        slots=inputs if slotted else None,
        entry_slots=tuple(tuple(b for b, _i in chunk) for chunk in pass_lists) if slotted else None,
        entry_copies=tuple(tuple(copy_of.get(entry, 0) for entry in chunk) for chunk in pass_lists)
        if slotted and layer_replication else None)
