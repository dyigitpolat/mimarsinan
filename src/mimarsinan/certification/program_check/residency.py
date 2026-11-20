"""[W15] Residency integrity (H10): a pass that executes on weights an earlier pass programmed holds, at every core
ordinal, exactly what that pass wrote (the chip does not reprogram it), and the executed residency plan's pinned
cores fit beside every pass on the declared pool, so no pass writes a pinned slot.

[E6, W03] A wholly resident pass (``schedule_weights_resident``: a bank-clustered or position-multiplexed run's later
pass) executes on the cores its programming pass loaded, ordinal by ordinal, so the fit counts it on its head's cores:
the cores a pass occupies, the other cores its head's group pins, and every other group's pins (W03's
``residency_rules.fit_pins`` pins every core of a program whose loading passes fit side by side). A core no declared
type decomposes (or decomposes in several ways) leaves the fit unread on the declared pool: ``undeclared_core``.

[fix round F1] Under per-hop retiming every pass executes as its retimed level units: a resident pass's level unit is
compared with the level unit of the same ordinal in the pass that programmed its cores.

[U-F4] A wavefront group's first pass loads the group's layout (every core it states, its dark cores included: loaded,
holding no placement) and each later pass of the group is wholly resident, each core run on the head's core of its
stated ordinal (``schedule_core_ordinals``): its content is compared with that core's, and its pins counted as that
core's."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mimarsinan.certification.program_check.declaration import CheckDeclaration
from mimarsinan.certification.program_check.report import VACUOUS, ConditionResult, Violation
from mimarsinan.certification.program_check.unit_forms.operands import static_part
from mimarsinan.certification.program_check.view import ProgramView
from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.mapping.packing.residency_tiles import core_type_tiles
from mimarsinan.mapping.support.neuron_thresholds import threshold_key
from mimarsinan.mapping.support.value_layouts import placement_value
from mimarsinan.mapping.support.period_census import SCHEDULE_RESIDENT_CORES
from mimarsinan.mapping.support.residency_rules import declared_pool, dominance_matching
from mimarsinan.mapping.support.wavefront_stages import SCHEDULE_CORE_ORDINALS, SCHEDULE_DARK_CORES, SCHEDULE_WAVEFRONT


def _region(core, axons: int, neurons: int, blocks=()) -> Optional[Tuple[np.ndarray, Any, Any]]:
    """The used grid (each operand block cleared: N2's payload is written per sample, never resident), bias and
    threshold."""
    grid = np.asarray(core.get_core_matrix())
    if grid.shape[0] < axons or grid.shape[1] < neurons:
        return None
    bias = placement_value(core, "hardware_bias", 0, neurons)  # [U-PCH3] by the core's declared layouts
    return (static_part(grid[:axons, :neurons], blocks), (None if bias is None else np.asarray(bias)),
            threshold_key(placement_value(core, "threshold", 0, neurons), neurons))  # each used neuron's register


def _resident_units(delta: ProgramView, h10: ConditionResult) -> int:
    """H10's resident content: a pass executes as one unit, or (per-hop retiming) as its retimed level units, so a
    programming pass heads its segment by its units in their order, and each unit of a resident pass is compared with
    the head's unit of the same ordinal [fix round F1: before, level units registered no head]."""
    heads: Dict[Any, List[Any]] = {}
    ordinals: Dict[int, int] = {}
    resident = 0
    for unit in delta.units:
        stage = unit.program_stage
        key = unit.segment
        ordinal = ordinals.get(unit.program_index, 0)
        ordinals[unit.program_index] = ordinal + 1
        if not getattr(stage, "schedule_weights_resident", False):
            if key is not None:
                if ordinal == 0:
                    heads[key] = []
                heads[key].append(unit)
            continue
        resident += 1
        head_units = heads.get(key)
        where = f"unit {unit.index} ({unit.stage.name})"
        if head_units is None:
            h10.add(Violation("H10", "no_programming_pass", where, "marked weight-resident with no earlier pass of "
                              "its segment that programmed the cores"))
            continue
        if ordinal >= len(head_units):
            h10.add(Violation("H10", "resident_content", where, f"level unit {ordinal} of its pass, while the pass "
                              f"that programmed its cores executes {len(head_units)} unit(s): this unit was never "
                              "written"))
            continue
        head = head_units[ordinal]
        at = _ordinals(stage, len(unit.cores))
        for h, cv in enumerate(unit.cores):
            h10.checked += 1
            if at[h] >= len(head.cores) or at[h] < 0:
                h10.add(Violation("H10", "resident_content", f"{where} core {h}", f"pass {head.index} programmed "
                                  f"{len(head.cores)} cores; ordinal {at[h]} was never written"))
                continue
            loaded = head.cores[at[h]]
            mine = _region(cv.core, cv.used_axons, cv.used_neurons, cv.blocks)
            theirs = _region(loaded.core, cv.used_axons, cv.used_neurons, loaded.blocks)
            same = mine is not None and theirs is not None and np.array_equal(mine[0], theirs[0]) and (
                (mine[1] is None and theirs[1] is None) or (mine[1] is not None and theirs[1] is not None
                                                            and np.array_equal(mine[1], theirs[1]))) and \
                mine[2] == theirs[2]
            if not same:
                h10.add(Violation("H10", "resident_content", f"{where} core {h}", f"executes weights, bias or "
                                  f"threshold other than what pass {head.index} wrote at that ordinal"))
    return resident


def _ordinals(stage: Any, count: int) -> List[int]:
    """Per core of ``stage``, the ordinal of its head's core it runs on (stated by a wavefront pass, else its own)."""
    stated = getattr(stage, SCHEDULE_CORE_ORDINALS, None)
    return list(range(count)) if stated is None else [int(o) for o in stated] + [-1] * (count - len(stated))


def _wavefront_groups(program: Any, h10: ConditionResult) -> int:
    """Each wavefront head loads exactly the layout it states, its dark cores hold no placement, and every later pass
    of its group is wholly resident on it with one head ordinal per core; returns the heads checked."""
    passes = [s for s in program.stages if s.kind == "neural"]
    heads = 0
    for q, stage in enumerate(passes):
        head = getattr(stage, SCHEDULE_WAVEFRONT, None)
        if head is None:
            continue
        heads += 1
        h10.checked += 1
        cores = stage.hard_core_mapping.cores
        records = list(stage.hard_core_mapping.soft_core_placements_per_hard_core or [])
        dark = sorted(int(i) for i in (getattr(stage, SCHEDULE_DARK_CORES, None) or ()))
        if int(head.layout_cores) != len(cores) or getattr(stage, "schedule_weights_resident", False):
            h10.add(Violation("H10", "wavefront_layout", f"pass {stage.name}", f"states a layout of "
                              f"{head.layout_cores} cores and loads {len(cores)} (resident: "
                              f"{bool(getattr(stage, 'schedule_weights_resident', False))})"))
        if any(i >= len(cores) or (i < len(records) and records[i]) for i in dark):
            h10.add(Violation("H10", "wavefront_dark_core", f"pass {stage.name}", f"dark cores {dark} hold a "
                              "placement or lie outside its layout"))
        group = passes[q + 1:q + int(head.passes)]
        for later in group:
            stated = getattr(later, SCHEDULE_CORE_ORDINALS, None)
            if (len(group) != int(head.passes) - 1 or not getattr(later, "schedule_weights_resident", False)
                    or later.schedule_segment_index != stage.schedule_segment_index or stated is None
                    or len(stated) != len(later.hard_core_mapping.cores)):
                h10.add(Violation("H10", "wavefront_layout", f"pass {later.name}", f"a later pass of the wavefront "
                                  f"group headed by {stage.name} that is not wholly resident on its layout"))
    return heads


def _heads(passes: Sequence[Any]) -> List[int]:
    """Per pass, the pass whose cores it executes on: itself, or (wholly resident) its segment's latest programming
    pass (a first pass marked resident heads itself; ``no_programming_pass`` names it)."""
    last: Dict[Any, int] = {}
    out: List[int] = []
    for q, stage in enumerate(passes):
        segment = getattr(stage, "schedule_segment_index", None)
        if getattr(stage, "schedule_weights_resident", False) and segment in last:
            out.append(last[segment])
            continue
        last[segment] = q
        out.append(q)
    return out


def _tally(need: Dict[Any, int], tiles) -> None:
    for t in tiles:
        need[t] = need.get(t, 0) + 1


def _pinned_fit(program: Any, decl: CheckDeclaration, h10: ConditionResult) -> int:
    passes = [s for s in program.stages if s.kind == "neural" and getattr(s, SCHEDULE_RESIDENT_CORES, None) is not None]
    if not passes:
        return 0
    declared = core_types_from_declaration(list(decl.core_types))
    pool = declared_pool(declared)
    tiles: List[List[Tuple[Tuple[int, int], ...]]] = []
    for stage in passes:
        mine = []
        for h, c in enumerate(stage.hard_core_mapping.cores):
            try:
                mine.append(core_type_tiles(c.axons_per_core, c.neurons_per_core, declared,
                                            getattr(c, "fused_component_axons", None)))
            except ResidencyPlanError:
                h10.add(Violation("H10", "undeclared_core", f"pass {stage.name} core {h}",
                                  f"a {c.axons_per_core} x {c.neurons_per_core} core decomposes into no declared core "
                                  f"type (or into several ways): the pinned fit is not read on the declared pool"))
                return len(passes)
        tiles.append(mine)
    pins = [sorted(int(i) for i in getattr(stage, SCHEDULE_RESIDENT_CORES)) for stage in passes]
    heads = _heads(passes)
    # Per group (a head and its wholly resident passes), its pinned ordinals and their tiles on the head's cores.
    pinned: Dict[int, Dict[int, Tuple[Tuple[int, int], ...]]] = {}
    at = [_ordinals(stage, len(tiles[q])) for q, stage in enumerate(passes)]
    for q, head in enumerate(heads):
        for i in pins[q]:
            if 0 <= i < len(tiles[q]):
                o = at[q][i] if head != q else i
                pinned.setdefault(head, {}).setdefault(o, tiles[head][o] if 0 <= o < len(tiles[head]) else tiles[q][i])
    for q, stage in enumerate(passes):
        h10.checked += 1
        bad = [i for i in pins[q] if not 0 <= i < len(tiles[q])]
        if bad:
            h10.add(Violation("H10", "pin_out_of_range", f"pass {stage.name}", f"pins cores {bad} it does not hold"))
            continue
        need: Dict[Any, int] = {}
        for core_tiles in tiles[q]:
            _tally(need, core_tiles)
        mine = set(at[q]) if heads[q] != q else set(range(len(tiles[q])))
        for head, ordinals in pinned.items():
            for i, core_tiles in ordinals.items():
                if head != heads[q] or i not in mine:   # the head's other pinned cores stay loaded
                    _tally(need, core_tiles)
        placed = dominance_matching(need, pool)
        if placed < sum(need.values()):
            h10.add(Violation("H10", "pinned_slot_overwritten", f"pass {stage.name}",
                              f"its cores and the other passes' pinned ones need {sum(need.values())} tiles of a pool "
                              f"of {sum(pool.values())} that holds {placed}: a core of this pass is written into a "
                              f"pinned slot"))
    return len(passes)


def check_residency(program: Any, delta: ProgramView, decl: CheckDeclaration, h10: ConditionResult) -> None:
    """H10 over the resident passes and the pinned cores; pins without a declared pool fail by name
    (``undeclared_pool``), never a vacuous pass. The fit asks whether every pass fits beside the other passes' pinned
    cores (a dominance matching exists); it does not read which slot the allocator writes."""
    groups = _wavefront_groups(program, h10)
    resident = _resident_units(delta, h10)
    pinned = _pinned_fit(program, decl, h10) if decl.core_types else 0
    if not decl.core_types:
        pins = [s.name for s in program.stages if s.kind == "neural" and getattr(s, SCHEDULE_RESIDENT_CORES, None)]
        if pins:
            h10.add(Violation("H10", "undeclared_pool", f"pass {pins[0]}", f"{len(pins)} pass(es) pin cores, but the "
                              "declaration names no core type, so no fit can be read"))
            pinned = len(pins)
    if not resident and not pinned:
        h10.status, h10.note = VACUOUS, "no pass executes on resident weights and no core is pinned"
    else:
        h10.note = f"{resident} weight-resident pass(es), {pinned} pass(es) with pinned cores" + (
            f", {groups} wavefront group(s)" if groups else "")
