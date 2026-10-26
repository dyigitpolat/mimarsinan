"""[W15] Residency integrity (H10): a pass that executes on weights an earlier pass programmed holds, at every core
ordinal, exactly what that pass wrote (the chip does not reprogram it), and the executed residency plan's pinned
cores fit beside every pass on the declared pool, so no pass writes a pinned slot."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mimarsinan.certification.program_check.declaration import CheckDeclaration
from mimarsinan.certification.program_check.report import VACUOUS, ConditionResult, Violation
from mimarsinan.certification.program_check.view import ProgramView
from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
from mimarsinan.mapping.packing.residency_tiles import core_type_tiles
from mimarsinan.mapping.support.period_census import SCHEDULE_RESIDENT_CORES
from mimarsinan.mapping.support.residency_rules import declared_pool, dominance_matching


def _region(core, axons: int, neurons: int) -> Optional[Tuple[np.ndarray, Any, Any]]:
    grid = np.asarray(core.get_core_matrix())
    if grid.shape[0] < axons or grid.shape[1] < neurons:
        return None
    bias = getattr(core, "hardware_bias", None)
    return grid[:axons, :neurons], (None if bias is None else np.asarray(bias)[:neurons]), core.threshold


def _resident_units(delta: ProgramView, h10: ConditionResult) -> int:
    heads: Dict[Any, Any] = {}
    resident = 0
    for unit in delta.units:
        stage = unit.program_stage
        key = unit.segment
        if not getattr(stage, "schedule_weights_resident", False):
            if key is not None and not unit.level:
                heads[key] = unit
            continue
        resident += 1
        head = heads.get(key)
        where = f"unit {unit.index} ({unit.stage.name})"
        if head is None:
            h10.add(Violation("H10", "no_programming_pass", where, "marked weight-resident with no earlier pass of "
                              "its segment that programmed the cores"))
            continue
        for h, cv in enumerate(unit.cores):
            h10.checked += 1
            if h >= len(head.cores):
                h10.add(Violation("H10", "resident_content", f"{where} core {h}", f"pass {head.index} programmed "
                                  f"{len(head.cores)} cores; this ordinal was never written"))
                continue
            mine = _region(cv.core, cv.used_axons, cv.used_neurons)
            theirs = _region(head.cores[h].core, cv.used_axons, cv.used_neurons)
            same = mine is not None and theirs is not None and np.array_equal(mine[0], theirs[0]) and (
                (mine[1] is None and theirs[1] is None) or (mine[1] is not None and theirs[1] is not None
                                                            and np.array_equal(mine[1], theirs[1]))) and \
                float(mine[2] or 0) == float(theirs[2] or 0)
            if not same:
                h10.add(Violation("H10", "resident_content", f"{where} core {h}", f"executes weights, bias or "
                                  f"threshold other than what pass {head.index} wrote at that ordinal"))
    return resident


def _pinned_fit(program: Any, decl: CheckDeclaration, h10: ConditionResult) -> int:
    passes = [s for s in program.stages if s.kind == "neural" and getattr(s, SCHEDULE_RESIDENT_CORES, None) is not None]
    if not passes:
        return 0
    declared = core_types_from_declaration(list(decl.core_types))
    pool = declared_pool(declared)
    tiles: List[List[Tuple[Tuple[int, int], ...]]] = []
    pins: List[Sequence[int]] = []
    for stage in passes:
        cores = stage.hard_core_mapping.cores
        tiles.append([core_type_tiles(c.axons_per_core, c.neurons_per_core, declared,
                                      getattr(c, "fused_component_axons", None)) for c in cores])
        pins.append(sorted(int(i) for i in getattr(stage, SCHEDULE_RESIDENT_CORES)))
    for q, stage in enumerate(passes):
        h10.checked += 1
        bad = [i for i in pins[q] if not 0 <= i < len(tiles[q])]
        if bad:
            h10.add(Violation("H10", "pin_out_of_range", f"pass {stage.name}", f"pins cores {bad} it does not hold"))
            continue
        need: Dict[Any, int] = {}
        for core_tiles in tiles[q]:
            for t in core_tiles:
                need[t] = need.get(t, 0) + 1
        for p in range(len(passes)):
            if p != q:
                for i in pins[p]:
                    if 0 <= i < len(tiles[p]):
                        for t in tiles[p][i]:
                            need[t] = need.get(t, 0) + 1
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
        h10.note = f"{resident} weight-resident pass(es), {pinned} pass(es) with pinned cores"
