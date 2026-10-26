"""[W15] The executor conditions a built program fixes statically: semantics fidelity (H0, with the level-unit rules
of R5 and the level flag the executors read), the discipline-matched carry per wire (H5-collapse), per-event order
(H7), the accumulation and upload bounds (H8), dependency order (H9) and dependency-ordered walks (R6). Each reads the
program as every executor's stage loop reads it; the code clauses these rest on are ``report.ASSUMED``."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Set

import numpy as np

from mimarsinan.certification.program_check.declaration import (
    EXACT_INTEGER_REACH, PER_EVENT, REQUIRED_TRANSFER, SYNCHRONIZED_TTFS, VALUE_INTEGER, WINDOWED_LIF,
    CheckDeclaration,
)
from mimarsinan.certification.program_check.report import ConditionResult, Violation
from mimarsinan.certification.program_check.tiles import SweepFacts
from mimarsinan.certification.program_check.timing import IdentityTiming, walked_latencies
from mimarsinan.certification.program_check.view import K_CORE, K_INPUT, ProgramView
from mimarsinan.mapping.ir.dynamic_bank import stage_payload_sources
from mimarsinan.mapping.latency.core_graph import topological_order
from mimarsinan.mapping.packing.executed_reads import COUNT, REPLAYED, SliceDelivery
from mimarsinan.mapping.packing.pass_timing import carried_nodes
from mimarsinan.mapping.support.schedule.pass_cut import VERBATIM
from mimarsinan.models.nn.activations.integer_law import law_magnitude_bound


def check_order(program: Any, delta: ProgramView, facts: SweepFacts, h9: ConditionResult,
                r6: ConditionResult) -> None:
    """H9: every unit reads only nodes an earlier unit or host stage produced (its slices and, N2, its operand banks'
    payload writes), every node is produced once, a coalescing group sits in one unit. R6: every unit's core graph is acyclic and every core read is walked."""
    produced: Set[int] = {-2}
    made_by: Dict[int, str] = {}
    by_unit = {u.index: u for u in delta.units}
    host_of = {index: stage for index, _, stage in delta.hosts}
    for index in sorted(set(by_unit) | set(host_of)):
        if index in host_of:
            stage = host_of[index]
            reads = {int(s.node_id) for s in np.asarray(stage.compute_op.input_sources).flatten()} if \
                stage.compute_op is not None else set()
            outputs = {int(stage.compute_op.id)} if stage.compute_op is not None else set()
            where = f"host stage {stage.name}"
            payloads: List[int] = []
        else:
            unit = by_unit[index]
            reads = {n for n, _, _ in unit.slices}
            outputs = {int(s.node_id) for s in unit.stage.output_map}
            where = f"unit {index} ({unit.stage.name})"
            # [N2] a neural stage reads its bound banks' payload writes from the state buffer when it runs.
            payloads = stage_payload_sources(unit.stage, delta.bindings or {})
        h9.checked += 1
        missing = sorted(n for n in reads if n >= 0 and n not in produced)
        if missing:
            h9.add(Violation("H9", "reads_unproduced", where, f"reads node(s) {missing[:6]} before any stage "
                             "produces them"))
        unwritten = [n for n in payloads if n not in produced]
        if unwritten:
            h9.add(Violation("H9", "payload_unproduced", where, f"reads the payload write(s) of node(s) "
                             f"{unwritten[:6]} before any stage produces them"))
        for node in sorted(outputs):
            if node in made_by:
                h9.add(Violation("H9", "produced_twice", where, f"node {node} is produced again (first by "
                                 f"{made_by[node]}); a later store overwrites it"))
            made_by[node] = where
        produced |= outputs
    tail = sorted(int(s.node_id) for s in np.asarray(program.output_sources).flatten()
                  if int(s.node_id) >= 0 and int(s.node_id) not in produced)
    if tail:
        h9.add(Violation("H9", "output_unproduced", "program", f"the network reads node(s) {tail[:6]} no stage "
                         "produces"))
    groups: Dict[int, Set[int]] = {}
    for placed in delta.placements.values():
        for p in placed:
            gid = p.record.get("coalescing_group_id")
            if gid is not None:
                groups.setdefault(int(gid), set()).add(delta.units[p.unit].index)
    for gid, units in sorted(groups.items()):
        h9.checked += 1
        if len(units) > 1:
            h9.add(Violation("H9", "coalescing_group_cut", f"coalescing group {gid}",
                             f"its parts run in units {sorted(units)}; a group is never cut across passes"))
    for u, unit in enumerate(delta.units):
        sources: List[List[int]] = []
        read: Set[int] = set()
        for h, cv in enumerate(unit.cores):
            live = facts.cores[(u, h)].live & (cv.kinds == K_CORE)
            mine = sorted(set(cv.aux[live].tolist()))
            sources.append(mine)
            read |= set(mine)
        out = np.asarray(unit.stage.hard_core_mapping.output_sources).flatten()
        read |= {int(s.core_) for s in out if not (s.is_off_ or s.is_input_ or s.is_always_on_)}
        r6.checked += 1
        try:
            topological_order(sources)
        except ValueError:
            r6.add(Violation("R6", "core_cycle", f"unit {unit.index} ({unit.stage.name})",
                             "its cores read each other in a cycle; no walk is a dependency order"))
        walked = walked_latencies(unit.stage.hard_core_mapping)
        for h in sorted(read):
            if walked is not None and 0 <= h < len(walked) and walked[h] is None:
                r6.add(Violation("R6", "unwalked_core", f"unit {unit.index} ({unit.stage.name}) core {h}",
                                 "a core whose neurons are read has no latency; the walks skip it and its "
                                 "readers read zeros"))


def check_semantics(program: Any, ir_graph: Any, delta: ProgramView, iota: ProgramView, ident: IdentityTiming,
                    decl: CheckDeclaration, results: Dict[str, ConditionResult], facts: SweepFacts) -> None:
    """H0 and R5: the program's form matches the declared semantics. Under windowed LIF every unit holds one identity
    level unit's neurons and is a level unit exactly where the identity's is, and no identity stage that streams an
    in-segment edge is left unlevelled; under any other semantics no unit is a level unit."""
    h0, r5 = results["H0"], results["R5"]
    if decl.windowed:
        for u, unit in enumerate(iota.units):
            if unit.level:
                continue
            h0.checked += 1
            streamed = sum(int((facts.identity_cores[(u, h)].live & (cv.kinds == K_CORE)).any())
                           for h, cv in enumerate(unit.cores))
            if streamed:
                h0.add(Violation("H0", "unlevelled_windowed_stage", f"identity stage {unit.stage.name}",
                                 f"{streamed} core(s) read another core of the stage and it has no level units: it "
                                 "runs streamed under windowed LIF (both programs)"))
    for unit in delta.units:
        homes: Set[int] = {ident.unit[p.node] for cv in unit.cores for p in cv.placements if p.node in ident.unit}
        where = f"unit {unit.index} ({unit.stage.name})"
        flag = bool(getattr(unit.stage, "is_retimed_level", False))
        if flag != unit.level:   # the executors read the flag: raster replay, delays and cached trains turn on it
            condition = "R5" if decl.windowed else "H0"
            results[condition].add(Violation(condition, "level_flag", where, f"is_retimed_level is {flag} on "
                                             f"{'a level unit' if unit.level else 'a stage'}; the stage loop reads "
                                             f"its inputs as {'counts' if flag else 'rasters, trains and delays'}"))
        if not decl.windowed:
            h0.checked += 1
            if unit.level:
                h0.add(Violation("H0", "level_unit", where, f"a re-timed level unit under {decl.semantics}: its "
                                 "inputs are count re-encodes"))
            continue
        r5.checked += 1
        h0.checked += 1
        if len(homes) > 1 and any(iota.units[u].level for u in homes):
            rule = "mixed_levels" if unit.level else "streamed_windowed_pass"
            (r5 if unit.level else h0).add(Violation("R5" if unit.level else "H0", rule, where,
                                                     f"holds neurons of identity level units {sorted(homes)}"))
        elif len(homes) == 1 and unit.level != iota.units[next(iter(homes))].level:
            r5.add(Violation("R5", "unit_kind", where, f"{'is' if unit.level else 'is not'} a level unit where "
                             "the identity's unit of its neurons is" + (" not" if unit.level else "")))


def check_transfer(program: Any, delta: ProgramView, facts: SweepFacts, deliveries: Dict[int, List[SliceDelivery]],
                   decl: CheckDeclaration, h5c: ConditionResult) -> None:
    """H5-collapse: the run's transfer is the one its semantics owes a cut, and every carried wire a live row reads is
    delivered as owed: its raster under streamed LIF (VERBATIM), a count re-encode otherwise (never a raster); under
    windowed LIF no count is read shifted (a delay drops the window's last cycles)."""
    carried = sum(len(carried_nodes(program.stages, u.stage) & {n for n, _, _ in u.slices}) for u in delta.units)
    want = REQUIRED_TRANSFER.get(decl.semantics)
    if carried and want is not None and decl.pass_transfer != want:
        h5c.add(Violation("H5-collapse", "transfer", "program", f"{carried} carried slice(s) cross under "
                          f"{decl.pass_transfer}; {decl.semantics} owes a cut {want}"))
    for u, unit in enumerate(delta.units):
        wires = carried_nodes(program.stages, unit.program_stage)
        live = {int(k) for h, cv in enumerate(unit.cores)
                for k in cv.aux[facts.cores[(u, h)].live & (cv.kinds == K_INPUT)].tolist()}
        where = f"unit {unit.index} ({unit.stage.name})"
        for d in deliveries[unit.index]:
            if decl.semantics == WINDOWED_LIF and d.delay and d.kind != COUNT:
                h5c.add(Violation("H5-collapse", "shifted_count", f"{where} slice {d.index}", f"node {d.node} is "
                                  f"read {-d.delay} cycle(s) late, so its count loses the window's last cycles"))
            if d.node not in wires or d.index not in live or want is None:
                continue
            h5c.checked += 1
            if (d.kind == REPLAYED) != (want == VERBATIM):
                h5c.add(Violation("H5-collapse", "wire_delivery", f"{where} slice {d.index}", f"carried node "
                                  f"{d.node} is delivered as {d.kind}; {decl.semantics} owes "
                                  f"{'its raster' if want == VERBATIM else 'a count re-encode'}"))


def check_per_event(delta: ProgramView, decl: CheckDeclaration, h7: ConditionResult) -> None:
    """H7 on a declared per-event core: no co-residency, no split neuron, no fused fan-in, no level unit."""
    if decl.firing_granularity != PER_EVENT:
        return
    for unit in delta.units:
        if unit.level:
            h7.add(Violation("H7", "level_unit", f"unit {unit.index}", "a level unit re-encodes per-event counts"))
        for h, cv in enumerate(unit.cores):
            h7.checked += 1
            where = f"unit {unit.index} ({unit.stage.name}) core {h}"
            if len(cv.placements) > 1:
                h7.add(Violation("H7", "co_residency", where, f"{len(cv.placements)} soft cores share a per-event core"))
            if any(p.record.get("split_group_id") is not None for p in cv.placements):
                h7.add(Violation("H7", "split_neuron", where, "a split fragment re-thresholds a partial sum"))
            if getattr(cv.core, "fused_component_axons", None) is not None:
                h7.add(Violation("H7", "fused_fan_in", where, "a per-event neuron's fan-in is fused across cores"))


def check_accumulation(delta: ProgramView, facts: SweepFacts, decl: CheckDeclaration, h8: ConditionResult,
                       laws: ConditionResult, counts: Dict[str, Any]) -> None:
    """H8: every weight and bias integral (the charge unit); every weight and input code exact in each executor's
    upload dtype; and the largest charge any core forms below each declared executor's exact integer reach (spiking:
    ``T (sum |w| + |b|) + theta (1 + |V0|)``, under synchronized TTFS the lattice numerator ``2S (sum |w| + |b|)``;
    the value domain: ``sum |w| L_in``, and the law's ``|M0 acc + B|`` on the stamped law). An integer value program's
    core without a law runs the float path: ``law_free_core`` (laws)."""
    worst, worst_where = 0.0, ""
    upload = {executor: EXACT_INTEGER_REACH[dtype] for executor, dtype in decl.upload.items()}
    every = [("deployment", key, f) for key, f in facts.cores.items()]
    every += [("identity", key, f) for key, f in facts.identity_cores.items()]
    for prog, (u, h), core_facts in every:
        h8.checked += 1
        where = f"{prog} unit {u} core {h}"
        if decl.semantics == VALUE_INTEGER and core_facts.law is None and core_facts.placed:
            laws.checked += 1
            laws.add(Violation("laws", "law_free_core", where, "a core of an integer value program carries no "
                               "readout law, so it runs the float path"))
        if not (core_facts.integral_weights and core_facts.integral_bias):
            h8.add(Violation("H8", "off_lattice", where, "a weight or bias is not integral, so partial sums are not "
                             "exact in the charge unit"))
            continue
        if math.isnan(core_facts.magnitude):
            h8.add(Violation("H8", "input_range", where, "an input's range is undeclared (no producer law, no "
                             "armed boundary grid)"))
            continue
        for executor, reach in sorted(upload.items()):
            if max(core_facts.max_weight, core_facts.max_level) >= reach:
                h8.add(Violation("H8", "upload_precision", where, f"a weight or code of up to "
                                 f"{max(core_facts.max_weight, core_facts.max_level):.6g} is not exact in the "
                                 f"{executor} executor's upload dtype (2^{int(math.log2(reach))})"))
        if decl.value:
            bound = core_facts.magnitude
        elif decl.semantics == SYNCHRONIZED_TTFS:
            bound = 2.0 * decl.window * core_facts.magnitude
        else:
            bound = decl.window * core_facts.magnitude + abs(core_facts.theta) * (1.0 + abs(decl.membrane_init))
        if bound > worst:
            worst, worst_where = bound, where
        if decl.value and core_facts.law is not None:
            z = law_magnitude_bound(int(core_facts.magnitude), core_facts.law.m0, core_facts.law.bias,
                                    core_facts.law.adc_shift)
            if z >= EXACT_INTEGER_REACH["float64"]:
                h8.add(Violation("H8", "law_magnitude", where, f"|M0 acc + B| reaches {z} >= 2^53"))
    for executor, dtype in sorted(decl.accumulation.items()):
        reach = EXACT_INTEGER_REACH[dtype]
        counts[f"h8_margin_bits_{executor}"] = int(math.floor(math.log2(reach / max(worst, 1.0))))
        if worst >= reach:
            h8.add(Violation("H8", "accumulation_bound", worst_where, f"a charge of up to {worst:.6g} reaches the "
                             f"{executor} executor's exact {dtype} range 2^{int(math.log2(reach))}"))
    counts["h8_charge_bound"] = int(min(worst, 2 ** 62))
