"""[W15] The static per-deployment check: the condition map, its named rules, and ``check_deployment``, which reads a
built program against its identity program (built from the same IR, at the declared geometry), and that IR against the
IR at unbounded geometry where it is given (E1 F1), and reports every condition of Theorem 1 and Proposition 2.

Input-independent and read-only: it compares what the executors read (executed weights, biases, thresholds, laws,
wiring, latencies, pass structure, slice deliveries, memos, residency marks), never runs an input."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from mimarsinan.certification.program_check import executor as E
from mimarsinan.certification.program_check import network as N
from mimarsinan.certification.program_check import reads as RD
from mimarsinan.certification.program_check import timing as TM
from mimarsinan.certification.program_check.declaration import (
    CASCADED_TTFS, OUTSIDE_SCOPE, STREAMED_LIF, SYNCHRONIZED_TTFS, WINDOWED_LIF, CheckDeclaration,
)
from mimarsinan.certification.program_check.forms import run_form_predicates, scan_forms
from mimarsinan.certification.program_check.report import (
    ESTABLISHED, FAILED, NOT_REQUIRED, VACUOUS, ConditionResult, ConditionSpec, StaticCheckReport,
)
from mimarsinan.certification.program_check.residency import check_residency
from mimarsinan.certification.program_check.tiles import sweep, wiring
from mimarsinan.certification.program_check.unbounded import check_ir_tiling
from mimarsinan.certification.program_check.unbounded.latency import IdentityFacts
from mimarsinan.certification.program_check.view import K_ON, ProgramView, build_view
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.support.bias_compensation import propagate_negative_shifts_to_hybrid

T1, P2 = "theorem1", "proposition2"
CONDITION_MAP: Tuple[ConditionSpec, ...] = (
    ConditionSpec("tiles", T1, "deployment", "the tiles reassemble to the identity program: every placed neuron's "
                  "executed fan-in summed per source, its bias and the stage and program I/O equal the identity's; "
                  "the placements are a bijection onto its neurons; every bank block is its core's grid; every memo "
                  "and span cache its derivation; host stages are its host ops, fixed functions"),
    ConditionSpec("H1-theta", T1, "program", "every core's stored singletons equal each resident's identity core's"),
    ConditionSpec("H1-latency", T1, "program", "the residents of a core share one identity (fused, ALAP) latency"),
    ConditionSpec("H5-collapse", T1, "program", "every carried wire delivers what the semantics owes a cut (the "
                  "raster under streamed LIF, a count re-encode otherwise), and no count is read shifted"),
    ConditionSpec("R1", T1, "program", "every in-unit edge keeps its fused latency difference as executed"),
    ConditionSpec("R2", T1, "program", "every carried slice is replayed at the fused delay L_c - 1 - L_p, and no "
                  "other slice is read late"),
    ConditionSpec("H5-skew", T1, "program", "R1 and R2"),
    ConditionSpec("H6", T1, "program", "a fused core's neurons carry their whole fan-in in one dot product"),
    ConditionSpec("H10", T1, "program", "a resident pass holds what its programming pass wrote; pinned cores fit "
                  "beside every pass on the declared pool"),
    ConditionSpec("R3", T1, "program", "a slice is carried exactly when the identity reads it inside its segment; no "
                  "identity segment boundary runs in-unit"),
    ConditionSpec("R4", T1, "program", "every slice a unit reads is of the kind the identity's unit reads it as, and "
                  "under a phase-dithered encode rotates by the identity program's channels at unbounded geometry"),
    ConditionSpec("R5", T1, "program", "under windowed LIF every unit is one identity level unit, of its kind and "
                  "flag"),
    ConditionSpec("laws", T1, "program", "every core of an integer value program carries a law, its stamped columns "
                  "the IR node's solved law"),
    ConditionSpec("H0", T1, "executor", "the program's form is the declared semantics' (level units exactly under "
                  "windowed LIF, every multi-depth windowed stage levelled, every level flag its unit's)"),
    ConditionSpec("H7", T1, "executor", "on a per-event core: no co-residency, split neuron, fused fan-in or level unit"),
    ConditionSpec("H8", T1, "executor", "integral weights and biases, every weight and code exact in each executor's "
                  "upload dtype, and every charge below each executor's exact integer reach"),
    ConditionSpec("H9", T1, "executor", "every unit reads only produced nodes, every node is produced once, no "
                  "coalescing group is cut, the refcount memos are fresh"),
    ConditionSpec("R6", T1, "executor", "every unit's core graph is acyclic and every core read is walked"),
    ConditionSpec("H2", P2, "network", "window coverage", "a latency gap above one leaves coverage to the data"),
    ConditionSpec("H3", P2, "network", "no terminal overshoot", "a signed fan-in leaves overshoot to the data"),
    ConditionSpec("H4", P2, "network", "chase completion", "burst debt drains depending on the arrival pattern"),
    ConditionSpec("H8-staircase", P2, "network", "every staircase on the integer charge (integral lattice)"),
    ConditionSpec("N-identity", P2, "network", "compaction drops only dead rows and columns"),
    ConditionSpec("seam-currency", P2, "network", "every core reading a host op's output re-encodes it at its trained "
                  "entry currency (calculus 11.2): coherent by construction at unit gauges",
                  "a non-unit seam's coherence depends on how its host op owns its scale domain, which the program "
                  "does not keep (the install certificate decides it)"),
    ConditionSpec("half-step", P2, "network", "synchronized TTFS: no half-step folded (P3, P4 vacuous)",
                  "a folded half-step's single fold and bias-lattice representation are the conversion's, not the "
                  "program's (P3, P4)"),
    ConditionSpec("ir-tiling", T1, "reference", "the reference is an IR at unbounded geometry, and the IR at the "
                  "declared geometry holds each of its neurons once, read through the identity builder's compaction, "
                  "with its fan-in (reassembled across tiles and coalescing groups), bias, threshold, scales, boundary "
                  "maps, grid, law and activation; under timed semantics both identity programs run it at one latency "
                  "in one stage; every host op the same function of the same neurons; the same program outputs; under "
                  "a dithered encode both identity programs key it on its channel (E1 F1; assumed where the IR at "
                  "unbounded geometry is not given)"),
)

#: Every named rule a condition's predicates raise (``test_condition_map`` holds each to a mutant or to ``UNMUTATED``).
RULES: Dict[str, Tuple[str, ...]] = {
    "tiles": ("fan_in", "coverage", "unplaced_column", "placement_extent", "foreign_node", "output_source",
              "output_width", "node_scales", "program_outputs", "host_op", "host_training", "identity_placement",
              "bank_content", "stale_memo", "fused_axons"),
    "H1-theta": ("threshold", "activation_scale", "parameter_scale", "input_activation_scale", "boundary_grid"),
    "H1-latency": ("co_resident_latencies",),
    "H5-collapse": ("transfer", "wire_delivery", "shifted_count"),
    "R1": ("pin_count", "pinned_latency"),
    "R2": ("delivery_delay", "delay_on_encoded_slice"),
    "H6": ("partial_sums",),
    "H10": ("resident_content", "no_programming_pass", "pinned_slot_overwritten", "pin_out_of_range",
            "undeclared_pool"),
    "R3": ("carry_scope", "mixed_segments", "boundary_edge_in_unit", "stale_memo"),
    "R4": ("phase_channel", "slice_kind", "multi_home_unit"),
    "R5": ("unit_kind", "mixed_levels", "level_flag"),
    "laws": ("stamped_law", "solved_law", "law_free_core"),
    "H0": ("unlevelled_windowed_stage", "streamed_windowed_pass", "level_unit", "level_flag"),
    "H7": ("co_residency", "split_neuron", "fused_fan_in", "level_unit"),
    "H8": ("off_lattice", "input_range", "accumulation_bound", "law_magnitude", "upload_precision"),
    "H9": ("reads_unproduced", "produced_twice", "output_unproduced", "coalescing_group_cut", "stale_memo"),
    "R6": ("core_cycle", "unwalked_core"),
    "H8-staircase": ("off_lattice",),
    "N-identity": ("dropped_live_row", "dropped_output", "dropped_read_column"),
    "ir-tiling": ("reference_geometry", "unkeyed_neuron", "duplicate_neuron", "partial_sum_group", "missing_neuron",
                  "extra_neuron", "fan_in", "bias", "threshold", "neuron_scale", "boundary_grid", "law", "activation",
                  "executed_latency", "timing_derivation", "host_op", "program_outputs", "phase_channel"),
}
#: The named rules no seeded program fault breaks, with the reason.
UNMUTATED: Dict[Tuple[str, str], str] = {
    ("tiles", "identity_placement"): "the identity builder places every IR node once by construction; the rule "
                                     "guards a change of that builder",
    ("H1-theta", "boundary_grid"): "the same comparison (CORE_SINGLETON_PROPERTIES) as the three scale singletons "
                                   "whose mutants bite; a grid object is a value-domain singleton",
    ("R3", "boundary_edge_in_unit"): "a live in-unit edge across identity segments needs a unit spanning a host op; "
                                     "no builder makes one, and a record edit fires mixed_segments first",
    ("R5", "mixed_levels"): "a level unit holding two identity level units is a builder change (merging level "
                            "units), not a program edit",
    ("laws", "solved_law"): "no deployment passes the calibrated ranges (law_inputs), so the re-solve runs only on "
                            "direct calls; the deployed check reads stamped against the IR's law",
}

#: Whether a semantics needs a condition (Lemmas 5, 6 and V; Proposition 2's cases), with the reason when not.
_SPIKING = lambda d: not d.value  # noqa: E731
REQUIRED: Dict[str, Tuple[Callable[[CheckDeclaration], bool], str]] = {
    "H1-latency": (lambda d: d.timed, "timing-free semantics: counts and values carry no spike timing"),
    "R1": (lambda d: d.timed, "timing-free semantics"), "R2": (lambda d: d.timed, "timing-free semantics"),
    "H5-skew": (lambda d: d.timed, "timing-free semantics"),
    "R3": (lambda d: d.timed, "a count or value carried across a cut equals its re-encode"),
    "R4": (_SPIKING, "the value domain has no spike encoder"),
    "R5": (lambda d: d.windowed, "only per-hop re-timed windowed LIF runs level units"),
    "laws": (lambda d: d.value, "spiking cores carry no readout law"),
    "H2": (lambda d: d.semantics == STREAMED_LIF, "holds inside a re-timed hop and under count or value executors"),
    "H3": (lambda d: d.semantics in (STREAMED_LIF, WINDOWED_LIF), "count and value executors are arrival-free"),
    "H4": (lambda d: d.semantics in (STREAMED_LIF, WINDOWED_LIF), "count and value executors are arrival-free"),
    "H8-staircase": (lambda d: not d.value and d.semantics != CASCADED_TTFS,
                     "no staircase (value domain) or Proposition 2 not claimed (cascaded TTFS)"),
    "seam-currency": (_SPIKING, "the value domain decodes no spike currency"),
    "half-step": (lambda d: d.semantics == SYNCHRONIZED_TTFS, "P3 and P4 are synchronized TTFS's preconditions"),
}
_VACUOUS_WHEN_EMPTY = ("H5-collapse", "R1", "R2", "H6", "laws", "R3", "R4", "R5", "seam-currency")


def _value_levels(iota: ProgramView) -> Callable[[Any, Any], np.ndarray]:
    """Per row, the input code magnitude the value domain reads: a producer core's law range, or the reading core's
    armed boundary grid; NaN where neither declares it; 0 on the always-on row (the bias is the law's)."""
    levels: Dict[int, float] = {}
    for unit in iota.units:
        for cv in unit.cores:
            law = getattr(cv.core, "readout_law", None)
            if law is None:
                continue
            for p in cv.placements:
                cols = slice(p.neuron_offset, p.neuron_offset + p.cols)
                levels[p.node] = float(max(np.abs(law.lo[cols]).max(initial=0), np.abs(law.hi[cols]).max(initial=0)))

    def rows(unit, cv) -> np.ndarray:
        grid = getattr(cv.core, "boundary_grid", None)
        fallback = float(grid.levels) if grid is not None and getattr(grid, "armed", False) else float("nan")
        nodes, inverse = np.unique((cv.codes >> 32) - 16, return_inverse=True)
        out = np.array([levels.get(int(n), fallback) for n in nodes], dtype=np.float64)[inverse]
        out[cv.kinds == K_ON] = 0.0
        return out

    return rows


def _finalize(decl: CheckDeclaration, results: Dict[str, ConditionResult]) -> None:
    for name, result in results.items():
        need, why = REQUIRED.get(name, (lambda d: True, ""))
        if not need(decl):
            extra = f"; {result.violation_count} violation(s) recorded" if result.violation_count else ""
            result.status, result.note = NOT_REQUIRED, why + extra
        elif result.violation_count:
            result.status = FAILED
        elif result.status == ESTABLISHED and result.checked == 0 and name in _VACUOUS_WHEN_EMPTY:
            result.status, result.note = VACUOUS, result.note or "the program holds no element it constrains"
    skew = results["H5-skew"]
    if skew.status != NOT_REQUIRED:
        parts = [results["R1"].status, results["R2"].status]
        skew.status = FAILED if FAILED in parts else (VACUOUS if all(p == VACUOUS for p in parts) else ESTABLISHED)


def check_deployment(program: Any, *, ir_graph: Any, declaration: CheckDeclaration, identity: Any = None,
                     law_inputs: Optional[Mapping[str, Any]] = None, compile_wall_s: Optional[float] = None,
                     compile_source: str = "built", unbounded_ir: Any = None) -> StaticCheckReport:
    """Every condition of the condition map on ``program`` against the identity program of ``ir_graph``, and, given
    ``unbounded_ir`` (the same network's IR at unbounded geometry), ``ir_graph`` against it per neuron (``ir-tiling``,
    which also reads what that identity program executes: ``report.identity_facts``, so a caller can run it later
    with ``check_ir_tiling``); a mapping loaded from a cache (``compile_source``) has no compile wall to price the
    check against."""
    start = time.perf_counter()
    decl = declaration
    results = {s.condition: ConditionResult(s.condition, ESTABLISHED) for s in CONDITION_MAP}
    forms = scan_forms(program, ir_graph)
    t_ident = time.perf_counter()
    if identity is None:
        identity = build_identity_hybrid_mapping(ir_graph=ir_graph, retimed_level_stages=decl.windowed)
        propagate_negative_shifts_to_hybrid(ir_graph, identity)
    t_ident = time.perf_counter() - t_ident
    delta, iota = build_view(program, foreign=-6), build_view(identity, foreign=-7)
    counts: Dict[str, Any] = {"compile_source": compile_source}
    levels = _value_levels(iota) if decl.value else (lambda unit, cv: np.ones(cv.used_axons, dtype=np.float64))
    facts = sweep(delta, iota, results, row_levels=levels, include_bias=not decl.value)
    wiring(delta, iota, program, identity, facts.width, results["tiles"])
    RD.check_banks(delta, results["tiles"], counts)
    RD.check_memos(program, results)
    deliveries, iota_deliveries = RD.unit_deliveries(program, delta, decl), RD.unit_deliveries(identity, iota, decl)
    ident = TM.identity_timing(iota, facts, latencies=decl.timed)
    TM.check_timing(program, delta, facts, ident, results, deliveries, timed=decl.timed)
    TM.check_slices(program, delta, iota, ident, deliveries, iota_deliveries, decl, results["R4"], ir_graph)
    E.check_order(program, delta, facts, results["H9"], results["R6"])
    E.check_semantics(program, ir_graph, delta, iota, ident, decl, results, facts)
    E.check_transfer(program, delta, facts, deliveries, decl, results["H5-collapse"])
    if decl.firing_granularity == "per_event":
        E.check_per_event(delta, decl, results["H7"])
    else:
        results["H7"].status, results["H7"].note = VACUOUS, "no core type declares per-event firing"
    E.check_accumulation(delta, facts, decl, results["H8"], results["laws"], counts)
    check_residency(program, delta, decl, results["H10"])
    N.check_staircase(facts, results["H8-staircase"])
    N.check_coverage(ident, results["H2"])
    N.check_overshoot(facts, results["H3"])
    N.chase_completion(results["H4"])
    N.check_identity_network(ir_graph, results["N-identity"])
    N.check_solved_laws(ir_graph, law_inputs, results["laws"])
    N.check_seams(ir_graph, identity, results["seam-currency"])
    N.check_half_step(decl, results["half-step"])
    run_form_predicates(program, delta, iota, decl, results)
    facts = IdentityFacts(latency=dict(ident.latency) if decl.timed else None, stage=dict(ident.segment),
                          channels=dict(getattr(identity, "node_phase_channels", None) or {}))
    t_tiling = time.perf_counter()
    check_ir_tiling(ir_graph, unbounded_ir, declaration=decl, result=results["ir-tiling"], identity_facts=facts)
    counts["ir_tiling_s"] = round(time.perf_counter() - t_tiling, 6)
    _finalize(decl, results)
    counts.update(units=len(delta.units), host_stages=len(delta.hosts),
                  cores=sum(len(u.cores) for u in delta.units), placements=sum(len(p) for p in delta.placements.values()),
                  identity_units=len(iota.units), identity_build_ms=int(1000 * t_ident))
    report = StaticCheckReport(semantics=decl.semantics, in_scope=decl.in_scope,
                               scope_note=OUTSIDE_SCOPE.get(decl.semantics, ""), results=results, specs=CONDITION_MAP,
                               unknown_forms=[v for v in forms if v.condition == "forms"],
                               unread_forms=[v for v in forms if v.condition == "forms_unread"],
                               compile_wall_s=compile_wall_s if compile_source == "built" else None, counts=counts,
                               dithered_encode=decl.dithered_encode, identity_facts=facts)
    report.wall_s = time.perf_counter() - start
    if decl.semantics == CASCADED_TTFS:
        report.proposition2_note = "not claimed: cascaded TTFS is in Theorem 1 only (its greedy schedule is lossy)"
    return report
