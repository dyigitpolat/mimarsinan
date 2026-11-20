"""[thru:w4] The static check on a program of input slots (the batch pipeline, PL/thru/design.md (c) 'Executor check').

A run of such a program serves B inputs, and the reference executors run it once per input slot on that slot's stages
(``wavefront_stages.serves_slot``), input k in slot k mod B. So every condition of Theorem 1 is read on each slot's
projection (``slot_projection``: its stages in program order, a multi-pass deployment of the network): producer order,
the reads and the carried slices are keyed by slot and node, and the slot's outputs are the identity program's by
Theorem 1 where they hold (the projection lemma). Read on the whole program instead:

- H10, since a group's first pass loads the layout once for every slot (its first stage loads, the other slots' stages
  of that pass run resident on it), and every resident stage runs on that load (``residency.check_residency``);
- the program's memos, each derived from its whole stage list;
- two rules of H9 that are the slots' own: a stage reads only its own slot's values (``slot_scope``: a value another
  slot's stage produced, which a machine holding one value per node would hand it) and a stage's slot lies below B
  (``slot_range``: no run would execute it).

The report merges the projections condition by condition (each violation named with its slot), H10 and the two rules
added, and is finalized as one program's."""

from __future__ import annotations

import copy
import time
from dataclasses import replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Set

import numpy as np

from mimarsinan.certification.program_check.declaration import OUTSIDE_SCOPE, CheckDeclaration
from mimarsinan.certification.program_check.reads import MEMO_CONDITION
from mimarsinan.certification.program_check.report import (
    ConditionResult, ESTABLISHED, StaticCheckReport, Violation)
from mimarsinan.certification.program_check.residency import check_residency
from mimarsinan.certification.program_check.view import build_view
from mimarsinan.mapping.packing.executed_reads import stale_memos
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.support.bias_compensation import propagate_negative_shifts_to_hybrid
from mimarsinan.mapping.support.wavefront_stages import SCHEDULE_INPUT_SLOT, program_slots, serves_slot

#: The program's memos: each derives from the whole stage list, so it is read on the program and reset on a projection.
PROGRAM_MEMOS = ("_consumer_counts_cache", "_carried_output_cache")
#: Conditions read on the reference alone (the IR against the IR at unbounded geometry): one projection reads them.
REFERENCE_CONDITIONS = ("ir-tiling",)


def slot_projection(program: Any, slot: int) -> Any:
    """``program`` as the reference executors run input slot ``slot``: its stages that serve the slot, in program
    order, with its program memos reset (a shallow copy: the stages are the program's own, nothing is written)."""
    out = copy.copy(program)
    out.stages = [stage for stage in program.stages if serves_slot(stage, slot)]
    for memo in PROGRAM_MEMOS:
        if hasattr(out, memo):
            setattr(out, memo, None)
    return out


def _slot(stage: Any) -> Any:
    """The input slot a stage states (``InputSlot``), None: a stage of every slot."""
    return getattr(stage, SCHEDULE_INPUT_SLOT, None)


def _reads(stage: Any) -> Set[int]:
    if stage.kind == "neural":
        return {int(s.node_id) for s in stage.input_map}
    op = getattr(stage, "compute_op", None)
    return set() if op is None else {int(s.node_id) for s in np.asarray(op.input_sources).flatten()}


def _outputs(stage: Any) -> Set[int]:
    if stage.kind == "neural":
        return {int(s.node_id) for s in stage.output_map}
    op = getattr(stage, "compute_op", None)
    return set() if op is None else {int(op.id)}


def check_slot_rules(program: Any, slots: int, h9: ConditionResult) -> None:
    """``slot_range``: every stage that states an input slot states one of the run's ``slots`` and a slot below it.
    ``slot_scope``: no stage of slot b reads a node that, before it, only stages of other slots produced (a stage that
    names no slot serves every slot, so it is read for each)."""
    produced: Dict[Optional[int], Set[int]] = {}
    for index, stage in enumerate(program.stages):
        slot = _slot(stage)
        where = f"stage {index} ({stage.name})"
        if slot is not None:
            h9.checked += 1
            if int(slot.slots) != int(slots) or not 0 <= int(slot.slot) < int(slots):
                h9.add(Violation("H9", "slot_range", where, f"states input slot {slot.slot} of {slot.slots}, and a run "
                                 f"of this program serves {slots} inputs in slots 0 to {slots - 1}: no run executes "
                                 "it as stated"))
        reads = _reads(stage)
        for b in (range(int(slots)) if slot is None else (int(slot.slot),)):
            own = produced.get(b, set()) | produced.get(None, set())
            foreign = sorted(n for n in reads if n >= 0 and n not in own
                             and any(n in nodes for key, nodes in produced.items() if key not in (b, None)))
            if foreign:
                h9.add(Violation("H9", "slot_scope", where, f"input slot {b} reads node(s) {foreign[:6]}, which before "
                                 "it only other slots' stages produced; a stage reads only its own slot's values, and "
                                 "a machine holding one value per node would hand it another input's"))
        produced.setdefault(None if slot is None else int(slot.slot), set()).update(_outputs(stage))


def _merged(name: str, parts: List[ConditionResult], first: Optional[ConditionResult] = None) -> ConditionResult:
    """One condition over the projections: every violation named with its input slot (``first``'s, the whole
    program's, ahead of them), the elements checked summed, the status the parts share (else established, which the
    finalization reads against the violations)."""
    statuses = {part.status for part in parts}
    out = ConditionResult(name, statuses.pop() if len(statuses) == 1 else ESTABLISHED, note=parts[0].note)
    labelled = [("", first)] if first is not None else []
    labelled += [(f"input slot {b}: ", part) for b, part in enumerate(parts)]
    for label, part in labelled:
        out.checked += part.checked
        for violation in part.violations:
            out.add(replace(violation, where=label + violation.where))
        out.violation_count += part.violation_count - len(part.violations)
    return out


def check_slotted(program: Any, *, ir_graph: Any, declaration: CheckDeclaration, identity: Any = None,
                  law_inputs: Optional[Mapping[str, Any]] = None, compile_wall_s: Optional[float] = None,
                  compile_source: str = "built", unbounded_ir: Any = None,
                  check: Callable[..., StaticCheckReport]) -> StaticCheckReport:
    """``check_deployment`` on a program of input slots: ``check`` (``check.check_program``) on each slot's
    projection against one identity program, H10 and the program's memos on the whole program, and the slots' two
    rules of H9, merged into one report."""
    from mimarsinan.certification.program_check.check import _finalize

    start = time.perf_counter()
    decl = declaration
    slots = int(program_slots(program.stages))
    if identity is None:
        identity = build_identity_hybrid_mapping(ir_graph=ir_graph, retimed_level_stages=decl.windowed)
        propagate_negative_shifts_to_hybrid(ir_graph, identity)
    parts = [check(slot_projection(program, b), ir_graph=ir_graph, declaration=decl, identity=identity,
                   law_inputs=law_inputs, compile_wall_s=compile_wall_s, compile_source=compile_source,
                   unbounded_ir=unbounded_ir if b == 0 else None, residency=False) for b in range(slots)]
    rules = ConditionResult("H9", ESTABLISHED)
    check_slot_rules(program, slots, rules)  # the whole program's: they name the cause ahead of each slot's reading
    results = {name: (parts[0].results[name] if name in REFERENCE_CONDITIONS else
                      _merged(name, [part.results[name] for part in parts], rules if name == "H9" else None))
               for name in parts[0].results}
    results["H9"].note = (results["H9"].note + "; " if results["H9"].note else "") + (
        f"read per input slot on each of {slots} projections")
    results["H10"] = ConditionResult("H10", ESTABLISHED)
    check_residency(program, build_view(program, foreign=-6), decl, results["H10"])
    for memo in stale_memos("program", program):
        condition = MEMO_CONDITION[memo]
        results[condition].add(Violation(condition, "stale_memo", "program", f"{memo} holds another value than its "
                                         "derivation; every executor that reads it runs the stale one"))
    _finalize(decl, results)
    counts: Dict[str, Any] = dict(parts[0].counts)
    for key in ("units", "host_stages", "cores", "placements"):
        counts[key] = sum(int(part.counts.get(key, 0)) for part in parts)
    counts["input_slots"] = slots

    def forms(attribute: str) -> List[Violation]:
        seen: Dict[str, Violation] = {}
        for part in parts:
            for violation in getattr(part, attribute):
                seen.setdefault(violation.text(), violation)
        return list(seen.values())

    report = StaticCheckReport(semantics=decl.semantics, in_scope=decl.in_scope,
                               scope_note=OUTSIDE_SCOPE.get(decl.semantics, ""), results=results,
                               specs=parts[0].specs, unknown_forms=forms("unknown_forms"),
                               unread_forms=forms("unread_forms"),
                               compile_wall_s=compile_wall_s if compile_source == "built" else None, counts=counts,
                               dithered_encode=decl.dithered_encode, identity_facts=parts[0].identity_facts,
                               operand_banks=any(part.operand_banks for part in parts))
    report.proposition2_note = parts[0].proposition2_note
    report.wall_s = time.perf_counter() - start
    return report
