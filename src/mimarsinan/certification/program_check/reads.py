"""[W15] The program elements the executors read beside the views: each executed unit's input-slice deliveries (kind
and applied delay, ``executed_reads.slice_deliveries``, the stage loop's own function), the bank blocks the HCM and
the TTFS executors upload (``bank_content``), the memos and span caches every executor reads (``stale_memo``: a memo
must equal its derivation), and the host ops' fixed function (``host_training``: no mode-dependent module in training
mode)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Set

import numpy as np
import torch

from mimarsinan.certification.program_check.declaration import SYNCHRONIZED_LIF, CheckDeclaration
from mimarsinan.certification.program_check.report import ConditionResult, Violation
from mimarsinan.certification.program_check.view import ProgramView, UnitView
from mimarsinan.mapping.packing.executed_reads import (
    SliceDelivery, bank_block, bank_read, slice_deliveries, stale_memos,
)
from mimarsinan.mapping.packing.pass_timing import carried_nodes
from mimarsinan.mapping.support.core_geometry import used_axons, used_neurons
from mimarsinan.mapping.support.schedule.pass_cut import VERBATIM
from mimarsinan.spiking.compute_boundary import publishes_boundary_train

#: The condition each memo steers: fused positions the encode (R4), carried outputs the carry store (R3), the
#: refcounts the state buffer's lifetimes (H9), the span caches every gather (tiles).
MEMO_CONDITION: Mapping[str, str] = {"_fused_position_cache": "R4", "_carried_output_cache": "R3",
                                     "_consumer_counts_cache": "H9", "_output_source_spans": "tiles",
                                     "_axon_source_spans": "tiles", "_source_counts_cache": "H9"}
#: Torch modules whose forward reads ``training``: a host op holding one in training mode is batch-dependent.
MODE_DEPENDENT = (torch.nn.modules.batchnorm._NormBase, torch.nn.modules.dropout._DropoutNd)


def cached_nodes(program: Any, decl: CheckDeclaration) -> Set[int]:
    """The host ops whose boundary leaves a cycle-accurate train (``publishes_boundary_train``)."""
    if not decl.caches_trains:
        return set()
    return {int(s.compute_op.id) for s in program.stages
            if s.kind == "compute" and s.compute_op is not None and publishes_boundary_train(s.compute_op)}


def unit_deliveries(program: Any, view: ProgramView, decl: CheckDeclaration) -> Dict[int, List[SliceDelivery]]:
    """Per executed unit, its slices as the stage loop delivers them: a carried raster where the run publishes one
    (VERBATIM, a streaming instance: the producer is an earlier pass of the unit's segment), the count under the
    count executor or in a level unit, a host op's cached train, else the encode; each at its applied delay."""
    publishes = decl.pass_transfer == VERBATIM and decl.semantics != SYNCHRONIZED_LIF
    cached = cached_nodes(program, decl)
    out: Dict[int, List[SliceDelivery]] = {}
    for unit in view.units:
        store = carried_nodes(program.stages, unit.program_stage) if publishes else set()
        deliveries = slice_deliveries(unit.stage, store, count_executor=decl.semantics == SYNCHRONIZED_LIF,
                                      cached=cached)
        out[unit.index] = deliveries if decl.replays else [SliceDelivery(d.index, d.node, d.kind, 0)
                                                           for d in deliveries]
    return out


def segment_carried(program: Any, unit: UnitView) -> Set[int]:
    """The nodes an earlier pass of ``unit``'s segment produces among its inputs (its carried wires)."""
    return carried_nodes(program.stages, unit.program_stage)


def _units(program: Any):
    for s, stage in enumerate(program.stages):
        for unit in [stage] + list(getattr(stage, "retimed_level_stages", None) or ()):
            yield s, unit


def check_memos(program: Any, results: Mapping[str, ConditionResult]) -> None:
    """``stale_memo``: every memo and span cache an executor reads holds its derivation (an empty memo is recomputed
    by its accessor). Both program memos survive pickle, so they are part of the deployed artifact."""
    def report(element: str, obj: Any, where: str) -> None:
        for attribute in stale_memos(element, obj):
            condition = MEMO_CONDITION[attribute]
            results[condition].add(Violation(condition, "stale_memo", where, f"{attribute} holds another value than "
                                             "its derivation; every executor that reads it runs the stale one"))
    for result in (results[c] for c in set(MEMO_CONDITION.values())):
        result.checked += 1
    report("program", program, "program")
    for s, unit in _units(program):
        where = f"stage {s} ({unit.name})"
        if unit.kind == "compute" and unit.compute_op is not None:
            report("host_op", unit.compute_op, where)
            continue
        hcm = unit.hard_core_mapping
        if hcm is None:
            continue
        report("core_mapping", hcm, where)
        for h, core in enumerate(hcm.cores):
            report("hard_core", core, f"{where} core {h}")


def check_banks(view: ProgramView, tiles: ConditionResult, counts: Dict[str, Any]) -> None:
    """``bank_content``: every core the HCM and the TTFS executors upload from a weight bank (``bank_read``) has a
    bank block equal to its grid block, so every executor (the value one and the backends read the grid) runs the
    same weights."""
    bank_cores = 0
    for unit in view.units:
        hcm = unit.stage.hard_core_mapping
        for h, cv in enumerate(unit.cores):
            read = bank_read(hcm, h)
            if read is None:
                continue
            bank_cores += 1
            tiles.checked += 1
            block = bank_block(hcm, read)
            want = (used_axons(cv.core, min_one=True), used_neurons(cv.core, min_one=True))
            if block is None:
                what = "is absent"
            elif block.shape != want:
                what = f"block {read.axons} x {read.neurons} has shape {block.shape}, the core uses {want}"
            elif not np.array_equal(np.asarray(block, dtype=np.float64),
                                    np.asarray(cv.core.get_core_matrix(), dtype=np.float64)[:want[0], :want[1]]):
                what = "block differs from the core's grid"
            else:
                continue
            tiles.add(Violation("tiles", "bank_content", f"unit {unit.index} ({unit.stage.name}) core {h}",
                                f"bank {read.bank_id} {what}; the HCM and the TTFS executors upload the bank"))
    counts["bank_cores"] = bank_cores


def training_module(params: Mapping[str, Any]) -> Optional[str]:
    """The first mode-dependent module (``MODE_DEPENDENT``) a host op holds in training mode, by type, or None."""
    for value in params.values():
        if isinstance(value, torch.nn.Module):
            for module in value.modules():
                if isinstance(module, MODE_DEPENDENT) and module.training:
                    return type(module).__name__
    return None
