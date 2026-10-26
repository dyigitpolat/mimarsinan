"""[E6, N7] Every row pruning drops reads integer 0 (condition ``N-identity``, rule ``dropped_offset_row``).

Compaction drops a reader's rows the model still reads: a dropped row reads value 0, its code ``k`` on the reader's
input grid (``k = 0`` without an offset). The model's accumulator then holds ``k Σ W`` over the dropped rows and the
identity program's does not, so their laws differ in ``z`` by ``(M0 k - M_o) Σ W`` once the ADC passes both
accumulators unchanged. The predicate reads, on every reader of an offset grid with dropped weight
(``entry_offsets.dropped_row_magnitude``: a bank slice's masked rows, an owned matrix's recorded
``pruned_row_magnitude``): value 0 is a whole-step code of the grid (``whole_step_code``), its integer image
``M0 k - M_o`` is 0 on every column with dropped weight (``M_o`` by the law's own offset multiplier,
``offset_multipliers``), and the ADC is the identity there at levels at least the model's reach (the kept rows' code
ranges times their weights, plus ``|k|`` times the dropped weight). A reach a row's undeclared range leaves open is not
static (``N-identity`` then stays with the sampled run, named)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from mimarsinan.certification.program_check.report import ESTABLISHED, NOT_STATIC, VACUOUS, ConditionResult, Violation
from mimarsinan.certification.program_check.unbounded.keys import kept
from mimarsinan.mapping.export.entry_offsets import (
    core_entry_offset, dropped_row_magnitude, offset_multipliers, whole_step_code,
)
from mimarsinan.mapping.ir import ComputeOp, NeuralCore, IRSource

RULE = "dropped_offset_row"


def reader_offset(node: NeuralCore) -> float:
    """The offset of the grid a law core's data rows read: a lane law's input offset (the accumulator rows' grid),
    else the declared entry grid's (``core_entry_offset``), as both law solvers fold it."""
    law = getattr(node, "readout_law")
    if hasattr(law, "input_offset"):
        return float(np.asarray(law.input_offset, dtype=np.float64).reshape(-1)[0]) if law.columns else 0.0
    return core_entry_offset(node)


def _row_levels(node: NeuralCore, ir_graph: Any, nodes: Dict[int, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Per row: whether it is a data row the identity builder keeps (not off, not always-on), and its code range: a
    producer core's law ``max(|lo|, |hi|)`` at its column, a host value's armed entry grid levels, NaN if undeclared."""
    sources = np.asarray(node.input_sources, dtype=object).flatten()
    rows, _cols = kept(node)
    keep = np.zeros(len(sources), dtype=bool)
    keep[rows] = True
    grid = getattr(node, "boundary_grid", None)
    host = float(grid.levels) if grid is not None and getattr(grid, "armed", False) else float("nan")
    levels = np.full(len(sources), np.nan)
    for r, s in enumerate(sources):
        if not isinstance(s, IRSource) or s.is_off() or s.is_always_on():
            keep[r] = False
            continue
        producer = nodes.get(int(s.node_id))
        law = getattr(producer, "readout_law", None) if isinstance(producer, NeuralCore) else None
        if law is not None and int(s.index) < law.columns:
            levels[r] = float(max(abs(int(law.lo[int(s.index)])), abs(int(law.hi[int(s.index)]))))
        elif s.is_input() or isinstance(producer, ComputeOp):
            levels[r] = host
    return keep, levels


def check_dropped_offset_rows(ir_graph: Any, result: ConditionResult, counts: Dict[str, Any]) -> None:
    """``dropped_offset_row`` on every offset reader with dropped weight; the readers checked in ``counts``."""
    nodes = {int(n.id): n for n in getattr(ir_graph, "nodes", ())}
    checked = undecided = 0
    for node in nodes.values():
        law = getattr(node, "readout_law", None)
        if not isinstance(node, NeuralCore) or law is None or not law.columns or reader_offset(node) == 0.0:
            continue
        where, offset = f"IR node {node.id} ({node.name})", reader_offset(node)
        try:
            magnitude, _counted = dropped_row_magnitude(node, ir_graph)
        except ValueError as exc:
            result.add(Violation("N-identity", RULE, where, f"the dropped rows' weight is not read: {exc}"))
            continue
        dropped = np.asarray(magnitude) != 0
        if not dropped.any():
            continue
        checked += 1
        step = float(np.asarray(law.input_step, dtype=np.float64).reshape(-1)[0])
        k = whole_step_code(offset, step)
        if k is None:
            result.add(Violation("N-identity", RULE, where, f"value 0 is no code of its input grid (offset {offset:g}, "
                                 f"step {step:g}), so the rows pruning dropped in {int(dropped.sum())} column(s) add a "
                                 "nonzero term the model keeps"))
            continue
        residue = np.asarray(law.m0, dtype=np.int64) * int(k) - offset_multipliers(law, float(node.threshold), offset)
        bad = np.flatnonzero(dropped & (residue != 0))
        if bad.size:
            result.add(Violation("N-identity", RULE, where, f"the image of value 0 (code {k}) is M0 k - M_o = "
                                 f"{int(residue[bad[0]])} in column {int(bad[0])} ({bad.size} column(s) with dropped "
                                 "weight): the model's z departs by that times the dropped rows' weight"))
            continue
        keep, levels = _row_levels(node, ir_graph, nodes)
        weights = np.abs(np.asarray(node.get_core_matrix(ir_graph), dtype=np.float64))[keep]
        reach = (weights * levels[keep][:, None]).sum(axis=0) + abs(int(k)) * np.asarray(magnitude, dtype=np.float64)
        if np.isnan(reach[dropped]).any():
            undecided += 1
            continue
        adc_shift = np.asarray(law.adc_shift, dtype=np.int64).reshape(-1)
        adc_levels = np.asarray(law.adc_levels, dtype=np.float64).reshape(-1)
        inexact = np.flatnonzero(dropped & ((adc_shift != 0) | (adc_levels < reach)))
        if inexact.size:
            c = int(inexact[0])
            result.add(Violation("N-identity", RULE, where, f"column {c}: the ADC (step 2^{int(adc_shift[c])}, "
                                 f"{int(adc_levels[c])} levels) does not pass the accumulator the model reaches "
                                 f"({int(reach[c])}, the dropped term {k} Σ W included) unchanged"))
    counts["dropped_offset_readers"] = checked
    result.checked += checked
    if checked and result.status == VACUOUS:
        result.status, result.note = ESTABLISHED, ""
    if checked and not result.violation_count:
        result.note = (result.note + "; " if result.note else "") + (
            f"{checked} offset reader(s) with dropped rows: value 0 is a whole-step code whose image is 0 on every "
            f"column with dropped weight, and {checked - undecided} of them pass the dropped term through an identity "
            "ADC")
    if undecided and not result.violation_count:
        result.note += (f"; {undecided} read an undeclared code range: whether their ADC passes the dropped term "
                        "unchanged stays with the sampled run")
        result.status = NOT_STATIC
