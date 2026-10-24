"""Operand rows of fused add lanes (WI-5e): one row per column, zero crossbar weights, before any bias row.

The row carries the operand's element as an ordinary axon source, so packing, neuron splitting, coalescing, pass
cuts and carries move it like every input; the crossbar adds nothing for it (its weights are 0) and the lane reads
its code. A weight bank gains its rows once, shared by every position core of the bank.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.ir.gather_plan import invalidate_gather_plan
from mimarsinan.mapping.ir.lane_law import OPERAND_AXONS
from mimarsinan.mapping.lanes.ir.refusal import LaneFusionError

_OFF = IRSource(node_id=-1, index=0)
_MASKS = ("pruned_row_mask", "pruned_col_mask", "pre_pruning_row_mask", "pre_pruning_col_mask")


def _refuse_unfusable(core: NeuralCore) -> None:
    masks = [name for name in _MASKS if getattr(core, name, None) is not None]
    if masks:
        raise LaneFusionError(f"core {core.name!r} is pruned ({masks[0]}): its compaction masks carry no operand "
                              "rows, so a fused add lane on a pruned core is refused")
    if core.psum_group_id is not None or core.coalescing_group_id is not None:
        raise LaneFusionError(f"core {core.name!r} is a partial-sum or coalescing part: a fused add needs the "
                              "whole accumulator in one core")


def _bias_row(sources) -> bool:
    return len(sources) > 0 and sources[-1].is_always_on()


def _owned(core: NeuralCore, columns: Dict[int, IRSource], attribute: str) -> int:
    matrix = np.asarray(core.core_matrix)
    sources = list(core.input_sources.flatten())
    position = len(sources) - 1 if _bias_row(sources) else len(sources)
    ordered = sorted(columns)
    zeros = np.zeros((len(ordered), matrix.shape[1]), dtype=matrix.dtype)
    core.core_matrix = np.concatenate([matrix[:position], zeros, matrix[position:]], axis=0)
    core.input_sources = np.array(sources[:position] + [columns[j] for j in ordered] + sources[position:], dtype=object)
    rows = np.full(matrix.shape[1], -1, dtype=np.int64)
    rows[ordered] = position + np.arange(len(ordered))
    setattr(core, attribute, rows)
    setattr(core, OPERAND_AXONS, len(ordered))
    return len(ordered)


def _banked(ir_graph: IRGraph, bank_id: int, operands: Dict[int, Dict[int, IRSource]], attribute: str) -> int:
    bank = ir_graph.weight_banks[bank_id]
    cores: List[NeuralCore] = [n for n in ir_graph.nodes if isinstance(n, NeuralCore) and n.weight_bank_id == bank_id]
    rows, columns = bank.core_matrix.shape
    outside = [c.name for c in cores if len(operands.get(int(c.id), {})) != c.get_output_count()]
    if outside:
        raise LaneFusionError(f"weight bank {bank_id} is shared by cores {outside[:3]} outside its add lane: one "
                              "operand row set per bank")
    if getattr(bank, "pre_pruning_snapshot", None) is not None:
        raise LaneFusionError(f"weight bank {bank_id} is pruned: a fused add lane on a pruned core is refused")
    biased = {_bias_row(list(c.input_sources.flatten())) for c in cores}
    if len(biased) != 1 or any(len(c.input_sources.flatten()) != rows for c in cores):
        raise LaneFusionError(f"the cores of weight bank {bank_id} do not share one row layout")
    position = rows - 1 if biased.pop() and bank.hardware_bias is None else rows
    matrix = np.asarray(bank.core_matrix)
    zeros = np.zeros((columns, columns), dtype=matrix.dtype)
    bank.core_matrix = np.concatenate([matrix[:position], zeros, matrix[position:]], axis=0)
    for core in cores:
        start = int(core.weight_row_slice[0]) if core.weight_row_slice is not None else 0
        sources = list(core.input_sources.flatten())
        lane = [_OFF] * columns
        for column, source in operands[int(core.id)].items():
            lane[start + int(column)] = source
        core.input_sources = np.array(sources[:position] + lane + sources[position:], dtype=object)
        setattr(core, attribute, position + start + np.arange(core.get_output_count(), dtype=np.int64))
        setattr(core, OPERAND_AXONS, columns)  # the bank's whole operand block: every column's row, others' off
    return columns


def insert_operand_rows(ir_graph: IRGraph, operands: Dict[int, Dict[int, IRSource]], attribute: str) -> int:
    """Give every column of ``operands`` (``{core id: {column: source}}``) its operand row; the rows inserted."""
    nodes = {int(n.id): n for n in ir_graph.nodes}
    banks = set()
    inserted = 0
    for core_id, columns in operands.items():
        core = nodes[core_id]
        assert isinstance(core, NeuralCore)
        _refuse_unfusable(core)
        if core.weight_bank_id is not None:
            banks.add(int(core.weight_bank_id))
        else:
            inserted += _owned(core, columns, attribute)
    for bank_id in sorted(banks):
        inserted += _banked(ir_graph, bank_id, operands, attribute)
    for core_id in operands:
        invalidate_gather_plan(nodes[core_id])
    return inserted
