"""[E6, N2] What the value executor reads of a program's operand banks, for the view: the program's binding table, per
hard core its operand blocks (the executor's own ``operand_blocks``; a fused core holding one is refused there by name,
``OperandPlacementError``, and flagged here), the core's static weights with every block cleared (``static_part``,
the placeholder never reaches an accumulator), and per core the payload's terms of the H8 bounds."""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np

from mimarsinan.chip_simulation.value_run.operand_blocks import OperandBlock, OperandPlacementError, operand_blocks
from mimarsinan.models.nn.activations.integer_law import grid_levels


def program_bindings(program: Any) -> Mapping[int, Any]:
    """The binding table the program's value executor reads (``ValueHybridCoreFlow``), empty without operand banks."""
    return getattr(program, "dynamic_bank_bindings", None) or {}


def _blocks_of(records: Sequence[Mapping[str, Any]], bindings: Mapping[int, Any]) -> List[OperandBlock]:
    return [OperandBlock(int(p["weight_bank_id"]), int(p["axon_offset"]), int(p["neuron_offset"]), int(p["axons"]),
                         int(p["neurons"]), tuple(p["bank_axon_range"]), tuple(p["bank_neuron_range"]))
            for p in records if p.get("weight_bank_id") is not None and int(p["weight_bank_id"]) in bindings]


def core_blocks(hcm: Any, bindings: Mapping[int, Any]) -> Tuple[List[List[OperandBlock]], List[int]]:
    """Per hard core of ``hcm`` its operand blocks, and the cores the executor refuses (a fused core holding one)."""
    if not bindings:
        return [[] for _ in hcm.cores], []
    try:
        return operand_blocks(hcm, bindings), []
    except OperandPlacementError:
        blocks = [_blocks_of(records, bindings) for records in hcm.soft_core_placements_per_hard_core]
        refused = [h for h, (core, mine) in enumerate(zip(hcm.cores, blocks))
                   if mine and getattr(core, "fused_component_axons", None) is not None]
        return blocks, refused


def static_part(weights: np.ndarray, blocks: Sequence[OperandBlock]) -> np.ndarray:
    """``weights`` (a core's used block) with every operand block cleared, a copy: the executor's static grid."""
    if not blocks:
        return weights
    out = np.array(weights, dtype=np.float64, copy=True)
    for block in blocks:
        out[block.axon_offset: block.axon_offset + block.axons,
            block.neuron_offset: block.neuron_offset + block.neurons] = 0.0
    return out


def payload_terms(blocks: Sequence[OperandBlock], bindings: Mapping[int, Any], row_levels: np.ndarray,
                  live: np.ndarray, columns: int) -> Tuple[np.ndarray, float, bool, np.ndarray]:
    """A core's payload terms of the H8 bounds: per column the reach of its payload cells (each live row's input
    level times the payload's largest code, ``grid_levels(payload_bits)``: the operand certificate's width, an assumed
    executor clause), the largest payload code, whether every payload is written as codes (a step declared), and the
    rows the payload blocks read (live reads, though their static weights are cleared)."""
    extra = np.zeros(int(columns), dtype=np.float64)
    rows_read = np.zeros(len(live), dtype=bool)
    top, coded = 0.0, True
    for block in blocks:
        binding = bindings[block.bank]
        level = float(grid_levels(int(binding.payload_bits)))
        rows = slice(block.axon_offset, block.axon_offset + block.axons)
        rows_read[rows] = True
        reach = float(np.asarray(row_levels[rows], dtype=np.float64)[np.asarray(live[rows], dtype=bool)].sum())
        extra[block.neuron_offset: block.neuron_offset + block.neurons] += reach * level
        top = max(top, level)
        coded = coded and binding.payload_step is not None
    return extra, top, coded, rows_read
