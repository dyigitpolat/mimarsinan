"""The int64 golden (E283): every core's law in exact int64, host ops by the same functions in fp64 on grid values.

[thru:w2] A program of input slots (the batch pipeline) runs by the value executor's one rule
(``hybrid_execution.input_slot_runs``): once per input slot on that slot's rows, each stage of another slot left out,
and a read of a row no earlier stage of its input wrote refused by name (``require_inputs_produced``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import input_slot_runs, scatter_rows
from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import require_inputs_produced
from mimarsinan.chip_simulation.value_run.host_ops import execute_host_op
from mimarsinan.chip_simulation.value_run.operand_blocks import (
    bank_payload,
    cells_beyond_width,
    operand_blocks,
    stage_banks,
)
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.ir.deployment_dtype import VALUE_ARITHMETIC
from mimarsinan.mapping.platform.mapping_structure import BatchPipelineError
from mimarsinan.mapping.support.wavefront_stages import serves_slot
from mimarsinan.models.nn.activations.integer_law import apply_law_int64

_RAW_INPUT = -2


class GoldenGridError(ToolchainRefusal, ValueError):
    """A golden read off its declared grid: a host value with no boundary grid, or two grids on one wire."""


def _snap(values: np.ndarray, grid) -> np.ndarray:
    """The entry snap's codes (the boundary SSOT's rounding, offset and code range) in int64."""
    return grid.entry_codes(values)


def _law_kwargs(law) -> dict:
    return dict(m0=law.m0, bias=law.bias, shift=law.shift, lo=law.lo, hi=law.hi,
                adc_shift=law.adc_shift, adc_levels=law.adc_levels)


def _row_steps(hard_core) -> np.ndarray:
    steps = np.full(int(hard_core.axons_per_core), np.nan)
    law = hard_core.readout_law
    for p in hard_core.matrix_placements:
        steps[p.axon_offset: p.axon_offset + p.axons] = law.input_step[p.neuron_offset]
    rows = np.asarray(getattr(law, "operand_row", ()), dtype=np.int64)
    if rows.size:  # a lane law's operand rows read their operand's grid (WI-5e)
        steps[rows[rows >= 0]] = law.operand_step[rows >= 0]
    return steps


@dataclass
class GoldenTrace:
    """What one golden run read and wrote beside its output codes: the payload codes of every bound bank
    (bank: (batch, axons, neurons)), each host op's fp64 values (op id: (batch, width)) and each core's codes."""

    payloads: Dict[int, np.ndarray] = field(default_factory=dict)
    host_values: Dict[int, np.ndarray] = field(default_factory=dict)
    core_codes: Dict[int, np.ndarray] = field(default_factory=dict)


def _row_offsets(hard_core) -> np.ndarray:
    """Per row, the offset of the grid the row reads (a lane law's; 0 elsewhere): a producer must write it."""
    offsets = np.zeros(int(hard_core.axons_per_core))
    law = hard_core.readout_law
    if not hasattr(law, "operand_row"):
        return offsets
    for p in hard_core.matrix_placements:
        offsets[p.axon_offset: p.axon_offset + p.axons] = law.input_offset[p.neuron_offset]
    rows = np.asarray(law.operand_row, dtype=np.int64)
    offsets[rows[rows >= 0]] = law.operand_offset[rows >= 0]
    return offsets


def _operand_codes(law, q: np.ndarray) -> tuple:
    """A lane law's operand term: each column's operand-row code (0 where it has none)."""
    rows = np.asarray(getattr(law, "operand_row", ()), dtype=np.int64)
    if not rows.size or not (rows >= 0).any():
        return ()
    codes = np.where(rows >= 0, q[:, np.maximum(rows, 0)], 0)
    return ((codes, law.operand_m),)


def output_offsets(law) -> np.ndarray:
    """The value of output code 0 is minus this, per column (0 off a lane law)."""
    return np.asarray(getattr(law, "output_offset", np.zeros(law.columns)), dtype=VALUE_ARITHMETIC.core_np)


class _Segment:
    """One neural stage evaluated on demand, core by core, from its segment input."""

    def __init__(self, hcm, seg_values: np.ndarray, seg_codes: np.ndarray, seg_steps: np.ndarray,
                 seg_offsets: "np.ndarray | None" = None, payloads: Optional[Dict[int, np.ndarray]] = None,
                 bindings=None) -> None:
        self.hcm = hcm
        self.seg_values, self.seg_codes, self.seg_steps = seg_values, seg_codes, seg_steps
        self.seg_offsets = np.zeros(seg_steps.shape) if seg_offsets is None else seg_offsets
        self.done: Dict[int, np.ndarray] = {}
        self.payloads = payloads or {}
        self.operand = operand_blocks(hcm, bindings) if bindings else None

    def core(self, index: int) -> np.ndarray:
        hit = self.done.get(index)
        if hit is not None:
            return hit
        hc = self.hcm.cores[index]
        law = getattr(hc, "readout_law", None)
        if law is None:
            raise GoldenGridError(f"hard core {index} carries no readout law")
        steps, row_offsets = _row_steps(hc), _row_offsets(hc)
        grid = getattr(hc, "boundary_grid", None)
        batch = self.seg_values.shape[0]
        q = np.zeros((batch, int(hc.axons_per_core)), dtype=np.int64)
        for row, source in enumerate(hc.axon_sources):
            if source.is_off_ or source.is_always_on_:
                continue
            if source.is_input_:
                position = int(source.neuron_)
                if self.seg_steps[position] == self.seg_steps[position]:
                    if self.seg_steps[position] != steps[row] or self.seg_offsets[position] != row_offsets[row]:
                        raise GoldenGridError(f"hard core {index} row {row}: producer and reader grids differ")
                    q[:, row] = self.seg_codes[:, position]
                elif grid is not None and grid.armed and float(grid.step) == steps[row]:
                    q[:, row] = _snap(self.seg_values[:, position], grid)
                else:
                    raise GoldenGridError(f"hard core {index} row {row}: host value without its boundary grid")
            else:
                q[:, row] = self.core(int(source.core_))[:, int(source.neuron_)]
        weight = np.asarray(hc.get_core_matrix(), dtype=VALUE_ARITHMETIC.core_np)
        blocks = self.operand[index] if self.operand is not None else []
        if blocks:
            weight = _per_sample_weight(weight, blocks, self.payloads, batch)
            acc = np.einsum("ba,ban->bn", q, weight)
        else:
            if not np.array_equal(weight, np.round(weight)):
                raise GoldenGridError(f"hard core {index}: weights are not integers")
            acc = q @ weight.astype(np.int64)
        codes = apply_law_int64(acc, **_law_kwargs(law), operands=_operand_codes(law, q))
        self.done[index] = codes
        return codes


def _per_sample_weight(static: np.ndarray, blocks, payloads: Dict[int, np.ndarray], batch: int) -> np.ndarray:
    """(batch, axons, neurons) int64: the core's integer grid with every operand block's payload codes written."""
    weight = static.copy()
    for block in blocks:
        weight[block.axon_offset: block.axon_offset + block.axons,
               block.neuron_offset: block.neuron_offset + block.neurons] = 0.0
    if not np.array_equal(weight, np.round(weight)):
        raise GoldenGridError("a core's static weights are not integers")
    out = np.repeat(weight.astype(np.int64)[None], batch, axis=0)
    for block in blocks:
        rows, cols = slice(*block.bank_axons), slice(*block.bank_neurons)
        out[:, block.axon_offset: block.axon_offset + block.axons,
            block.neuron_offset: block.neuron_offset + block.neurons] = payloads[block.bank][:, rows, cols]
    return out


def _stage_payloads(stage, values, bindings, trace: Optional[GoldenTrace]) -> Dict[int, np.ndarray]:
    """{bank: int64 (batch, axons, neurons)}: each bound bank's codes recovered from its write's fp64 values on its
    payload grid (an off-grid value refuses by name, ``OffGridInputError``), each within its ``payload_bits``."""
    out: Dict[int, np.ndarray] = {}
    for bank in stage_banks(stage, bindings):
        binding = bindings[bank]
        if binding.payload_step is None:
            raise GoldenGridError(f"operand bank {bank} has no payload grid: the golden reads law programs only")
        codes = bank_payload(torch.as_tensor(values[int(binding.source_node_id)]), binding)
        out[bank] = codes.numpy().astype(np.int64)
        beyond = cells_beyond_width(out[bank], binding.payload_bits)
        if beyond:
            raise GoldenGridError(f"operand bank {bank}: {beyond} payload cells lie beyond its payload_bits "
                                  f"{binding.payload_bits} grid, which bounds its accumulator and sizes its ADC")
        if trace is not None:
            trace.payloads[bank] = out[bank]
    return out


def _neural_stage(stage, values, codes, steps, offsets: "Dict[int, np.ndarray] | None" = None,
                  bindings=None, trace: Optional[GoldenTrace] = None) -> None:
    offsets = {} if offsets is None else offsets
    hcm = stage.hard_core_mapping
    width = max((s.offset + s.size for s in stage.input_map), default=0)
    batch = next(iter(values.values())).shape[0]
    seg_values = np.zeros((batch, width))
    seg_codes = np.zeros((batch, width), dtype=np.int64)
    seg_steps = np.full(width, np.nan)
    seg_offsets = np.zeros(width)
    for s in stage.input_map:
        seg_values[:, s.offset: s.offset + s.size] = values[int(s.node_id)][:, : s.size]
        if int(s.node_id) in codes:
            seg_codes[:, s.offset: s.offset + s.size] = codes[int(s.node_id)][:, : s.size]
            seg_steps[s.offset: s.offset + s.size] = steps[int(s.node_id)][: s.size]
            if int(s.node_id) in offsets:
                seg_offsets[s.offset: s.offset + s.size] = offsets[int(s.node_id)][: s.size]
    payloads = _stage_payloads(stage, values, bindings, trace) if bindings else None
    segment = _Segment(hcm, seg_values, seg_codes, seg_steps, seg_offsets, payloads, bindings)
    out_sources = list(np.asarray(hcm.output_sources).flatten())
    out_codes = np.zeros((batch, len(out_sources)), dtype=np.int64)
    out_steps = np.zeros(len(out_sources))
    out_offsets = np.zeros(len(out_sources))
    for position, source in enumerate(out_sources):
        if source.is_off_:
            continue
        if source.is_input_ or source.is_always_on_:
            raise GoldenGridError("a segment output that no core computes carries no law grid")
        core = int(source.core_)
        out_codes[:, position] = segment.core(core)[:, int(source.neuron_)]
        out_steps[position] = hcm.cores[core].readout_law.output_step[int(source.neuron_)]
        out_offsets[position] = output_offsets(hcm.cores[core].readout_law)[int(source.neuron_)]
    for s in stage.output_map:
        node = int(s.node_id)
        codes[node] = out_codes[:, s.offset: s.offset + s.size]
        steps[node] = out_steps[s.offset: s.offset + s.size]
        offsets[node] = out_offsets[s.offset: s.offset + s.size]
        values[node] = codes[node] * steps[node] - offsets[node]
    if trace is not None:
        for index, core_codes in segment.done.items():
            trace.core_codes[id(hcm.cores[index])] = core_codes


def run_golden(program, x, *, trace: Optional[GoldenTrace] = None,
               keep: Optional[Dict[str, Dict[int, np.ndarray]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """``(output codes int64, output steps)`` of ``program`` on the host input ``x`` (integer or real values:
    the entry snap rounds each host value onto its boundary grid, as the executor does; never truncated). Host
    ops are replayed segment by segment in fp64 by the executor's seam; bound banks read their payload codes
    per sample (N2). ``trace`` collects the payloads, host values and core codes the certificate compares.
    ``keep`` (a dict) receives every node output the program materializes, by node id: ``codes``, ``steps`` and
    ``values`` (codes x steps) of segment outputs, and ``values`` of host ops, for per-operator censuses [CL7].
    [thru:w2] A program of input slots runs once per slot on its rows (``input_slot_runs``), the codes put back in
    batch order; its ``trace`` and ``keep`` (one run's reads) are refused by name."""
    runs = input_slot_runs(program.stages, len(x))
    if runs == [(None, None)]:
        return _run(program, x, trace=trace, keep=keep)
    if trace is not None or keep is not None:
        raise BatchPipelineError("batch_pipeline_slots: the golden's trace and kept node outputs read one run of the "
                                 "program, and a program of input slots runs once per input slot")
    x = np.asarray(x, dtype=VALUE_ARITHMETIC.core_np)
    parts = [(rows, _run(program, x[rows], input_slot=b)) for b, rows in runs]
    steps = parts[0][1][1]
    if any(not np.array_equal(run[1], steps) for _rows, run in parts):
        raise GoldenGridError("batch_pipeline_slots: the input slots' runs read their outputs on different grids")
    return scatter_rows([(rows, run[0]) for rows, run in parts], len(x)), steps


def _run(program, x, *, trace: Optional[GoldenTrace] = None, keep: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
         input_slot: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """One run of ``program`` (see :func:`run_golden`); [thru:w2] ``input_slot``: the run of that slot's rows."""
    bindings = getattr(program, "dynamic_bank_bindings", None) or None
    x = np.asarray(x, dtype=VALUE_ARITHMETIC.core_np)
    flat = x.reshape(x.shape[0], -1)
    values: Dict[int, np.ndarray] = {_RAW_INPUT: flat}
    codes: Dict[int, np.ndarray] = {}
    steps: Dict[int, np.ndarray] = {}
    offsets: Dict[int, np.ndarray] = {}
    twins: Dict = {}
    x_torch = torch.as_tensor(flat)
    for stage in program.stages:
        if input_slot is not None:  # [thru:w2] this slot's stages and every input's, each read written earlier
            if not serves_slot(stage, input_slot):
                continue
            require_inputs_produced(stage, values, input_slot)
        if stage.kind == "neural":
            _neural_stage(stage, values, codes, steps, offsets, bindings, trace)
        elif stage.kind == "compute":
            op = stage.compute_op
            state = {k: torch.as_tensor(v, dtype=VALUE_ARITHMETIC.core_dtype) for k, v in values.items()}
            with torch.no_grad():
                values[int(op.id)] = execute_host_op(
                    op, x_torch, state, dtype=VALUE_ARITHMETIC.core_dtype, twins=twins,
                ).numpy()
            if trace is not None:
                trace.host_values[int(op.id)] = values[int(op.id)]
    if keep is not None:
        keep.update(codes=codes, steps=steps, values={k: v for k, v in values.items() if k != _RAW_INPUT})
    out = np.asarray(program.output_sources).flatten()
    out_codes = np.zeros((flat.shape[0], out.size), dtype=np.int64)
    out_steps = np.zeros(out.size)
    for position, source in enumerate(out):
        if not isinstance(source, IRSource) or source.is_off():
            continue
        if int(source.node_id) not in codes:
            raise GoldenGridError(f"output {position} is not a law-carrying core's output (node {source.node_id})")
        out_codes[:, position] = codes[int(source.node_id)][:, int(source.index)]
        out_steps[position] = steps[int(source.node_id)][int(source.index)]
    return out_codes, out_steps


def evaluate(program, x) -> np.ndarray:
    """The program's output codes (int64) on the host input ``x`` (integer or real values, see :func:`run_golden`)."""
    return run_golden(program, x)[0]
