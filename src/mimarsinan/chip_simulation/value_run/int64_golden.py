"""The int64 golden (E283): every core's law in exact int64, host ops by the same functions in fp64 on grid values."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from mimarsinan.chip_simulation.value_run.value_flow import execute_host_op
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import IRSource
from mimarsinan.models.nn.activations.integer_law import apply_law_int64, round_half_away

_RAW_INPUT = -2


class GoldenGridError(ToolchainRefusal, ValueError):
    """A golden read off its declared grid: a host value with no boundary grid, or two grids on one wire."""


def _snap(values: np.ndarray, grid) -> np.ndarray:
    """The entry snap's codes (the boundary SSOT's rounding) in int64."""
    quotient = torch.as_tensor(values, dtype=torch.float64) / float(grid.step)
    return torch.clamp(round_half_away(quotient), -grid.levels, grid.levels).numpy().astype(np.int64)


def _law_kwargs(law) -> dict:
    return dict(m0=law.m0, bias=law.bias, shift=law.shift, lo=law.lo, hi=law.hi,
                adc_shift=law.adc_shift, adc_levels=law.adc_levels)


def _row_steps(hard_core) -> np.ndarray:
    steps = np.full(int(hard_core.axons_per_core), np.nan)
    for p in hard_core.matrix_placements:
        steps[p.axon_offset: p.axon_offset + p.axons] = hard_core.readout_law.input_step[p.neuron_offset]
    return steps


class _Segment:
    """One neural stage evaluated on demand, core by core, from its segment input."""

    def __init__(self, hcm, seg_values: np.ndarray, seg_codes: np.ndarray, seg_steps: np.ndarray) -> None:
        self.hcm = hcm
        self.seg_values, self.seg_codes, self.seg_steps = seg_values, seg_codes, seg_steps
        self.done: Dict[int, np.ndarray] = {}

    def core(self, index: int) -> np.ndarray:
        hit = self.done.get(index)
        if hit is not None:
            return hit
        hc = self.hcm.cores[index]
        law = getattr(hc, "readout_law", None)
        if law is None:
            raise GoldenGridError(f"hard core {index} carries no readout law")
        steps = _row_steps(hc)
        grid = getattr(hc, "boundary_grid", None)
        batch = self.seg_values.shape[0]
        q = np.zeros((batch, int(hc.axons_per_core)), dtype=np.int64)
        for row, source in enumerate(hc.axon_sources):
            if source.is_off_ or source.is_always_on_:
                continue
            if source.is_input_:
                position = int(source.neuron_)
                if self.seg_steps[position] == self.seg_steps[position]:
                    if self.seg_steps[position] != steps[row]:
                        raise GoldenGridError(f"hard core {index} row {row}: producer and reader grids differ")
                    q[:, row] = self.seg_codes[:, position]
                elif grid is not None and grid.armed and float(grid.step) == steps[row]:
                    q[:, row] = _snap(self.seg_values[:, position], grid)
                else:
                    raise GoldenGridError(f"hard core {index} row {row}: host value without its boundary grid")
            else:
                q[:, row] = self.core(int(source.core_))[:, int(source.neuron_)]
        weight = np.asarray(hc.get_core_matrix(), dtype=np.float64)
        if not np.array_equal(weight, np.round(weight)):
            raise GoldenGridError(f"hard core {index}: weights are not integers")
        acc = q @ weight.astype(np.int64)
        codes = apply_law_int64(acc, **_law_kwargs(law))
        self.done[index] = codes
        return codes


def _neural_stage(stage, values, codes, steps) -> None:
    hcm = stage.hard_core_mapping
    width = max((s.offset + s.size for s in stage.input_map), default=0)
    batch = next(iter(values.values())).shape[0]
    seg_values = np.zeros((batch, width))
    seg_codes = np.zeros((batch, width), dtype=np.int64)
    seg_steps = np.full(width, np.nan)
    for s in stage.input_map:
        seg_values[:, s.offset: s.offset + s.size] = values[int(s.node_id)][:, : s.size]
        if int(s.node_id) in codes:
            seg_codes[:, s.offset: s.offset + s.size] = codes[int(s.node_id)][:, : s.size]
            seg_steps[s.offset: s.offset + s.size] = steps[int(s.node_id)][: s.size]
    segment = _Segment(hcm, seg_values, seg_codes, seg_steps)
    out_sources = list(np.asarray(hcm.output_sources).flatten())
    out_codes = np.zeros((batch, len(out_sources)), dtype=np.int64)
    out_steps = np.zeros(len(out_sources))
    for position, source in enumerate(out_sources):
        if source.is_off_:
            continue
        if source.is_input_ or source.is_always_on_:
            raise GoldenGridError("a segment output that no core computes carries no law grid")
        core = int(source.core_)
        out_codes[:, position] = segment.core(core)[:, int(source.neuron_)]
        out_steps[position] = hcm.cores[core].readout_law.output_step[int(source.neuron_)]
    for s in stage.output_map:
        node = int(s.node_id)
        codes[node] = out_codes[:, s.offset: s.offset + s.size]
        steps[node] = out_steps[s.offset: s.offset + s.size]
        values[node] = codes[node] * steps[node]


def run_golden(program, x) -> Tuple[np.ndarray, np.ndarray]:
    """``(output codes int64, output steps)`` of ``program`` on the host input ``x`` (integer or real values:
    the entry snap rounds each host value onto its boundary grid, as the executor does; never truncated)."""
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(x.shape[0], -1)
    values: Dict[int, np.ndarray] = {_RAW_INPUT: flat}
    codes: Dict[int, np.ndarray] = {}
    steps: Dict[int, np.ndarray] = {}
    twins: Dict = {}
    x_torch = torch.as_tensor(flat)
    for stage in program.stages:
        if stage.kind == "neural":
            _neural_stage(stage, values, codes, steps)
        elif stage.kind == "compute":
            op = stage.compute_op
            state = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in values.items()}
            with torch.no_grad():
                values[int(op.id)] = execute_host_op(
                    op, x_torch, state, dtype=torch.float64, twins=twins,
                ).numpy()
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
