"""The lane law solver (WI-5e): per column ``y = clamp(rnd_s(M0 f_adc(acc) + M_1 q + B), lo, hi)``.

For ``((Δin acc - o_in Σ_i W_i + b) / θ + (q Δop - o_op) + o_out) / Δout`` on the output grid: ``M0 = rnd(2^s Δin /
(θ Δout))`` and ``M_1 = rnd(2^s Δop / Δout)`` at one shift ``s`` (the largest that keeps both within ``mult_bits``),
``B = rnd(2^s (b / θ - o_op + o_out) / Δout)``, and the input offset's term ``M_o Σ_i W_i`` folded per core
(``M_o = rnd(2^s o_in / (θ Δout))``, step 1's entry fold over every accumulator row). Without operand and offsets
the law is ``solve_core_law``'s, integer for integer.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, Hashable, Mapping, Optional, Tuple

import numpy as np

from mimarsinan.mapping.export.entry_offsets import (
    FoldRefit,
    offset_multipliers,
    offset_reader_bounds,
    refit_folded_laws,
    refuse_inexact_offset_readers,
)
from mimarsinan.mapping.export.law_rules import (
    ReadoutLawSpec,
    adc_step_exponent,
    core_bias_and_data_weights,
    multiplier_and_shift,
    refuse_psum,
    round_ints,
)
from mimarsinan.mapping.export.output_law import solve_identity_law
from mimarsinan.mapping.export.law_grids import CoreGrid
from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.ir.core_law import CoreLaw
from mimarsinan.mapping.ir.lane_law import LaneLaw
from mimarsinan.mapping.lanes.ir.fusion import OPERAND_ROWS
from mimarsinan.mapping.lanes.law.lane_grids import CoreLaneGrid, LaneGrid, assign_lane_grids, data_rows
from mimarsinan.mapping.lanes.law.multiplier_paths import refuse_lost_paths
from mimarsinan.mapping.support.bias_drive import reads_a_core, refuse_drives_off_their_grid
from mimarsinan.models.nn.activations.integer_law import (
    FP64_EXACT_BOUND,
    IntegerLawRangeError,
    accumulator_bounds,
    accumulator_width,
    clamp_reach,
    grid_levels,
    law_magnitude_bound,
)


def data_row_weight_sums(core: NeuralCore, ir_graph) -> np.ndarray:
    """``Σ_i W_ij`` (int64) over the accumulator's data rows: the rows an input offset applies to."""
    matrix = np.asarray(core.get_core_matrix(ir_graph), dtype=np.float64)
    sums = matrix[data_rows(core), :].sum(axis=0)
    if not np.array_equal(sums, np.round(sums)):
        raise ValueError(f"core {core.name!r}: the offset fold reads integer weight codes")
    return sums.astype(np.int64)


def _shift(ratios, mult_bits: int, core: str) -> int:
    return min(multiplier_and_shift(float(r), mult_bits, core=core)[1] for r in ratios)


def operand_terms(operand_m, grid: CoreLaneGrid) -> Tuple[Tuple[int, int], ...]:
    """Per column, the interval of the fused operand's term ``M_1 q`` over the codes its grid declares (``(0, 0)``
    without one): an operand reads a core's output, which that core's law clamps to those codes."""
    ends = [(0, 0) if op is None else (int(m) * op.codes[0], int(m) * op.codes[1])
            for m, op in zip(np.asarray(operand_m).tolist(), grid.operands)]
    return tuple((min(e), max(e)) for e in ends)


def solve_lane_law(*, core: NeuralCore, bias: np.ndarray, acc_bounds, grid: CoreLaneGrid,
                   spec: ReadoutLawSpec) -> LaneLaw:
    """The lane law of one core (offset fold excluded: :func:`fold_input_offset`). [U-LNR] Below the accumulator
    width the ADC's step spans, per column, the law's reach to its clamp ``[lo, hi]`` on both sides over the fused
    operand's declared codes (:func:`clamp_reach`), as ``solve_core_law`` does on a symmetric grid: every accumulator
    beyond it reads ``lo`` or ``hi`` for every operand code, so the ADC's clamp never changes an output."""
    name, theta = core.name, float(core.threshold)
    columns = int(np.asarray(bias).size)
    ratio0 = grid.inputs.step / (theta * grid.output_step)
    ratio1 = [None if op is None else op.step / grid.output_step for op in grid.operands]
    shift = _shift([ratio0] + [r for r in ratio1 if r is not None], spec.mult_bits, name)
    m0 = int(round_ints(np.ldexp(ratio0, shift)))
    m1 = np.array([0 if r is None else int(round_ints(np.ldexp(r, shift))) for r in ratio1], dtype=np.int64)
    o_op = np.array([0.0 if op is None else op.offset for op in grid.operands])
    scaled = (np.asarray(bias, dtype=np.float64) / (theta * grid.output_step)
              + (grid.output_offset - o_op) / grid.output_step)  # bias term first: solve_core_law's value at 0
    b = round_ints(np.ldexp(scaled, shift))
    bounds = [int(v) for v in acc_bounds]
    width = accumulator_width(max(bounds, default=0))
    if spec.adc_bits >= width:
        adc_levels, adc_shift = grid_levels(width), [0] * columns
    else:
        adc_levels = grid_levels(spec.adc_bits)
        adc_shift = [adc_step_exponent(min(bound, clamp_reach(m0, int(bj), shift, grid.lo, grid.hi, term)),
                                       adc_levels, core=name)
                     for bound, bj, term in zip(bounds, b.tolist(), operand_terms(m1, grid))]
    refuse_lost_paths(core, shift=shift, mult_bits=spec.mult_bits, ratio0=ratio0, m0=m0, ratio1=ratio1, m1=m1.tolist(),
                      bias=b.tolist(), acc_bounds=bounds, adc_shift=adc_shift, lo=grid.lo, hi=grid.hi,
                      codes=[None if op is None else op.codes for op in grid.operands])  # [U-LMP]
    op_levels = np.array([0 if op is None else op.levels for op in grid.operands], dtype=np.int64)
    bound = law_magnitude_bound(bounds, m0, b, adc_shift) + int((np.abs(m1) * op_levels).max(initial=0))
    if bound >= FP64_EXACT_BOUND:
        raise IntegerLawRangeError(f"core {name!r}: the lane law's |z| reaches {bound} >= 2^53 (accumulator width "
                                   f"{width} bits, mult_bits {spec.mult_bits}); fp64 execution is not exact")
    full = np.full(columns, 1)
    rows = getattr(core, OPERAND_ROWS, None)
    return LaneLaw(
        input_step=full * grid.inputs.step, m0=full * m0, shift=full * shift, bias=b, lo=full * grid.lo,
        hi=full * grid.hi, output_step=full * grid.output_step, adc_shift=np.asarray(adc_shift, dtype=np.int64),
        adc_levels=full * adc_levels, input_offset=full * grid.inputs.offset, output_offset=full * grid.output_offset,
        operand_row=np.full(columns, -1) if rows is None else np.where(
            [op is None for op in grid.operands], -1, np.asarray(rows)), operand_m=m1,
        operand_step=np.array([0.0 if op is None else op.step for op in grid.operands]),
        operand_offset=o_op,
    )


def operand_term(law: LaneLaw, grid: CoreLaneGrid) -> int:
    """The largest ``|M_1| L_op`` of a lane law's fused operands (0 without one): the |z| bound's operand term."""
    levels = np.array([0 if op is None else op.levels for op in grid.operands], dtype=np.int64)
    return int((np.abs(np.asarray(law.operand_m, dtype=np.int64)) * levels).max(initial=0))


def fold_input_offset(law: LaneLaw, core: NeuralCore, ir_graph) -> LaneLaw:
    """``B - M_o Σ_i W_i`` over the data rows (the law itself without an input offset); the integers that read the
    bias are refit from the folded one per channel (``refit_folded_laws``)."""
    offset = float(law.input_offset[0]) if law.columns else 0.0
    if offset == 0.0:
        return law
    term = offset_multipliers(law, float(core.threshold), offset) * data_row_weight_sums(core, ir_graph)
    return dataclasses.replace(law, bias=law.bias - term)


def _identity(core: NeuralCore, bias, bounds, grid: CoreLaneGrid, spec: ReadoutLawSpec) -> LaneLaw:
    if any(op is not None for op in grid.operands):
        raise IntegerLawRangeError(f"core {core.name!r}: an identity output law takes no fused operand")
    plain = CoreGrid(input_step=grid.inputs.step, input_levels=grid.inputs.levels, output_step=1.0,
                     output_range=0.0)
    law: CoreLaw = solve_identity_law(core=core.name, theta=float(core.threshold), bias=bias, acc_bounds=bounds,
                                      grid=plain, spec=spec)
    return LaneLaw.of(law, input_offset=grid.inputs.offset)


def solve_lane_laws(ir_graph: IRGraph, owners: Mapping[int, LaneGrid], spec: ReadoutLawSpec, *,
                    channel_keys: Callable[[NeuralCore], Tuple[Hashable, ...]], real_bias: bool = False,
                    identity_outputs: frozenset = frozenset()) -> Dict[int, LaneLaw]:
    """``{core id: LaneLaw}`` for every core of a lane program; equal law inputs share one object. ``channel_keys``
    names, per column, the channel the model reads one law from (the stamps' grouping): an input offset's refit
    runs over each channel's carriers."""
    refuse_psum(ir_graph)
    grids = assign_lane_grids(ir_graph, owners, spec.activation_bits)
    nodes_by_id = {int(n.id): n for n in ir_graph.nodes}
    # [bias-row gauge] a core's row drive is the top code value of the input grid assigned here (m-5).
    refuse_drives_off_their_grid(ir_graph, lambda core: (
        grids[int(core.id)].inputs.levels * grids[int(core.id)].inputs.step - grids[int(core.id)].inputs.offset
        if reads_a_core(np.asarray(core.input_sources.flatten())[data_rows(core)], nodes_by_id) else None))
    memo: Dict[tuple, LaneLaw] = {}
    laws: Dict[int, LaneLaw] = {}
    folds: Dict[int, FoldRefit[LaneLaw]] = {}
    readers: Dict[int, Tuple[NeuralCore, Tuple[int, ...], float]] = {}
    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        bias, weights = core_bias_and_data_weights(node, ir_graph, real_bias=real_bias)
        grid = grids[int(node.id)]
        bounds = tuple(accumulator_bounds(weights, grid.inputs.levels))
        if grid.inputs.offset != 0.0:  # N7: the bounds cover the rows pruning dropped, read by the model at a code
            bounds = offset_reader_bounds(node, ir_graph, bounds, grid.inputs.levels)
        identity = node.perceptron_index is not None and node.perceptron_index in identity_outputs
        rows: Optional[np.ndarray] = getattr(node, OPERAND_ROWS, None)
        key = (node.weight_bank_id, node.weight_row_slice, float(node.threshold), bias.tobytes(), bounds, grid,
               identity, None if rows is None else np.asarray(rows).tobytes())
        law = memo.get(key)
        if law is None:
            law = (_identity(node, bias, bounds, grid, spec) if identity
                   else solve_lane_law(core=node, bias=bias, acc_bounds=bounds, grid=grid, spec=spec))
            memo[key] = law
        if law.columns and float(law.input_offset[0]) != 0.0:
            readers[int(node.id)] = (node, bounds, float(law.input_offset[0]))
            fold_key = (id(law), data_row_weight_sums(node, ir_graph).tobytes())
            if fold_key not in memo:
                memo[fold_key] = fold_input_offset(law, node, ir_graph)
            narrow = spec.adc_bits < accumulator_width(max(bounds, default=0))
            folds[int(node.id)] = FoldRefit(
                node.name, memo[fold_key], bounds, identity, channels=channel_keys(node),
                reach_hi=int(grid.hi) if narrow and not identity else None,
                extra=operand_term(law, grid), reach_lo=int(grid.lo), terms=operand_terms(law.operand_m, grid))
        laws[int(node.id)] = law
    laws.update(refit_folded_laws(folds))  # per channel: a conv's position cores keep one law per channel
    refuse_inexact_offset_readers(readers, laws, ir_graph)
    return laws
