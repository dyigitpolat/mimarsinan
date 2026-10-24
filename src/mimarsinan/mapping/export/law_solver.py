"""E282.S: solve the per-column integer readout law of every core and stamp it on the IR and the model."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from mimarsinan.mapping.export.entry_offsets import (
    FoldRefit,
    core_entry_offset,
    fold_entry_offset,
    host_row_weight_sums,
    refit_folded_laws,
)
from mimarsinan.mapping.export.law_grids import CoreGrid, assign_core_grids
from mimarsinan.mapping.export.law_rules import (  # the law's public names stay importable from here
    MAX_SHIFT as MAX_SHIFT,
    ReadoutLawDeclarationError,
    ReadoutLawPsumError as ReadoutLawPsumError,
    ReadoutLawRatioError as ReadoutLawRatioError,
    ReadoutLawSpec,
    ReadoutLawStaleRangeError as ReadoutLawStaleRangeError,
    adc_step_exponent,
    core_bias_and_data_weights,
    multiplier_and_shift,
    refuse_psum,
    refuse_stale_fixed_range,
    round_ints,
)
from mimarsinan.mapping.export.law_stamp import channel_keys, readout_quantizers_by_perceptron, stamp_model_laws
from mimarsinan.mapping.export.operand_laws import (
    bind_payload_steps,
    operand_floors,
    operand_ranges,
    output_groups_of,
    stamp_operand_laws,
)
from mimarsinan.mapping.export.output_law import OutputLawError, identity_output_indices, solve_identity_law
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir.core_law import CoreLaw
from mimarsinan.mapping.lanes.declaration import lanes_executed
from mimarsinan.mapping.lanes.install import install_lane_laws
from mimarsinan.mapping.support.grid_declarations import ParameterGrids, identity_output_law
from mimarsinan.models.nn.activations.integer_law import (
    FP64_EXACT_BOUND,
    IntegerLawRangeError,
    accumulator_bounds,
    accumulator_width,
    grid_levels,
    law_magnitude_bound,
    saturation_reach,
)

def solve_core_law(*, core: str, theta: float, bias: np.ndarray, acc_bounds: Sequence[int], grid: CoreGrid,
                   spec: ReadoutLawSpec, floor: Optional[int] = None) -> CoreLaw:
    """``y = clamp(rnd_s(M0 f_adc(acc) + B), lo, hi)`` per column, for ``(Δin acc + b) / θ`` on the output grid.

    ``acc_bounds`` bounds each column's |acc| (:func:`accumulator_bounds`); A is the width of the widest.
    An ADC of at least A bits is the identity, stored as the A-bit ADC. A narrower ADC takes, per column,
    the finest step ``2^t`` whose codes span every accumulator that is both reachable and unsaturated, so its
    clamp never changes an output: only its step does (the declared function). A grid that states per-column
    steps (N2 consumer groups) solves each column on its own step and levels; ``floor`` 0 is an absorbed ReLU.
    """
    columns = int(np.asarray(bias).size)
    bias = np.asarray(bias, dtype=np.float64).reshape(-1)
    bounds = [int(v) for v in acc_bounds]
    if len(bounds) != columns:
        raise ValueError(f"core {core!r}: {len(bounds)} accumulator bounds for {columns} columns")
    width = accumulator_width(max(bounds, default=0))
    steps = grid.column_steps or (grid.output_step,) * columns
    levels = grid.column_levels or (grid.output_levels or grid_levels(spec.activation_bits),) * columns
    out = {name: np.zeros(columns, dtype=np.int64) for name in ("m0", "shift", "bias", "lo", "hi", "adc_shift",
                                                                 "adc_levels")}
    out_step = np.zeros(columns)
    for step, levels_out in dict.fromkeys(zip(steps, levels)):
        cols = [j for j in range(columns) if (steps[j], levels[j]) == (step, levels_out)]
        m0, shift = multiplier_and_shift(grid.input_step / (theta * step), spec.mult_bits, core=core)
        b = round_ints(np.ldexp(bias[cols] / (theta * step), shift))
        sub = [bounds[j] for j in cols]
        if spec.adc_bits >= width:
            adc_levels, adc_shift = grid_levels(width), [0] * len(cols)
        else:
            adc_levels = grid_levels(spec.adc_bits)
            adc_shift = [
                adc_step_exponent(min(bound, saturation_reach(m0, int(bj), shift, levels_out)), adc_levels, core=core)
                for bound, bj in zip(sub, b.tolist())
            ]
        bound = law_magnitude_bound(sub, m0, b, adc_shift)
        if bound >= FP64_EXACT_BOUND:
            raise IntegerLawRangeError(
                f"core {core!r}: |z| reaches {bound} >= 2^53 (accumulator width {width} bits, mult_bits "
                f"{spec.mult_bits}); fp64 execution of its law is not exact"
            )
        out["m0"][cols], out["shift"][cols], out["bias"][cols] = m0, shift, b
        out["lo"][cols] = -levels_out if floor is None else int(floor)
        out["hi"][cols], out["adc_levels"][cols] = levels_out, adc_levels
        out["adc_shift"][cols] = np.asarray(adc_shift, dtype=np.int64)
        out_step[cols] = step
    return CoreLaw(input_step=np.full(columns, 1) * grid.input_step, output_step=out_step, **out)


def _bias_and_bounds(node: NeuralCore, ir_graph, grid: CoreGrid, *, real_bias: bool
                     ) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """A core's bias and per-column |acc| bounds (``real_bias``: the bias held at the law's resolution, WI-5c); a
    bound operand core's payload codes span ``±(2^(payload_bits-1) - 1)`` in every cell and it has no bias (N2)."""
    binding = ir_graph.dynamic_bank_bindings.get(node.weight_bank_id) if node.weight_bank_id is not None else None
    if binding is None:
        bias, weights = core_bias_and_data_weights(node, ir_graph, real_bias=real_bias)
        return bias, tuple(accumulator_bounds(weights, grid.input_levels))
    axons, neurons = binding.payload_shape
    bound = int(axons) * grid_levels(binding.payload_bits) * int(grid.input_levels)
    return np.zeros(int(neurons)), (bound,) * int(neurons)


def solve_ir_laws(ir_graph, perceptron_ranges: Mapping[int, float], spec: ReadoutLawSpec, *,
                  grids: Optional[Mapping[int, CoreGrid]] = None,
                  floors: Optional[Mapping[int, int]] = None,
                  real_bias: bool = False, identity_outputs: frozenset = frozenset()) -> Dict[int, CoreLaw]:
    """{core id: CoreLaw}; cores with equal law inputs (a bank slice's positions) share one object. ``grids``
    (``assign_core_grids``) defaults to the per-perceptron grids of ``perceptron_ranges``; ``floors`` gives a
    perceptron's output floor (0: an absorbed ReLU). Cores of the ``identity_outputs`` perceptrons take the
    identity output law; an entry offset folds into each core's bias, and the integers reading the folded bias are
    refit per channel (:func:`refit_folded_laws`)."""
    refuse_psum(ir_graph)
    if grids is None:
        grids, _steps = assign_core_grids(ir_graph, perceptron_ranges, spec.activation_bits)
    floors = dict(floors or {})
    memo: Dict[tuple, CoreLaw] = {}
    laws: Dict[int, CoreLaw] = {}
    folds: Dict[int, FoldRefit[CoreLaw]] = {}
    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        grid = grids[int(node.id)]
        bias, bounds = _bias_and_bounds(node, ir_graph, grid, real_bias=real_bias)
        floor = floors.get(int(node.perceptron_index)) if node.perceptron_index is not None else None
        identity = node.perceptron_index in identity_outputs
        if identity and floor is not None:
            raise OutputLawError(
                f"core {node.name!r}: the identity output law has no clamp, but its perceptron's operand product "
                "absorbed a ReLU (an output floor 0, N2): the two are not composed on this tree")
        key = (node.weight_bank_id, node.weight_row_slice, float(node.threshold), bias.tobytes(), bounds, grid, floor,
               identity)
        law = memo.get(key)
        if law is None:
            if identity:
                law = solve_identity_law(core=node.name, theta=float(node.threshold), bias=bias, acc_bounds=bounds,
                                         grid=grid, spec=spec)
            else:
                law = solve_core_law(core=node.name, theta=float(node.threshold), bias=bias, acc_bounds=bounds,
                                     grid=grid, spec=spec, floor=floor)
            memo[key] = law
        if core_entry_offset(node) != 0.0:
            fold_key = (id(law), host_row_weight_sums(node, ir_graph).tobytes())
            if fold_key not in memo:
                memo[fold_key] = fold_entry_offset(law, node, ir_graph)
            narrow = spec.adc_bits < accumulator_width(max(bounds, default=0))
            folds[int(node.id)] = FoldRefit(
                node.name, memo[fold_key], bounds, identity, channels=channel_keys(node),
                reach_hi=grid_levels(spec.activation_bits) if narrow and not identity else None)
        laws[int(node.id)] = law
    laws.update(refit_folded_laws(folds))  # per channel: a conv's position cores keep one law per channel
    return laws


def install_readout_laws(ir_graph, model, platform_constraints: Mapping[str, Any], *,
                         weight_quantization: bool, weight_bits: int) -> int:
    """The Soft Core Mapping hook: solve and stamp every core's law and the model's; 0 when undeclared."""
    readout_law: Optional[Mapping[str, Any]] = platform_constraints.get("readout_law")
    if readout_law is None:
        return 0
    if not weight_quantization:
        raise ReadoutLawDeclarationError("readout_law needs weight quantization: the law reads integer weights")
    spec = ReadoutLawSpec.from_declaration(
        readout_law, activation_bits=platform_constraints.get("activation_bits"), weight_bits=weight_bits,
    )
    lanes = lanes_executed(platform_constraints)
    if lanes is not None:  # WI-5e: the declared lanes fuse first, then every core takes its lane law
        return install_lane_laws(ir_graph, model, platform_constraints, spec, lanes, weight_bits=weight_bits)
    model_repr = model.get_mapper_repr()
    quantizers = readout_quantizers_by_perceptron(model_repr)
    operands = operand_ranges(model_repr)
    if not quantizers and not operands:
        raise ReadoutLawDeclarationError("readout_law declared but no readout quantizer is installed (Readout Law)")
    ranges = {**{index: float(entry[2].range) for index, entry in quantizers.items()}, **operands}
    identity = frozenset(identity_output_indices(ir_graph, model_repr, quantizers)
                         if identity_output_law(readout_law) else ())
    refuse_stale_fixed_range(quantizers, readout_law.get("range_value"), skip=identity)
    groups = output_groups_of(model_repr)
    grids, steps = assign_core_grids(ir_graph, ranges, spec.activation_bits, groups=groups)
    bind_payload_steps(ir_graph, model_repr, steps, groups, spec.activation_bits)
    laws = solve_ir_laws(ir_graph, ranges, spec, grids=grids, floors=operand_floors(model_repr),
                         real_bias=ParameterGrids.from_config(platform_constraints).law_bias,
                         identity_outputs=identity)
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            node.readout_law = laws[int(node.id)]  # pyright: ignore[reportAttributeAccessIssue] -- declared-only attribute
    stamp_model_laws(model, ir_graph, laws, weight_bits=weight_bits)
    stamp_operand_laws(model_repr, ir_graph, laws)
    return len(laws)
