"""E282.S: solve the per-column integer readout law of every core and stamp it on the IR and the model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.export.law_grids import CoreGrid, assign_core_grids
from mimarsinan.mapping.export.law_stamp import readout_quantizers_by_perceptron, stamp_model_laws
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir.core_law import CoreLaw
from mimarsinan.models.nn.activations.integer_law import (
    FP64_EXACT_BOUND,
    IntegerLawRangeError,
    accumulator_bounds,
    accumulator_width,
    grid_levels,
    law_magnitude_bound,
    round_half_away,
    saturation_reach,
)

#: The int64 golden shifts by at most this many bits (``1 << (s - 1)`` stays an int64).
MAX_SHIFT = 62


class ReadoutLawDeclarationError(ToolchainRefusal, ValueError):
    """``readout_law`` declared where the integer value path cannot run."""


class ReadoutLawPsumError(ToolchainRefusal, ValueError):
    """A readout law on a partial-sum core, whose law must apply to the final sum (E285, Wave 2)."""


class ReadoutLawRatioError(ToolchainRefusal, ValueError):
    """A requantization ratio that no ``mult_bits`` multiplier with a non-negative shift represents."""


class ReadoutLawStaleRangeError(ToolchainRefusal, ValueError):
    """A declared fixed ``range_value`` that a readout quantizer does not carry (a stale Readout Law run)."""


@dataclass(frozen=True)
class ReadoutLawSpec:
    """The declared law parameters the solver reads (the run door judged the value)."""

    adc_bits: int
    mult_bits: int
    activation_bits: int
    weight_bits: int

    @classmethod
    def from_declaration(cls, readout_law: Mapping[str, Any], *, activation_bits, weight_bits) -> "ReadoutLawSpec":
        if activation_bits is None:
            raise ReadoutLawDeclarationError(
                "readout_law needs activation_bits: the output requantization writes the activation (DAC) grid"
            )
        return cls(
            adc_bits=int(readout_law["adc_bits"]), mult_bits=int(readout_law["mult_bits"]),
            activation_bits=int(activation_bits), weight_bits=int(weight_bits),
        )


def _round_ints(values) -> np.ndarray:
    return round_half_away(torch.as_tensor(np.asarray(values, dtype=np.float64))).numpy().astype(np.int64)


def multiplier_and_shift(ratio: float, mult_bits: int, *, core: str) -> Tuple[int, int]:
    """The largest shift s whose ``M0 = rnd(ratio 2^s)`` fits ``mult_bits`` unsigned bits, and that M0."""
    m_max = (1 << int(mult_bits)) - 1
    if not (math.isfinite(ratio) and ratio > 0.0):
        raise ReadoutLawRatioError(f"core {core!r}: requantization ratio {ratio!r} is not a positive number")
    if int(_round_ints(ratio)) > m_max:
        raise ReadoutLawRatioError(
            f"core {core!r}: requantization ratio {ratio:.6g} needs more than mult_bits={mult_bits} bits at shift 0"
        )
    shift = max(0, int(math.floor(math.log2(m_max / ratio))))
    while shift > 0 and int(_round_ints(math.ldexp(ratio, shift))) > m_max:
        shift -= 1
    while int(_round_ints(math.ldexp(ratio, shift + 1))) <= m_max:
        shift += 1
    if shift > MAX_SHIFT:
        raise IntegerLawRangeError(f"core {core!r}: requantization shift {shift} exceeds {MAX_SHIFT} bits")
    return int(_round_ints(math.ldexp(ratio, shift))), shift


def core_bias_and_data_weights(core: NeuralCore, ir_graph) -> Tuple[np.ndarray, np.ndarray]:
    """(integer bias per column in weight-grid units, integer weight codes of the non-bias rows) of a
    quantized core; off and pruned rows stay in, so a bank slice's positions share one bound."""
    matrix = np.asarray(core.get_core_matrix(ir_graph))
    sources = core.input_sources.flatten()
    if len(sources) != matrix.shape[0]:
        raise ValueError(f"core {core.name!r}: {len(sources)} sources for a {matrix.shape[0]}-row matrix")
    on = np.array([source.is_always_on() for source in sources], dtype=bool)
    bias = matrix[on, :].astype(np.float64).sum(axis=0) if on.any() else np.zeros(matrix.shape[1])
    if core.hardware_bias is not None:
        bias = bias + np.asarray(core.hardware_bias, dtype=np.float64)
    if not np.array_equal(bias, np.round(bias)):
        raise ReadoutLawDeclarationError(f"core {core.name!r}: bias is not on the weight grid")
    return bias, matrix[~on, :]


def adc_step_exponent(need: int, levels: int, *, core: str) -> int:
    """The smallest t with ``levels 2^t >= need``: the ADC's codes span ``[-need, need]`` unclamped."""
    for t in range(MAX_SHIFT + 1):
        if (int(levels) << t) >= int(need):
            return t
    raise ReadoutLawDeclarationError(f"core {core!r}: an ADC of {levels} levels cannot span |acc| <= {need}")


def solve_core_law(*, core: str, theta: float, bias: np.ndarray, acc_bounds: Sequence[int], grid: CoreGrid,
                   spec: ReadoutLawSpec) -> CoreLaw:
    """``y = clamp(rnd_s(M0 f_adc(acc) + B), lo, hi)`` per column, for ``(Δin acc + b) / θ`` on the output grid.

    ``acc_bounds`` bounds each column's |acc| (:func:`accumulator_bounds`); A is the width of the widest.
    An ADC of at least A bits is the identity, stored as the A-bit ADC. A narrower ADC takes, per column,
    the finest step ``2^t`` whose codes span every accumulator that is both reachable and unsaturated, so its
    clamp never changes an output: only its step does (the declared function).
    """
    columns = int(np.asarray(bias).size)
    levels_out = grid_levels(spec.activation_bits)
    m0, shift = multiplier_and_shift(grid.input_step / (theta * grid.output_step), spec.mult_bits, core=core)
    b = _round_ints(np.ldexp(np.asarray(bias, dtype=np.float64) / (theta * grid.output_step), shift))
    bounds = [int(v) for v in acc_bounds]
    if len(bounds) != columns:
        raise ValueError(f"core {core!r}: {len(bounds)} accumulator bounds for {columns} columns")
    width = accumulator_width(max(bounds, default=0))
    if spec.adc_bits >= width:
        adc_levels, adc_shift = grid_levels(width), [0] * columns
    else:
        adc_levels = grid_levels(spec.adc_bits)
        adc_shift = [
            adc_step_exponent(min(bound, saturation_reach(m0, int(bj), shift, levels_out)), adc_levels, core=core)
            for bound, bj in zip(bounds, b.tolist())
        ]
    bound = law_magnitude_bound(bounds, m0, b, adc_shift)
    if bound >= FP64_EXACT_BOUND:
        raise IntegerLawRangeError(
            f"core {core!r}: |z| reaches {bound} >= 2^53 (accumulator width {width} bits, mult_bits "
            f"{spec.mult_bits}); fp64 execution of its law is not exact"
        )
    full = np.full(columns, 1)
    return CoreLaw(
        input_step=full * grid.input_step, m0=full * m0, shift=full * shift, bias=b,
        lo=full * -levels_out, hi=full * levels_out, output_step=full * grid.output_step,
        adc_shift=np.asarray(adc_shift, dtype=np.int64), adc_levels=full * adc_levels,
    )


def _refuse_psum(ir_graph) -> None:
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore) and node.psum_role is not None:
            raise ReadoutLawPsumError(
                f"core {node.name!r} is a partial-sum core ({node.psum_role}); its law must apply to the final "
                f"sum (E285)"
            )


def solve_ir_laws(ir_graph, perceptron_ranges: Mapping[int, float], spec: ReadoutLawSpec) -> Dict[int, CoreLaw]:
    """{core id: CoreLaw}; cores with equal law inputs (a bank slice's positions) share one object."""
    _refuse_psum(ir_graph)
    grids, _steps = assign_core_grids(ir_graph, perceptron_ranges, spec.activation_bits)
    memo: Dict[tuple, CoreLaw] = {}
    laws: Dict[int, CoreLaw] = {}
    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        bias, weights = core_bias_and_data_weights(node, ir_graph)
        grid = grids[int(node.id)]
        bounds = tuple(accumulator_bounds(weights, grid.input_levels))
        key = (node.weight_bank_id, node.weight_row_slice, float(node.threshold), bias.tobytes(), bounds, grid)
        law = memo.get(key)
        if law is None:
            law = solve_core_law(core=node.name, theta=float(node.threshold), bias=bias, acc_bounds=bounds,
                                 grid=grid, spec=spec)
            memo[key] = law
        laws[int(node.id)] = law
    return laws


def _refuse_stale_fixed_range(quantizers, declared) -> None:
    """A declared ``range_value`` must be every quantizer's range (in the buffer's own dtype), so a law
    re-solved under a new declaration (CL6) never reads a range calibrated under an earlier one."""
    if declared is None:
        return
    stale = {}
    for index, (_mapper, _perceptron, quantizer) in quantizers.items():
        want = torch.tensor(float(declared), dtype=quantizer.range.dtype, device=quantizer.range.device)
        if not bool(torch.equal(quantizer.range, want)):
            stale[index] = float(quantizer.range)
    if stale:
        raise ReadoutLawStaleRangeError(
            f"readout_law declares range_value {float(declared)!r} but the readout quantizers of perceptrons "
            f"{sorted(stale)} carry {sorted(set(stale.values()))}: rerun Readout Law under this declaration"
        )


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
    quantizers = readout_quantizers_by_perceptron(model.get_mapper_repr())
    if not quantizers:
        raise ReadoutLawDeclarationError("readout_law declared but no readout quantizer is installed (Readout Law)")
    ranges = {index: float(entry[2].range) for index, entry in quantizers.items()}
    _refuse_stale_fixed_range(quantizers, readout_law.get("range_value"))
    laws = solve_ir_laws(ir_graph, ranges, spec)
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            node.readout_law = laws[int(node.id)]  # pyright: ignore[reportAttributeAccessIssue] -- declared-only attribute
    stamp_model_laws(model, ir_graph, laws, weight_bits=weight_bits)
    return len(laws)
