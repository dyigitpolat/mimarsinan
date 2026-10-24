"""E282.S: the readout law's declared parameters, its multiplier rule, the bias a core's law reads and the solver's
refusals (shared by the per-core solver ``law_solver`` and the lane solver ``mapping/lanes/law/solve.py``)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import numpy as np
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.support.quantizer_state import DECLARED_STATE_ATTRIBUTE
from mimarsinan.models.nn.activations.integer_law import IntegerLawRangeError, round_half_away

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


#: Half-away rounding to int64 (public for the lane solver, WI-5e).
round_ints = _round_ints


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


def core_bias_and_data_weights(core: NeuralCore, ir_graph, *,
                               real_bias: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """(bias per column in weight-grid units, integer weight codes of the non-bias rows) of a quantized core; off
    and pruned rows stay in, so a bank slice's positions share one bound. The bias is an integer unless declared
    at the law's resolution (``real_bias``)."""
    matrix = np.asarray(core.get_core_matrix(ir_graph))
    sources = core.input_sources.flatten()
    if len(sources) != matrix.shape[0]:
        raise ValueError(f"core {core.name!r}: {len(sources)} sources for a {matrix.shape[0]}-row matrix")
    on = np.array([source.is_always_on() for source in sources], dtype=bool)
    bias = matrix[on, :].astype(np.float64).sum(axis=0) if on.any() else np.zeros(matrix.shape[1])
    if core.hardware_bias is not None:
        bias = bias + np.asarray(core.hardware_bias, dtype=np.float64)
    if not real_bias and not np.array_equal(bias, np.round(bias)):
        raise ReadoutLawDeclarationError(f"core {core.name!r}: bias is not on the weight grid")
    return bias, matrix[~on, :]


def adc_step_exponent(need: int, levels: int, *, core: str) -> int:
    """The smallest t with ``levels 2^t >= need``: the ADC's codes span ``[-need, need]`` unclamped."""
    for t in range(MAX_SHIFT + 1):
        if (int(levels) << t) >= int(need):
            return t
    raise ReadoutLawDeclarationError(f"core {core!r}: an ADC of {levels} levels cannot span |acc| <= {need}")


def _refuse_psum(ir_graph) -> None:
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore) and node.psum_role is not None:
            raise ReadoutLawPsumError(
                f"core {node.name!r} is a partial-sum core ({node.psum_role}); its law must apply to the final "
                f"sum (E285)"
            )


def _refuse_stale_fixed_range(quantizers, declared, skip=frozenset()) -> None:
    """A declared ``range_value`` must be every quantizer's range (in the buffer's own dtype), so a law
    re-solved under a new declaration (CL6) never reads a range calibrated under an earlier one; a range from
    the declared quantizer state and an identity output's inert range are judged by their own declarations."""
    if declared is None:
        return
    stale = {}
    for index, (_mapper, _perceptron, quantizer) in quantizers.items():
        if index in skip or hasattr(quantizer, DECLARED_STATE_ATTRIBUTE):
            continue
        want = torch.tensor(float(declared), dtype=quantizer.range.dtype, device=quantizer.range.device)
        if not bool(torch.equal(quantizer.range, want)):
            stale[index] = float(quantizer.range)
    if stale:
        raise ReadoutLawStaleRangeError(
            f"readout_law declares range_value {float(declared)!r} but the readout quantizers of perceptrons "
            f"{sorted(stale)} carry {sorted(set(stale.values()))}: rerun Readout Law under this declaration"
        )


#: Public readings of the same refusals (the lane solver, WI-5e).
refuse_psum = _refuse_psum
refuse_stale_fixed_range = _refuse_stale_fixed_range
