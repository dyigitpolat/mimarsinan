"""Dyadic-grid exactness preconditions: bit-exact comparison is well-posed only on grid-closed programs."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch.nn as nn

from mimarsinan.mapping.ir import ComputeOp, IRGraph, NeuralCore
from mimarsinan.mapping.pruning.certificate.errors import (
    CascadeCertificatePreconditionError,
)
from mimarsinan.mapping.pruning.certificate.zero_preserving import (
    op_outputs_fully_constant,
)

DEFAULT_FRACTION_BITS = 8

# Host ops that map dyadic-grid values back onto the grid, so per-column
# fp64 accumulations downstream stay EXACT integers-times-2^-k and are
# invariant to the BLAS lane reassignment that column compaction causes.
_GRID_PRESERVING_TYPES = (
    nn.Identity,
    nn.ReLU,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.Flatten,
    nn.Dropout,
    nn.Dropout2d,
)


def _is_power_of_two(value: float) -> bool:
    if not math.isfinite(value) or value <= 0.0:
        return False
    mantissa, _ = math.frexp(value)
    return mantissa == 0.5


def is_on_grid(array, *, fraction_bits: int = DEFAULT_FRACTION_BITS) -> bool:
    """Is every value exactly on the dyadic grid ``k * 2^-fraction_bits``?

    THE SSOT for "exactly representable, hence invariant to precision and to summation order".
    The certificate enforces it as a precondition; the constant analysis consumes the same
    predicate to classify each fold as grid-certifiable or execution-exact-only. A second copy
    of this idea anywhere is a drift hazard -- import this one.
    """
    scaled = np.ldexp(np.asarray(array, dtype=np.float64), fraction_bits)
    return bool(np.all(np.isfinite(scaled)) and np.array_equal(scaled, np.rint(scaled)))


def _op_emits_on_grid_constants(
    op: ComputeOp, constant_outputs, fraction_bits: int
) -> bool:
    """[W4b-2] The op emits ONLY lattice constants, all of them on the grid.

    A fully resolved op cannot move anything off-grid: it emits exactly the
    values the fold wrote into the consumers' carriers. ``sigmoid(0) = 0.5``
    qualifies (0.5 is dyadic); ``GELU(c)`` for ``c != 0`` does not, so a
    GELU-constant fold stays execution-exact but REFUSES to certify — the
    established honesty pattern, not a silent pass.
    """
    if not op_outputs_fully_constant(op, constant_outputs):
        return False
    values = [
        v for (op_id, _), v in constant_outputs.items() if op_id == op.id
    ]
    return is_on_grid(np.asarray(values, dtype=np.float64), fraction_bits=fraction_bits)


def _op_preserves_dyadic_grid(op: ComputeOp) -> bool:
    module = (getattr(op, "params", None) or {}).get("module")
    if module is None:
        return str(getattr(op, "op_type", "")).lower() == "identity"
    if isinstance(module, (nn.AvgPool1d, nn.AvgPool2d)):
        divisor = getattr(module, "divisor_override", None)
        if divisor is None:
            kernel = module.kernel_size
            divisor = (
                int(np.prod(kernel)) if isinstance(kernel, (tuple, list))
                else int(kernel) ** (2 if isinstance(module, nn.AvgPool2d) else 1)
            )
        return _is_power_of_two(float(divisor))
    if isinstance(module, nn.LeakyReLU):
        return _is_power_of_two(float(module.negative_slope))
    return type(module) in _GRID_PRESERVING_TYPES


def assert_dyadic_exactness_grid(
    ir_graph: IRGraph,
    *,
    fraction_bits: int = DEFAULT_FRACTION_BITS,
    constant_outputs=None,
) -> None:
    """Refuse unless every value the program can produce stays on a dyadic grid.

    On grid-closed programs fp64 sums are exact, so dropping exactly-zero
    structure is bit-neutral regardless of kernel blocking; off the grid,
    column compaction shifts BLAS lanes and ulp reassociation would make the
    zero-tolerance comparison trip on legitimate programs.

    ``constant_outputs`` [W4b-2] is the constant lattice: an op that emits only
    on-grid constants is grid-preserving whatever its type, which is what lets
    a ``sigmoid(0) = 0.5`` fold certify while a ``GELU(c != 0)`` fold still
    refuses.
    """
    problems: list[str] = []
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            if node.core_matrix is not None and not is_on_grid(node.core_matrix, fraction_bits=fraction_bits):
                problems.append(
                    f"NeuralCore id={node.id} core_matrix is off the dyadic "
                    f"2^-{fraction_bits} grid"
                )
            bias = getattr(node, "hardware_bias", None)
            if bias is not None and not is_on_grid(bias, fraction_bits=fraction_bits):
                problems.append(
                    f"NeuralCore id={node.id} hardware_bias is off the dyadic grid"
                )
            if not _is_power_of_two(float(node.threshold)):
                problems.append(
                    f"NeuralCore id={node.id} threshold {node.threshold} is not a "
                    "power of two"
                )
            grid = getattr(node, "boundary_grid", None)
            if grid is not None and getattr(grid, "armed", False):
                problems.append(
                    f"NeuralCore id={node.id} carries an armed boundary_grid "
                    "(AQ snapping is off the certificate's dyadic grid)"
                )
        elif isinstance(node, ComputeOp):
            if not _op_preserves_dyadic_grid(node) and not (
                _op_emits_on_grid_constants(node, constant_outputs, fraction_bits)
            ):
                problems.append(
                    f"ComputeOp id={node.id} op_type={node.op_type!r} maps "
                    "dyadic values off-grid"
                )
    for bank_id, bank in (getattr(ir_graph, "weight_banks", None) or {}).items():
        if not is_on_grid(bank.core_matrix, fraction_bits=fraction_bits):
            problems.append(
                f"WeightBank id={bank_id} core_matrix is off the dyadic grid"
            )
        bias = getattr(bank, "hardware_bias", None)
        if bias is not None and not is_on_grid(bias, fraction_bits=fraction_bits):
            problems.append(f"WeightBank id={bank_id} hardware_bias is off the dyadic grid")
    if problems:
        raise CascadeCertificatePreconditionError(
            "bit-exact certification requires a dyadic-grid-closed instance; "
            f"snap it first (snap_ir_graph_to_dyadic_grid): {problems}"
        )


def snap_ir_graph_to_dyadic_grid(
    ir_graph: IRGraph, *, fraction_bits: int = DEFAULT_FRACTION_BITS
) -> IRGraph:
    """Deep-copied certification twin with weights/biases rounded onto the
    2^-fraction_bits grid and thresholds snapped to the nearest power of two.

    Host ComputeOps are left untouched: grid-breaking ops still refuse in
    :func:`assert_dyadic_exactness_grid`.
    """
    graph = copy.deepcopy(ir_graph)

    def _snap(array):
        arr = np.asarray(array)
        snapped = np.ldexp(
            np.rint(np.ldexp(arr.astype(np.float64), fraction_bits)),
            -fraction_bits,
        )
        return snapped.astype(arr.dtype)

    for node in graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        if node.core_matrix is not None:
            node.core_matrix = _snap(node.core_matrix)
        bias = getattr(node, "hardware_bias", None)
        if bias is not None:
            node.hardware_bias = _snap(bias)
        theta = float(node.threshold)
        if theta > 0.0 and not _is_power_of_two(theta):
            node.threshold = float(2.0 ** round(math.log2(theta)))
    for bank in (getattr(graph, "weight_banks", None) or {}).values():
        bank.core_matrix = _snap(bank.core_matrix)
        bias = getattr(bank, "hardware_bias", None)
        if bias is not None:
            bank.hardware_bias = _snap(bias)
    return graph
