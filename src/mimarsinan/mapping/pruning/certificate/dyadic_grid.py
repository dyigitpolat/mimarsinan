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


def _on_grid(array, fraction_bits: int) -> bool:
    scaled = np.ldexp(np.asarray(array, dtype=np.float64), fraction_bits)
    return bool(np.all(np.isfinite(scaled)) and np.array_equal(scaled, np.rint(scaled)))


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
    ir_graph: IRGraph, *, fraction_bits: int = DEFAULT_FRACTION_BITS
) -> None:
    """Refuse unless every value the program can produce stays on a dyadic grid.

    On grid-closed programs fp64 sums are exact, so dropping exactly-zero
    structure is bit-neutral regardless of kernel blocking; off the grid,
    column compaction shifts BLAS lanes and ulp reassociation would make the
    zero-tolerance comparison trip on legitimate programs.
    """
    problems: list[str] = []
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            if node.core_matrix is not None and not _on_grid(
                node.core_matrix, fraction_bits
            ):
                problems.append(
                    f"NeuralCore id={node.id} core_matrix is off the dyadic "
                    f"2^-{fraction_bits} grid"
                )
            bias = getattr(node, "hardware_bias", None)
            if bias is not None and not _on_grid(bias, fraction_bits):
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
            if not _op_preserves_dyadic_grid(node):
                problems.append(
                    f"ComputeOp id={node.id} op_type={node.op_type!r} maps "
                    "dyadic values off-grid"
                )
    for bank_id, bank in (getattr(ir_graph, "weight_banks", None) or {}).items():
        if not _on_grid(bank.core_matrix, fraction_bits):
            problems.append(
                f"WeightBank id={bank_id} core_matrix is off the dyadic grid"
            )
        bias = getattr(bank, "hardware_bias", None)
        if bias is not None and not _on_grid(bias, fraction_bits):
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
