"""Quantize IR NeuralCore / WeightBank matrices for chip deployment."""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.transformations.quantization_bounds import quantization_bounds
from mimarsinan.transformations.quantization_verify import assert_integer_scaled_matrix


def quantize_ir_graph(
    ir_graph: IRGraph,
    bits: int,
    *,
    weight_quantization: bool,
) -> None:
    """In-place quantize weights/thresholds/hardware_bias on ``ir_graph``."""
    q_min, q_max = quantization_bounds(bits)
    q_dtype = np.int8 if bits <= 8 else np.int16
    eps = 1e-12

    if not weight_quantization:
        for bank in ir_graph.weight_banks.values():
            bank.parameter_scale = torch.tensor(1.0)
        for node in ir_graph.nodes:
            if isinstance(node, NeuralCore):
                node.threshold = 1.0
                node.parameter_scale = torch.tensor(1.0)
        return

    quantized_banks: set[int] = set()
    bank_scale_used: dict[int, float] = {}
    for bank_id, bank in ir_graph.weight_banks.items():
        scale = _matrix_scale(bank.core_matrix, bank.parameter_scale, q_max, eps)
        bank.core_matrix = np.clip(np.round(bank.core_matrix * scale), q_min, q_max).astype(q_dtype)
        bank.parameter_scale = torch.tensor(1.0)
        quantized_banks.add(bank_id)
        bank_scale_used[bank_id] = scale

    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        if node.has_weight_bank():
            if node.weight_bank_id in quantized_banks:
                scale_used = bank_scale_used[node.weight_bank_id]
                node.threshold = scale_used
                node.parameter_scale = torch.tensor(1.0)
                _scale_hardware_bias(node, scale_used, q_min, q_max, q_dtype, eps)
            continue
        matrix = node.core_matrix
        if matrix is None:
            raise ValueError(
                f"quantize_ir_graph: NeuralCore {node.name} (id={node.id}) has "
                f"neither a weight bank nor an owned core_matrix."
            )
        scale = _matrix_scale(matrix, node.parameter_scale, q_max, eps)
        node.core_matrix = np.clip(np.round(matrix * scale), q_min, q_max).astype(q_dtype)
        node.threshold = scale
        node.parameter_scale = torch.tensor(1.0)
        _scale_hardware_bias(node, scale, q_min, q_max, q_dtype, eps)


def _matrix_scale(matrix, parameter_scale, q_max: int, eps: float) -> float:
    ps = float(
        parameter_scale.item()
        if hasattr(parameter_scale, "item")
        else parameter_scale
    )
    if abs(ps) > eps:
        return ps
    w_max = float(np.max(np.abs(matrix)))
    w_max = max(w_max, eps)
    return q_max / w_max


def _bias_grid_ratio(node: NeuralCore, weight_scale: float, eps: float) -> int:
    """Integer ratio r = weight grid steps per bias grid step (1 == shared grid).

    The two-scale NAPQ install snaps ``bias_scale = parameter_scale / r`` with
    integer r, so the quantized bias is exactly ``r * bias_int`` in weight-grid
    units — the torch<->chip parity lattice. A non-integer ratio would break
    that lattice and must fail loud."""
    bias_scale = getattr(node, "bias_scale", None)
    if bias_scale is None:
        return 1
    bs = float(
        bias_scale.item() if hasattr(bias_scale, "item") else bias_scale
    )
    if abs(bs) <= eps:
        return 1
    ratio = weight_scale / bs
    snapped = max(1, int(round(ratio)))
    if abs(ratio - snapped) > 1e-3 * snapped:
        raise ValueError(
            f"quantize_ir_graph: NeuralCore {node.name} bias grid ratio "
            f"{ratio} (parameter_scale/bias_scale) is not integer-snapped to "
            f"the weight grid; the chip bias lattice cannot represent it."
        )
    return snapped


def _scale_hardware_bias(
    node: NeuralCore, scale: float, q_min: int, q_max: int, q_dtype, eps: float
) -> None:
    """Quantize hardware_bias with the same saturation NAPQ applies — ±q_max on
    the bias's OWN grid, i.e. ±q_max*r in weight-grid units — clamping BEFORE
    the int cast so an out-of-range bias saturates instead of wrapping
    (NF↔SCM parity), on a dtype wide enough for the ratio lattice."""
    if node.hardware_bias is None:
        return
    ratio = _bias_grid_ratio(node, scale, eps)
    lo, hi = q_min * ratio, q_max * ratio
    dtype_info = np.iinfo(q_dtype)
    dtype = q_dtype if (lo >= dtype_info.min and hi <= dtype_info.max) else np.int32
    node.hardware_bias = np.clip(
        np.round(node.hardware_bias * scale), lo, hi
    ).astype(dtype)
    # Consumed: the bias now lives as integers in weight-grid units.
    node.bias_scale = None


def verify_ir_graph_quantized(ir_graph: IRGraph, bits: int) -> None:
    """Raise ``AssertionError`` if any NeuralCore / bank fails integer quant checks."""
    q_min, q_max = quantization_bounds(bits)
    failures: list[str] = []
    bank_checked: set[int] = set()
    for core in ir_graph.get_neural_cores():
        ps = core.parameter_scale
        scale = float(ps.item() if hasattr(ps, "item") else ps)
        if core.hardware_bias is not None:
            hb = np.asarray(core.hardware_bias)
            if not np.issubdtype(hb.dtype, np.integer) and not np.allclose(
                hb.astype(np.float64), np.round(hb.astype(np.float64))
            ):
                failures.append(
                    f"{core.name}: hardware_bias is not integer-quantized"
                )
        bank_id = getattr(core, "weight_bank_id", None)
        if bank_id is not None:
            if bank_id in bank_checked:
                continue
            bank_checked.add(bank_id)
        mat = core.get_core_matrix(ir_graph)
        failures.extend(
            assert_integer_scaled_matrix(
                mat, scale, q_min, q_max, name=core.name
            )
        )
    if failures:
        msg = "IR graph quantization verification FAILED:\n" + "\n".join(
            f"  - {e}" for e in failures[:50]
        )
        if len(failures) > 50:
            msg += f"\n  ... (+{len(failures) - 50} more)"
        raise AssertionError(msg)
