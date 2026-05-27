"""Quantize IR NeuralCore / WeightBank matrices for chip deployment."""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.transformations.quantization_bounds import quantization_bounds


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
                _scale_hardware_bias(node, scale_used, q_dtype)
            continue
        scale = _matrix_scale(node.core_matrix, node.parameter_scale, q_max, eps)
        node.core_matrix = np.clip(np.round(node.core_matrix * scale), q_min, q_max).astype(q_dtype)
        node.threshold = scale
        node.parameter_scale = torch.tensor(1.0)
        _scale_hardware_bias(node, scale, q_dtype)


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


def _scale_hardware_bias(node: NeuralCore, scale: float, q_dtype) -> None:
    if node.hardware_bias is not None:
        node.hardware_bias = np.round(node.hardware_bias * scale).astype(q_dtype)


def verify_ir_graph_quantized(ir_graph: IRGraph, bits: int) -> None:
    """Raise ``AssertionError`` if any NeuralCore / bank fails integer quant checks."""
    from mimarsinan.transformations.quantization_verify import assert_integer_scaled_matrix

    q_min, q_max = quantization_bounds(bits)
    failures: list[str] = []
    bank_checked: set[int] = set()
    for core in ir_graph.get_neural_cores():
        ps = core.parameter_scale
        try:
            scale = float(ps.item())
        except Exception:
            scale = float(ps)
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
