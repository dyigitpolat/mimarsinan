"""Quantize IR NeuralCore / WeightBank matrices for chip deployment."""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.ir.dynamic_bank import dynamic_bank_ids
from mimarsinan.mapping.support.grid_declarations import GridDeclarationError, ParameterGrids
from mimarsinan.transformations.quantization_bounds import core_weight_bits, quantization_bounds
from mimarsinan.transformations.quantization_verify import assert_integer_scaled_matrix


def quantize_ir_graph(
    ir_graph: IRGraph,
    bits: int,
    *,
    weight_quantization: bool,
    grids: ParameterGrids = ParameterGrids(),
) -> None:
    """In-place quantize weights/thresholds/hardware_bias on ``ir_graph`` (``grids``: the declared scale and bias
    rules of a value-domain program; the default is the integer lattice with the bias on the weight grid)."""
    q_min, q_max = quantization_bounds(bits)
    q_dtype = code_dtype(bits)
    eps = 1e-12

    if not weight_quantization:
        for bank in ir_graph.weight_banks.values():
            bank.parameter_scale = torch.tensor(1.0)
        for node in ir_graph.nodes:
            if isinstance(node, NeuralCore):
                node.threshold = 1.0
                node.parameter_scale = torch.tensor(1.0)
        return

    if grids.law_bias:
        refuse_law_bias_rows(ir_graph)
    quantized_banks: set[int] = set()
    bank_scale_used: dict[int, float] = {}
    # N2: a bound bank's matrix is a payload of integer codes written at run time; its law carries the scale.
    dynamic = dynamic_bank_ids(ir_graph)
    for bank_id, bank in ir_graph.weight_banks.items():
        if bank_id in dynamic:
            continue
        b_min, b_max = _bounds_of(bank, bits, q_min, q_max)  # [WI-5g] the bank's core-type width
        scale = _matrix_scale(bank.core_matrix, bank.parameter_scale, b_max, eps, grids.fractional_scale)
        bank.core_matrix = np.clip(np.round(bank.core_matrix * scale), b_min, b_max).astype(
            _dtype_of(bank, bits, q_dtype))
        bank.parameter_scale = torch.tensor(1.0)
        quantized_banks.add(bank_id)
        bank_scale_used[bank_id] = scale

    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        n_min, n_max = _bounds_of(node, bits, q_min, q_max)
        n_dtype = _dtype_of(node, bits, q_dtype)
        if node.has_weight_bank():
            if node.weight_bank_id in dynamic:
                continue
            if node.weight_bank_id in quantized_banks:
                scale_used = bank_scale_used[node.weight_bank_id]
                node.threshold = scale_used
                node.parameter_scale = torch.tensor(1.0)
                _scale_hardware_bias(node, scale_used, n_min, n_max, n_dtype, eps, grids.law_bias)
            continue
        matrix = node.core_matrix
        if matrix is None:
            raise ValueError(
                f"quantize_ir_graph: NeuralCore {node.name} (id={node.id}) has "
                f"neither a weight bank nor an owned core_matrix."
            )
        scale = _matrix_scale(matrix, node.parameter_scale, n_max, eps, grids.fractional_scale)
        node.core_matrix = np.clip(np.round(matrix * scale), n_min, n_max).astype(n_dtype)
        node.threshold = scale
        node.parameter_scale = torch.tensor(1.0)
        _scale_hardware_bias(node, scale, n_min, n_max, n_dtype, eps, grids.law_bias)


def refuse_law_bias_rows(ir_graph: IRGraph) -> None:
    """``bias_resolution: law`` holds each core's bias in its readout law; a bias riding an always-on axon row would
    be rounded and clipped onto the weight grid instead, so a core carrying one is refused by name (the Weight
    Quantization step refuses the declaration earlier on the pipeline route, ``projection_grids``)."""
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore) and any(s.is_always_on() for s in node.input_sources.flatten()):
            raise GridDeclarationError(
                f"bias_resolution='law' needs an on-chip bias, but core {node.name!r} carries its bias as an "
                "always-on axon row on the weight grid: declare has_bias (hardware_bias) or the weight-grid bias")


def _bounds_of(holder, bits: int, q_min: int, q_max: int) -> tuple[int, int]:
    """The grid bounds of an IR core or bank: its core type's width (WI-5g), else the program's ``bits``."""
    width = core_weight_bits(holder, bits)
    return (q_min, q_max) if width == int(bits) else quantization_bounds(width)


def _dtype_of(holder, bits: int, q_dtype):
    """The code dtype of an IR core or bank: the program's ``q_dtype``, or its core type's own width's, which holds
    every code of that width (int32 above 16 bits, so a wide type's codes never wrap)."""
    width = core_weight_bits(holder, bits)
    return q_dtype if width == int(bits) else code_dtype(width)


def code_dtype(bits: int):
    """The integer dtype holding every code of a ``bits``-wide weight grid: int8, int16, and int32 above 16 bits
    (the registry admits widths up to 32), so no code wraps."""
    return np.int8 if bits <= 8 else np.int16 if bits <= 16 else np.int32


def _matrix_scale(matrix, parameter_scale, q_max: int, eps: float, fractional: bool = False) -> float:
    """Integer-lattice grid scale (== the deployed integer threshold).

    A TRAINED scale (NAPQ parameter_scale) must already be integral — the
    network was optimized against that lattice, so fractional drift here
    deploys a different fire condition than was trained and fails loud.
    The mapping-time fallback (never-trained cores, e.g. relays) rounds.
    A declared fractional rule (a value-domain law carries the scale in its
    multiplier) deploys the trained scale as is, and q_max / max|w| otherwise."""
    ps = float(
        parameter_scale.item()
        if hasattr(parameter_scale, "item")
        else parameter_scale
    )
    if fractional:
        return ps if abs(ps) > eps else q_max / max(float(np.max(np.abs(matrix))), eps)
    if abs(ps) > eps:
        snapped = max(1.0, float(round(ps)))
        if abs(ps - snapped) > 1e-6 * snapped:
            raise ValueError(
                f"quantize_ir_graph: trained parameter_scale {ps} is not on "
                f"the integer lattice (chip threshold_t=int); NAPQ must snap "
                f"the grid during training — refusing to truncate at deploy."
            )
        return snapped
    w_max = float(np.max(np.abs(matrix)))
    w_max = max(w_max, eps)
    # FLOOR keeps w_max*scale <= q_max (no clip saturation; relay margin survives).
    return max(1.0, float(np.floor(q_max / w_max)))


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
    node: NeuralCore, scale: float, q_min: int, q_max: int, q_dtype, eps: float, law_bias: bool = False
) -> None:
    """Quantize hardware_bias with the same saturation NAPQ applies — ±q_max on
    the bias's OWN grid, i.e. ±q_max*r in weight-grid units — clamping BEFORE
    the int cast so an out-of-range bias saturates instead of wrapping
    (NF↔SCM parity), on a dtype wide enough for the ratio lattice. A bias
    declared at the law's resolution stays real (fp64, weight-grid units):
    the readout law rounds it, the float executor adds it as is."""
    if node.hardware_bias is None:
        return
    if law_bias:
        node.hardware_bias = np.asarray(node.hardware_bias, dtype=np.float64) * float(scale)
        node.bias_scale = None
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


def verify_ir_graph_quantized(ir_graph: IRGraph, bits: int, *, grids: ParameterGrids = ParameterGrids()) -> None:
    """Raise ``AssertionError`` if any NeuralCore / bank fails integer quant checks (a declared fractional scale
    or law-resolution bias is not one)."""
    q_min, q_max = quantization_bounds(bits)
    failures: list[str] = []
    bank_checked: set[int] = set()
    dynamic = dynamic_bank_ids(ir_graph)
    for core in ir_graph.get_neural_cores():
        if core.weight_bank_id is not None and int(core.weight_bank_id) in dynamic:
            continue
        c_min, c_max = _bounds_of(core, bits, q_min, q_max)
        ps = core.parameter_scale
        scale = float(ps.item() if hasattr(ps, "item") else ps)
        th = float(core.threshold)
        if not grids.fractional_scale and abs(th - round(th)) > 1e-6:
            failures.append(
                f"{core.name}: threshold {th} is not integral — the chip "
                f"threshold register is an integer (threshold_t=int); a "
                f"fractional theta deploys a different fire condition than "
                f"the SSOT simulates"
            )
        if core.hardware_bias is not None and not grids.law_bias:
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
                mat, scale, c_min, c_max, name=core.name
            )
        )
    if failures:
        msg = "IR graph quantization verification FAILED:\n" + "\n".join(
            f"  - {e}" for e in failures[:50]
        )
        if len(failures) > 50:
            msg += f"\n  ... (+{len(failures) - 50} more)"
        raise AssertionError(msg)
