"""Quantize IR NeuralCore / WeightBank matrices for chip deployment."""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.ir.dynamic_bank import dynamic_bank_ids
from mimarsinan.mapping.support.bias_rows import bias_register_span
from mimarsinan.mapping.support.grid_declarations import GridDeclarationError, ParameterGrids
from mimarsinan.mapping.support.neuron_thresholds import (
    bank_neuron_channels, is_per_neuron, localized_neuron_scales, node_neuron_channels, threshold_columns,
    thresholds_integral, trained_neuron_scales,
)
from mimarsinan.transformations.quantization_bounds import _bounds_of, _dtype_of, code_dtype, quantization_bounds
from mimarsinan.transformations.quantization_verify import assert_integer_scaled_matrix


#: [U-BRANGE2] The IR core attribute counting the on-chip bias register codes its export saturated (a census).
BIAS_REGISTER_CLIPPED = "bias_register_clipped"


def bias_register_census(ir_graph: IRGraph) -> dict:
    """[U-BRANGE2] ``{core name: codes}`` of the cores whose on-chip bias register saturated any code at export (a
    compensated bias beyond the register's span saturates as NAPQ saturates it; the count is a census, not a refusal)."""
    return {str(n.name): int(n.__dict__.get(BIAS_REGISTER_CLIPPED, 0)) for n in ir_graph.nodes
            if isinstance(n, NeuralCore) and int(n.__dict__.get(BIAS_REGISTER_CLIPPED, 0)) > 0}


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
    bank_scale_used: dict[int, "float | np.ndarray"] = {}
    # N2: a bound bank's matrix is a payload of integer codes written at run time; its law carries the scale.
    dynamic = dynamic_bank_ids(ir_graph)
    peaks = _untrained_perceptron_peaks(ir_graph, eps, payload_banks=dynamic)
    for bank_id, bank in ir_graph.weight_banks.items():
        if bank_id in dynamic:
            continue
        b_min, b_max = _bounds_of(bank, bits, q_min, q_max)  # [WI-5g] the bank's core-type width
        weight_scale = bank.parameter_scale
        if is_per_neuron(weight_scale):  # [U-PCH] each bank column on its neuron's own grid (read, never written back)
            weight_scale, _ = localized_neuron_scales(
                bank, bank_neuron_channels(bank_id, bank.core_matrix.shape[1], ir_graph.nodes))
        scale = _matrix_scale(bank.core_matrix, weight_scale, b_max, eps, grids.fractional_scale,
                              peaks.get(getattr(bank, "perceptron_index", None)))
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
        weight_scale, bias_scale = node.parameter_scale, node.bias_scale
        if is_per_neuron(weight_scale) or is_per_neuron(bias_scale):  # [U-PCH] at this core's columns (read only)
            weight_scale, bias_scale = localized_neuron_scales(
                node, node_neuron_channels(node, node.get_output_count()))
        if node.has_weight_bank():
            if node.weight_bank_id in dynamic:
                continue
            if node.weight_bank_id in quantized_banks:
                scale_used = threshold_columns(bank_scale_used[node.weight_bank_id], *(node.weight_row_slice or (0, None)))
                node.threshold = scale_used
                node.parameter_scale = torch.tensor(1.0)
                _scale_hardware_bias(node, scale_used, n_min, n_max, n_dtype, eps, grids.law_bias, grids.register_bias,
                                     bias_scale)
            continue
        matrix = node.core_matrix
        if matrix is None:
            raise ValueError(
                f"quantize_ir_graph: NeuralCore {node.name} (id={node.id}) has "
                f"neither a weight bank nor an owned core_matrix."
            )
        scale = _matrix_scale(matrix, weight_scale, n_max, eps, grids.fractional_scale,
                              peaks.get(node.perceptron_index))
        node.core_matrix = np.clip(np.round(matrix * scale), n_min, n_max).astype(n_dtype)
        node.threshold = scale
        node.parameter_scale = torch.tensor(1.0)
        _scale_hardware_bias(node, scale, n_min, n_max, n_dtype, eps, grids.law_bias, grids.register_bias, bias_scale)


def refuse_law_bias_rows(ir_graph: IRGraph) -> None:
    """``bias_resolution: law`` holds each core's bias in its readout law; a bias riding an always-on axon row would
    be rounded and clipped onto the weight grid instead, so a core carrying one is refused by name (the Weight
    Quantization step refuses the declaration earlier on the pipeline route, ``projection_grids``)."""
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore) and any(s.is_always_on() for s in node.input_sources.flatten()):
            raise GridDeclarationError(
                f"bias_resolution='law' needs an on-chip bias, but core {node.name!r} carries its bias as an "
                "always-on axon row on the weight grid: declare has_bias (hardware_bias) or the weight-grid bias")


def _untrained(parameter_scale, eps: float) -> bool:
    if is_per_neuron(parameter_scale):  # [U-PCH] a per-neuron scale is a projection's, always trained
        return False
    ps = float(parameter_scale.item() if hasattr(parameter_scale, "item") else parameter_scale)
    return abs(ps) <= eps


def _untrained_perceptron_peaks(ir_graph: IRGraph, eps: float, *, payload_banks=frozenset()) -> dict:
    """[E1] Per perceptron with no trained scale, the largest weight magnitude over every core and bank of it: the
    fallback grid is one per perceptron, as the trained lattice is, so the IR's tiling at the declared geometry never
    changes a perceptron's integers or threshold (a core without a perceptron, a relay, keeps its own). A bank bound
    to a payload (N2, ``payload_banks``) holds no weights, only a placeholder the quantizer skips, so it sets no grid."""
    peaks: dict = {}
    owned = [(n.perceptron_index, n.parameter_scale, n.core_matrix) for n in ir_graph.nodes
             if isinstance(n, NeuralCore) and not n.has_weight_bank() and n.core_matrix is not None]
    banked = [(getattr(b, "perceptron_index", None), b.parameter_scale, b.core_matrix)
              for bank_id, b in ir_graph.weight_banks.items() if bank_id not in payload_banks]
    for perceptron, parameter_scale, matrix in owned + banked:
        if perceptron is not None and _untrained(parameter_scale, eps) and np.size(matrix):
            peaks[perceptron] = max(peaks.get(perceptron, 0.0), float(np.max(np.abs(matrix))))
    return peaks


def _matrix_scale(matrix, parameter_scale, q_max: int, eps: float, fractional: bool = False,
                  peak: float | None = None) -> "float | np.ndarray":
    """Integer-lattice grid scale (== the deployed integer threshold).

    A TRAINED scale (NAPQ parameter_scale) must already be integral — the
    network was optimized against that lattice, so fractional drift here
    deploys a different fire condition than was trained and fails loud.
    The mapping-time fallback (never-trained cores, e.g. relays) rounds, on
    ``peak`` (its perceptron's largest magnitude, ``_untrained_perceptron_peaks``)
    where the core has a perceptron, else on its own.
    A declared fractional rule (a value-domain law carries the scale in its
    multiplier) deploys the trained scale as is, and the fallback q_max / max|w|
    otherwise, on the same ``peak`` (one grid per perceptron under either rule). [U-PCH] A per-neuron scale (read at
    the core's columns) is one threshold register per neuron column."""
    if is_per_neuron(parameter_scale):
        return trained_neuron_scales(parameter_scale, fractional=fractional)
    ps = float(
        parameter_scale.item()
        if hasattr(parameter_scale, "item")
        else parameter_scale
    )
    if fractional:
        return ps if abs(ps) > eps else q_max / _fallback_magnitude(matrix, peak, eps)
    if abs(ps) > eps:
        snapped = max(1.0, float(round(ps)))
        if abs(ps - snapped) > 1e-6 * snapped:
            raise ValueError(
                f"quantize_ir_graph: trained parameter_scale {ps} is not on "
                f"the integer lattice (chip threshold_t=int); NAPQ must snap "
                f"the grid during training — refusing to truncate at deploy."
            )
        return snapped
    # FLOOR keeps w_max*scale <= q_max (no clip saturation; relay margin survives).
    return max(1.0, float(np.floor(q_max / _fallback_magnitude(matrix, peak, eps))))


def _fallback_magnitude(matrix, peak: float | None, eps: float) -> float:
    """The magnitude the mapping-time fallback grid divides by: the perceptron's ``peak``, else the core's own."""
    return max(float(np.max(np.abs(matrix))) if peak is None else float(peak), eps)


def _bias_grid_ratio(node: NeuralCore, weight_scale: "float | np.ndarray", eps: float, bias_scale=None) -> int:
    """Integer ratio r = weight grid steps per bias grid step (1 == shared grid, and the on-chip register route).

    The two-scale NAPQ install's rows route snaps ``bias_scale = parameter_scale / r``
    with integer r, so the quantized bias is exactly ``r * bias_int`` in weight-grid
    units — the torch<->chip parity lattice. A non-integer ratio would break
    that lattice and must fail loud. The register route codes the bias on the
    weight lattice itself (ratio 1); its register span comes from the codes
    (``_scale_hardware_bias``)."""
    if bias_scale is None:  # [U-PCH3] the caller's read at this core's columns
        return 1
    bs = np.asarray(bias_scale.detach().cpu() if hasattr(bias_scale, "detach") else bias_scale, dtype=np.float64)
    if np.all(np.abs(bs) <= eps):
        return 1
    # [U-PCH] per-neuron grids give one ratio per neuron column; the rows they share take one r
    ratio = np.asarray(weight_scale, dtype=np.float64) / bs
    snapped = max(1, int(round(float(np.max(ratio)))))
    if np.any(np.abs(ratio - snapped) > 1e-3 * snapped):
        raise ValueError(
            f"quantize_ir_graph: NeuralCore {node.name} bias grid ratio "
            f"{ratio} (parameter_scale/bias_scale) is not integer-snapped to "
            f"the weight grid; the chip bias lattice cannot represent it."
        )
    return snapped


def _scale_hardware_bias(
    node: NeuralCore, scale: "float | np.ndarray", q_min: int, q_max: int, q_dtype, eps: float, law_bias: bool = False,
    register_bias: bool = False, bias_scale=None,
) -> None:
    """Quantize hardware_bias with the same saturation NAPQ applies — ±q_max on
    the bias's OWN grid, i.e. ±q_max*r in weight-grid units — clamping BEFORE
    the int cast so an out-of-range bias saturates instead of wrapping
    (NF↔SCM parity), on a dtype wide enough for the ratio lattice. On the
    two-scale projection's on-chip register route (``register_bias``) NAPQ codes
    ``round(b s_w)`` within ±q_max*r with r = ceil(max|code| / q_max): the
    register spans the bias's own codes (never narrower than a stamped ratio, so
    a ratio-grid bias exports as before). A bias declared at the law's
    resolution stays real (fp64, weight-grid units): the readout law rounds it,
    the float executor adds it as is."""
    if node.hardware_bias is None:
        return
    if law_bias:
        node.hardware_bias = np.asarray(node.hardware_bias, dtype=np.float64) * float(scale)
        node.bias_scale = None
        return
    ratio = _bias_grid_ratio(node, scale, eps, bias_scale)
    codes = np.round(node.hardware_bias * scale)
    if register_bias and np.size(codes):
        ratio = max(ratio, bias_register_span(float(np.max(np.abs(codes))), float(q_max)))
    lo, hi = q_min * ratio, q_max * ratio
    dtype_info = np.iinfo(q_dtype)
    dtype = q_dtype if (lo >= dtype_info.min and hi <= dtype_info.max) else np.int32
    # [U-BRANGE2] the register saturates as NAPQ does (NF and SCM parity); the codes it clips are counted, never silent
    clipped = int(np.count_nonzero((codes < lo) | (codes > hi)))
    if clipped:
        vars(node)[BIAS_REGISTER_CLIPPED] = clipped
    node.hardware_bias = np.clip(codes, lo, hi).astype(dtype)
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
        th = core.threshold  # [U-PCH] one register, or one per neuron column
        if not grids.fractional_scale and not thresholds_integral(th, tol=1e-6):
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

