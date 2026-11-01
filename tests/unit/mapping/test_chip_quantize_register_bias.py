"""[U-WQB A, phase C defect D-WQ] The chip export of a bias coded on the weight lattice of an on-chip register.

On the two-scale projection's on-chip register route the bias is ``round(b s_w)`` in weight-grid units and
``bias_scale == parameter_scale`` (``test_napq_register_bias.py``); the codes reach ``±q_max r``, the span the export
allocates. The export reads the span from the bias's own codes, ``r = ceil(max|round(b s)| / q_max)`` (never narrower
than a stamped ratio, so a ratio-grid bias exports as before), and widens the code dtype when the span outgrows it.
Without the route declaration the export is unchanged: a shared-grid bias saturates at ``±q_max`` and a ratio-grid bias
at ``±q_max r`` (``test_chip_quantize_bias_clamp.py``, ``test_chip_quantize_two_scale.py``).

The route reaches ``quantize_ir_graph`` as a grid declaration (``ParameterGrids.register_bias``) that Soft Core Mapping
sets from the run's route; a tree that names no route (R) exports with its default grids.
"""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph, verify_ir_graph_quantized
from mimarsinan.mapping.ir import IRGraph, NeuralCore, WeightBank
from mimarsinan.mapping.ir.source import IRSource
from mimarsinan.mapping.support.grid_declarations import ParameterGrids
from mimarsinan.transformations.quantization_bounds import quantization_bounds

BITS = 4
Q_MIN, Q_MAX = quantization_bounds(BITS)  # -8, 7
S_W = 58.0


def _register_route() -> ParameterGrids:
    """The export's grids on the register route, where the tree declares the route; R declares none."""
    if "register_bias" in getattr(ParameterGrids, "__dataclass_fields__", {}):
        return ParameterGrids(register_bias=True)
    return ParameterGrids()


def _core(codes, *, bias_scale=S_W, parameter_scale=S_W, node_id=0):
    matrix = np.array([[0.12, -0.05], [0.03, 7.0 / S_W]], dtype=np.float64)
    return NeuralCore(
        id=node_id, name=f"c{node_id}",
        input_sources=np.array([IRSource(node_id=-2, index=i) for i in range(matrix.shape[0])]),
        core_matrix=matrix,
        hardware_bias=np.asarray(codes, dtype=np.float64) / float(parameter_scale),
        parameter_scale=torch.tensor(float(parameter_scale)),
        bias_scale=None if bias_scale is None else torch.tensor(float(bias_scale)),
    )


def _graph(core):
    out = np.array([IRSource(node_id=core.id, index=j) for j in range(core.core_matrix.shape[1])])
    return IRGraph(nodes=[core], output_sources=out)


def test_a_register_bias_beyond_q_max_is_kept():
    """T-A2. ``bias_scale == parameter_scale``, code 40 at q_max 7 (span r = ceil(40 / 7) = 6): kept, not clipped.
    R clips it to 7."""
    graph = _graph(_core([40.0, -23.0]))
    quantize_ir_graph(graph, BITS, weight_quantization=True, grids=_register_route())
    node = graph.nodes[0]
    np.testing.assert_array_equal(node.hardware_bias, np.array([40, -23]))
    assert node.threshold == S_W
    assert node.core_matrix.max() <= Q_MAX and node.core_matrix.min() >= Q_MIN
    verify_ir_graph_quantized(graph, BITS, grids=_register_route())


def test_the_register_widens_the_code_dtype_beyond_int8():
    """Codes up to q_max r = 7 x 28 = 196 (S4's largest r at 4 bits) exceed int8: the export widens, never wraps."""
    graph = _graph(_core([196.0, -201.0]))
    quantize_ir_graph(graph, BITS, weight_quantization=True, grids=_register_route())
    hb = graph.nodes[0].hardware_bias
    assert np.issubdtype(hb.dtype, np.integer) and np.iinfo(hb.dtype).max >= 201
    np.testing.assert_array_equal(hb.astype(np.int64), np.array([196, -201]))


def test_the_register_keeps_the_proportionality_with_the_threshold():
    """NF to SCM: hardware_bias / threshold reproduces the effective bias exactly on the register route."""
    codes = np.array([40.0, -57.0])
    graph = _graph(_core(codes))
    quantize_ir_graph(graph, BITS, weight_quantization=True, grids=_register_route())
    node = graph.nodes[0]
    np.testing.assert_allclose(node.hardware_bias.astype(np.float64) / node.threshold, codes / S_W, rtol=0, atol=1e-12)


def test_a_ratio_grid_bias_exports_as_before_on_the_register_route():
    """A cache projected on the ratio grid (bias_scale = s_w / 13) exports the same codes and dtype on the route."""
    ratio = 13
    codes = [7 * ratio, -5 * ratio]
    routed = _graph(_core(codes, bias_scale=S_W / ratio))
    plain = _graph(_core(codes, bias_scale=S_W / ratio))
    quantize_ir_graph(routed, BITS, weight_quantization=True, grids=_register_route())
    quantize_ir_graph(plain, BITS, weight_quantization=True)
    np.testing.assert_array_equal(routed.nodes[0].hardware_bias, plain.nodes[0].hardware_bias)
    assert routed.nodes[0].hardware_bias.dtype == plain.nodes[0].hardware_bias.dtype


def test_a_banked_register_bias_is_kept():
    bank_matrix = np.array([[0.12, -0.05], [0.03, 7.0 / S_W]], dtype=np.float64)
    codes = np.array([40.0, -30.0])
    bank = WeightBank(id=0, core_matrix=bank_matrix, parameter_scale=torch.tensor(S_W),
                      bias_scale=torch.tensor(S_W), hardware_bias=codes / S_W)
    node = NeuralCore(id=0, name="banked", input_sources=np.array([IRSource(node_id=-2, index=i) for i in range(2)]),
                      core_matrix=None, weight_bank_id=0, weight_row_slice=(0, 2), hardware_bias=codes / S_W,
                      parameter_scale=torch.tensor(S_W), bias_scale=torch.tensor(S_W))
    graph = IRGraph(nodes=[node], output_sources=np.array([IRSource(node_id=0, index=j) for j in range(2)]),
                    weight_banks={0: bank})
    quantize_ir_graph(graph, BITS, weight_quantization=True, grids=_register_route())
    np.testing.assert_array_equal(graph.nodes[0].hardware_bias, np.array([40, -30]))


def test_without_the_route_the_shared_grid_still_saturates():
    """Guard: the default grids keep the weight register's saturation (``test_chip_quantize_bias_clamp.py``)."""
    graph = _graph(_core([40.0, -23.0]))
    quantize_ir_graph(graph, BITS, weight_quantization=True)
    np.testing.assert_array_equal(graph.nodes[0].hardware_bias, np.array([Q_MAX, Q_MIN]))
