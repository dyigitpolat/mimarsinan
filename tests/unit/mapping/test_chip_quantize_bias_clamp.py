"""``quantize_ir_graph`` must clamp ``hardware_bias`` to ``[q_min, q_max]`` — the same
saturation the NF-side ``NormalizationAwarePerceptronQuantization`` applies — instead of
silently wrapping modulo on the ``int8``/``int16`` cast.

Reproduction for the NF↔SCM bias-quantization asymmetry (finding
``nf_scm_parity_boundary_transcoding_ssot.md`` §3.3): NAPQ snaps the effective bias with a
clamp to ``[q_min, q_max]`` (saturating), while ``chip_quantize._scale_hardware_bias``
rounded then ``.astype(int8)`` with NO clamp — wrapping modulo on overflow. That is a
silent divergence between the two sides of the parity contract whenever
``round(bias * scale)`` exceeds the representable range (the weights-only
``parameter_scale`` path). The deployed membrane adds ``hardware_bias`` as an integer in
the same range as the quantized weights, so out-of-range biases are a genuine bug, not a
representable value.
"""

import numpy as np
import torch

from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.ir.types import IRSource
from mimarsinan.transformations.quantization_bounds import quantization_bounds

BITS = 8


def _single_core_graph(core_matrix, hardware_bias, parameter_scale):
    core_matrix = np.asarray(core_matrix, dtype=np.float64)
    n_axons, n_neurons = core_matrix.shape
    core = NeuralCore(
        id=0,
        name="c0",
        input_sources=np.array([IRSource(node_id=-2, index=i) for i in range(n_axons)]),
        core_matrix=core_matrix,
        hardware_bias=np.asarray(hardware_bias, dtype=np.float64),
        parameter_scale=torch.tensor(float(parameter_scale)),
    )
    out = np.array([IRSource(node_id=0, index=j) for j in range(n_neurons)])
    return IRGraph(nodes=[core], output_sources=out)


def test_hardware_bias_saturates_on_overflow_like_napq():
    q_min, q_max = quantization_bounds(BITS)
    # Weights bounded by 0.1 -> a weights-scaled parameter_scale = q_max/0.1 makes a
    # large bias overflow: round(10 * 1270) = 12700 >> q_max. Without a clamp,
    # astype(int8) wraps modulo (-100 here); NAPQ saturates to q_max.
    w_max = 0.1
    scale = q_max / w_max
    graph = _single_core_graph(
        core_matrix=np.array([[w_max]]),
        hardware_bias=np.array([10.0]),
        parameter_scale=scale,
    )
    quantize_ir_graph(graph, BITS, weight_quantization=True)

    hb = graph.nodes[0].hardware_bias
    assert hb.min() >= q_min and hb.max() <= q_max, (
        f"hardware_bias escaped [{q_min}, {q_max}]: {hb} (silent int-cast wraparound)"
    )
    assert int(hb[0]) == q_max, (
        f"an overflowing bias must saturate to q_max={q_max} (matching NAPQ), got {hb[0]}"
    )


def test_in_range_hardware_bias_is_exact():
    graph = _single_core_graph(
        core_matrix=np.array([[0.5]]),
        hardware_bias=np.array([3.0]),
        parameter_scale=10.0,  # round(3 * 10) = 30, well within int8
    )
    quantize_ir_graph(graph, BITS, weight_quantization=True)
    assert int(graph.nodes[0].hardware_bias[0]) == 30


def test_negative_overflow_saturates_to_q_min():
    q_min, q_max = quantization_bounds(BITS)
    w_max = 0.1
    scale = q_max / w_max
    graph = _single_core_graph(
        core_matrix=np.array([[w_max]]),
        hardware_bias=np.array([-10.0]),
        parameter_scale=scale,
    )
    quantize_ir_graph(graph, BITS, weight_quantization=True)
    hb = graph.nodes[0].hardware_bias
    assert int(hb[0]) == q_min, (
        f"a negative-overflowing bias must saturate to q_min={q_min}, got {hb[0]}"
    )
