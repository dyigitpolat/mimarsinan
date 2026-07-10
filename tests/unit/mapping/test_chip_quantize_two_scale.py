"""Chip export of the M2 two-scale WQ projection: ``hardware_bias`` carries its
own scale as an exact integer lattice in weight-grid units.

After the two-scale NAPQ install the effective bias sits on
``scale_b = parameter_scale / r`` (integer ``r``), so ``bias * parameter_scale``
is EXACTLY ``r * bias_int`` — an integer that may exceed the weight register's
``q_max``.  ``quantize_ir_graph`` must therefore clamp the bias at the NAPQ
saturation bound one level up (``±q_max * r``, mirroring the torch-side clamp on
the bias's own grid) instead of the weight bound, and pick an int dtype wide
enough to hold it.  The legacy path (no ``bias_scale`` / ratio 1) stays
byte-identical, including the ``±q_max`` clamp of
``test_chip_quantize_bias_clamp.py``.
"""

import numpy as np
import torch

from mimarsinan.mapping.export.chip_quantize import (
    quantize_ir_graph,
    verify_ir_graph_quantized,
)
from mimarsinan.mapping.ir import IRGraph, NeuralCore, WeightBank
from mimarsinan.mapping.ir.types import IRSource
from mimarsinan.transformations.quantization_bounds import quantization_bounds

BITS = 5
Q_MIN, Q_MAX = quantization_bounds(BITS)  # -16, 15

# A bias-dominant two-scale install: weight grid 1/250, bias ratio r = 13.
SCALE_W = float(Q_MAX) / 0.06  # 250.0
RATIO = 13
SCALE_B = SCALE_W / RATIO


def _core(core_matrix, hardware_bias, parameter_scale, bias_scale=None, node_id=0):
    core_matrix = np.asarray(core_matrix, dtype=np.float64)
    n_axons, n_neurons = core_matrix.shape
    return NeuralCore(
        id=node_id,
        name=f"c{node_id}",
        input_sources=np.array(
            [IRSource(node_id=-2, index=i) for i in range(n_axons)]
        ),
        core_matrix=core_matrix,
        hardware_bias=(
            None if hardware_bias is None
            else np.asarray(hardware_bias, dtype=np.float64)
        ),
        parameter_scale=torch.tensor(float(parameter_scale)),
        bias_scale=(None if bias_scale is None else torch.tensor(float(bias_scale))),
    )


def _graph(core):
    n_neurons = core.core_matrix.shape[1]
    out = np.array([IRSource(node_id=core.id, index=j) for j in range(n_neurons)])
    return IRGraph(nodes=[core], output_sources=out)


class TestTwoScaleBiasExport:
    def test_bias_ints_are_exact_ratio_multiples_beyond_q_max(self):
        # bias_int = [15, -8] on the bias grid -> effective bias k*r/scale_w.
        eff_bias = np.array([15 * RATIO, -8 * RATIO], dtype=np.float64) / SCALE_W
        graph = _graph(_core(
            core_matrix=np.array([[0.06, -0.02], [0.004, 0.052]]),
            hardware_bias=eff_bias,
            parameter_scale=SCALE_W,
            bias_scale=SCALE_B,
        ))
        quantize_ir_graph(graph, BITS, weight_quantization=True)

        node = graph.nodes[0]
        assert node.threshold == SCALE_W
        assert np.issubdtype(node.core_matrix.dtype, np.integer)
        assert node.core_matrix.max() <= Q_MAX and node.core_matrix.min() >= Q_MIN
        np.testing.assert_array_equal(
            node.hardware_bias, np.array([15 * RATIO, -8 * RATIO])
        )
        assert node.hardware_bias.max() > Q_MAX, (
            "the two-scale bias lattice legitimately exceeds the weight q_max"
        )

    def test_bias_saturates_at_q_max_times_ratio(self):
        # Effective bias far beyond the register range (post-WQ mutations can
        # push it off-grid): saturate at q_max * r, never wrap, never q_max.
        graph = _graph(_core(
            core_matrix=np.array([[0.06]]),
            hardware_bias=np.array([2.0]),  # 2.0 * 250 = 500 > 15 * 13 = 195
            parameter_scale=SCALE_W,
            bias_scale=SCALE_B,
        ))
        quantize_ir_graph(graph, BITS, weight_quantization=True)
        hb = graph.nodes[0].hardware_bias
        assert int(hb[0]) == Q_MAX * RATIO

    def test_negative_bias_saturates_at_q_min_times_ratio(self):
        graph = _graph(_core(
            core_matrix=np.array([[0.06]]),
            hardware_bias=np.array([-2.0]),
            parameter_scale=SCALE_W,
            bias_scale=SCALE_B,
        ))
        quantize_ir_graph(graph, BITS, weight_quantization=True)
        assert int(graph.nodes[0].hardware_bias[0]) == Q_MIN * RATIO

    def test_dtype_widens_to_hold_the_ratio_lattice(self):
        eff_bias = np.array([15 * RATIO], dtype=np.float64) / SCALE_W
        graph = _graph(_core(
            core_matrix=np.array([[0.06]]),
            hardware_bias=eff_bias,
            parameter_scale=SCALE_W,
            bias_scale=SCALE_B,
        ))
        quantize_ir_graph(graph, BITS, weight_quantization=True)
        hb = graph.nodes[0].hardware_bias
        assert np.issubdtype(hb.dtype, np.integer)
        assert int(hb[0]) == 15 * RATIO, "no int8 wraparound on the wider lattice"

    def test_torch_chip_proportionality_is_exact(self):
        """NF<->SCM contract: hardware_bias / threshold reproduces the effective
        bias exactly (the deployed membrane is scale_w times the torch one)."""
        eff_bias = np.array([7 * RATIO, -3 * RATIO], dtype=np.float64) / SCALE_W
        graph = _graph(_core(
            core_matrix=np.array([[0.06, -0.02], [0.004, 0.052]]),
            hardware_bias=eff_bias.copy(),
            parameter_scale=SCALE_W,
            bias_scale=SCALE_B,
        ))
        quantize_ir_graph(graph, BITS, weight_quantization=True)
        node = graph.nodes[0]
        np.testing.assert_allclose(
            node.hardware_bias.astype(np.float64) / node.threshold,
            eff_bias,
            rtol=0,
            atol=1e-12,
        )

    def test_verify_passes_after_two_scale_quantization(self):
        eff_bias = np.array([15 * RATIO, -8 * RATIO], dtype=np.float64) / SCALE_W
        graph = _graph(_core(
            core_matrix=np.array([[0.06, -0.02], [0.004, 0.052]]),
            hardware_bias=eff_bias,
            parameter_scale=SCALE_W,
            bias_scale=SCALE_B,
        ))
        quantize_ir_graph(graph, BITS, weight_quantization=True)
        verify_ir_graph_quantized(graph, BITS)


class TestLegacySharedGridExportUnchanged:
    def test_no_bias_scale_keeps_q_max_clamp_and_dtype(self):
        graph = _graph(_core(
            core_matrix=np.array([[0.1]]),
            hardware_bias=np.array([10.0]),
            parameter_scale=Q_MAX / 0.1,
        ))
        quantize_ir_graph(graph, BITS, weight_quantization=True)
        hb = graph.nodes[0].hardware_bias
        assert int(hb[0]) == Q_MAX
        assert hb.dtype == np.int8

    def test_equal_scales_behave_exactly_like_legacy(self):
        scale = Q_MAX / 0.1
        legacy = _graph(_core(
            core_matrix=np.array([[0.1, 0.05]]),
            hardware_bias=np.array([0.06, -10.0]),
            parameter_scale=scale,
        ))
        ratio_one = _graph(_core(
            core_matrix=np.array([[0.1, 0.05]]),
            hardware_bias=np.array([0.06, -10.0]),
            parameter_scale=scale,
            bias_scale=scale,
        ))
        quantize_ir_graph(legacy, BITS, weight_quantization=True)
        quantize_ir_graph(ratio_one, BITS, weight_quantization=True)
        np.testing.assert_array_equal(
            legacy.nodes[0].hardware_bias, ratio_one.nodes[0].hardware_bias
        )
        assert legacy.nodes[0].hardware_bias.dtype == ratio_one.nodes[0].hardware_bias.dtype


class TestBankBackedTwoScaleExport:
    def _bank_graph(self):
        bank_matrix = np.array(
            [[0.06, -0.02], [0.004, 0.052]], dtype=np.float64
        )  # (axons, neurons)
        eff_bias = np.array([15 * RATIO, -8 * RATIO], dtype=np.float64) / SCALE_W
        bank = WeightBank(
            id=0,
            core_matrix=bank_matrix,
            parameter_scale=torch.tensor(SCALE_W),
            bias_scale=torch.tensor(SCALE_B),
            hardware_bias=eff_bias,
        )
        node = NeuralCore(
            id=0,
            name="banked",
            input_sources=np.array(
                [IRSource(node_id=-2, index=i) for i in range(2)]
            ),
            core_matrix=None,
            weight_bank_id=0,
            weight_row_slice=(0, 2),
            hardware_bias=eff_bias.copy(),
            parameter_scale=torch.tensor(SCALE_W),
            bias_scale=torch.tensor(SCALE_B),
        )
        out = np.array([IRSource(node_id=0, index=j) for j in range(2)])
        return IRGraph(nodes=[node], output_sources=out, weight_banks={0: bank})

    def test_banked_node_bias_uses_the_bank_scale_pair(self):
        graph = self._bank_graph()
        quantize_ir_graph(graph, BITS, weight_quantization=True)
        node = graph.nodes[0]
        assert node.threshold == SCALE_W
        np.testing.assert_array_equal(
            node.hardware_bias, np.array([15 * RATIO, -8 * RATIO])
        )
        bank = graph.weight_banks[0]
        assert np.issubdtype(bank.core_matrix.dtype, np.integer)
        verify_ir_graph_quantized(graph, BITS)
