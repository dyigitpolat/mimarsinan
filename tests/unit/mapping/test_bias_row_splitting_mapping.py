"""Bias-row splitting AT THE MAPPING SSOT: k always-on rows, counted into the
axon envelope, emitted by every backend from the same core matrix.

The row count is not plumbed as a side channel — it is recovered from the grids
the WQ install stamped (``parameter_scale / bias_scale``), so a mapper, a
layout estimate and a packed chip cannot disagree about it. With splitting off
every path here is the legacy single-row path, byte for byte.
"""

import numpy as np
import pytest
import torch

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.platform.mapping_structure import (
    WideFanInUnsupportedError,
    compute_core_input_count,
    compute_fc_tiling_mode,
)
from mimarsinan.mapping.support.bias_rows import core_matrix_with_bias_rows

SCALE_W = 240.0


def _inp(idx):
    return IRSource(node_id=-2, index=idx)


def _map_fc(mapper, *, bias_scale, in_features=4, out_features=3, biases=None):
    if biases is None:
        biases = torch.tensor([0.5, -0.25, 0.125])[:out_features]
    mapper.map_fc(
        input_tensor_sources=np.array([_inp(i) for i in range(in_features)]),
        output_shape=np.array([out_features]),
        fc_weights=torch.randn(out_features, in_features) * 0.01,
        fc_biases=biases,
        parameter_scale=torch.tensor(SCALE_W),
        bias_scale=bias_scale,
        name="fc",
    )
    return mapper.nodes[-1]


class TestCoreInputCount:
    def test_the_default_is_the_legacy_single_bias_row(self):
        assert compute_core_input_count(64, has_bias=True, hardware_bias=False) == 65

    def test_k_rows_are_counted_into_the_axon_envelope(self):
        assert compute_core_input_count(
            64, has_bias=True, hardware_bias=False, bias_rows=8
        ) == 72

    def test_a_hardware_bias_lane_spends_no_rows_whatever_k_says(self):
        assert compute_core_input_count(
            64, has_bias=True, hardware_bias=True, bias_rows=8
        ) == 64

    def test_a_biasless_layer_spends_no_rows(self):
        assert compute_core_input_count(
            64, has_bias=False, hardware_bias=False, bias_rows=8
        ) == 64


class TestGeometryRefusal:
    def test_bias_rows_can_push_a_layer_past_the_axon_envelope(self):
        """THE geometry refusal: a fan-in that fits with one bias row need not
        fit with k, and the mapper must say so instead of emitting it."""
        assert compute_fc_tiling_mode(
            121, 10, 128, 256, True, False, False, bias_rows=1
        ) == "single"
        with pytest.raises(WideFanInUnsupportedError, match="129"):
            compute_fc_tiling_mode(
                121, 10, 128, 256, True, False, False, bias_rows=8
            )

    def test_a_coalescing_chip_tiles_the_widened_layer_instead(self):
        assert compute_fc_tiling_mode(
            121, 10, 128, 256, True, False, True, bias_rows=8
        ) == "coalescing"


class TestCoreMatrixAssembly:
    def test_one_row_reproduces_the_legacy_layout(self):
        w_t = np.arange(12, dtype=float).reshape(4, 3)
        b = np.array([0.5, -0.25, 0.125])
        matrix = core_matrix_with_bias_rows(w_t, b, 1)
        assert matrix.shape == (5, 3)
        np.testing.assert_array_equal(matrix[:4, :], w_t)
        np.testing.assert_array_equal(matrix[-1, :], b)

    def test_k_rows_append_below_the_weights_and_sum_to_the_bias(self):
        w_t = np.arange(12, dtype=float).reshape(4, 3)
        b = np.array([0.5, -0.25, 0.125])
        matrix = core_matrix_with_bias_rows(w_t, b, 4)
        assert matrix.shape == (8, 3)
        np.testing.assert_array_equal(matrix[:4, :], w_t)
        np.testing.assert_allclose(matrix[4:, :].sum(axis=0), b, atol=1e-15)


class TestIRMappingEmission:
    def test_splitting_off_still_refuses_a_two_scale_bias(self):
        mapper = IRMapping(hardware_bias=False, max_axons=256, max_neurons=256)
        with pytest.raises(ValueError, match="bias_row_splitting"):
            _map_fc(mapper, bias_scale=torch.tensor(SCALE_W / 8))

    def test_splitting_off_maps_the_shared_grid_byte_identically(self):
        mapper = IRMapping(hardware_bias=False, max_axons=256, max_neurons=256)
        core = _map_fc(mapper, bias_scale=torch.tensor(SCALE_W))
        assert core.core_matrix.shape[0] == 5
        assert sum(s.is_always_on() for s in core.input_sources.flatten()) == 1

    def test_splitting_on_emits_k_rows_and_k_always_on_sources(self):
        mapper = IRMapping(
            hardware_bias=False, max_axons=256, max_neurons=256,
            bias_row_splitting=True,
        )
        core = _map_fc(mapper, bias_scale=torch.tensor(SCALE_W / 8))
        assert core.core_matrix.shape[0] == 4 + 8
        assert sum(s.is_always_on() for s in core.input_sources.flatten()) == 8
        # The always-on rows are the LAST rows: axon i of the matrix is source i.
        for src in core.input_sources.flatten()[4:]:
            assert src.is_always_on()

    def test_the_emitted_rows_sum_to_the_mapped_bias(self):
        b = torch.tensor([0.5, -0.25, 0.125])
        mapper = IRMapping(
            hardware_bias=False, max_axons=256, max_neurons=256,
            bias_row_splitting=True,
        )
        core = _map_fc(mapper, bias_scale=torch.tensor(SCALE_W / 5), biases=b)
        np.testing.assert_allclose(
            core.core_matrix[4:, :].sum(axis=0), b.numpy(), atol=1e-12
        )

    def test_a_shared_weight_bank_splits_the_same_way(self):
        mapper = IRMapping(
            hardware_bias=False, max_axons=256, max_neurons=256,
            bias_row_splitting=True,
        )
        bank_id = mapper.register_weight_bank(
            weights=torch.randn(3, 4) * 0.01,
            biases=torch.tensor([0.5, -0.25, 0.125]),
            parameter_scale=torch.tensor(SCALE_W),
            bias_scale=torch.tensor(SCALE_W / 6),
        )
        assert mapper._weight_banks[bank_id].core_matrix.shape[0] == 4 + 6
        mapper.add_shared_neural_core(
            input_sources=np.array([_inp(i) for i in range(4)]),
            weight_bank_id=bank_id,
            name="shared",
        )
        core = mapper.nodes[-1]
        assert sum(s.is_always_on() for s in core.input_sources.flatten()) == 6

    def test_a_hardware_bias_platform_never_spends_a_row(self):
        mapper = IRMapping(
            hardware_bias=True, max_axons=256, max_neurons=256,
            bias_row_splitting=True,
        )
        core = _map_fc(mapper, bias_scale=torch.tensor(SCALE_W / 8))
        assert core.hardware_bias is not None
        assert core.core_matrix.shape[0] == 4


class TestDeployedIntegerExactness:
    def test_the_deployed_ints_sum_to_the_rounded_bias_with_no_residue(self):
        """The whole point: k rows on the WEIGHT grid, each inside +/-q_max,
        reproducing round(b_j * s_w) exactly."""
        from mimarsinan.mapping.export.chip_quantize import (
            quantize_ir_graph,
            verify_ir_graph_quantized,
        )

        bits, k = 4, 8
        q_max = (2 ** (bits - 1)) - 1
        weight_scale = 40.0
        bias_scale = weight_scale / k
        bias_ints = np.array([7.0, -6.0, 3.0])
        b = torch.tensor(bias_ints / bias_scale, dtype=torch.float32)

        torch.manual_seed(0)
        w = torch.round(torch.randn(3, 4) * 3.0) / weight_scale

        mapper = IRMapping(
            q_max=float(q_max), hardware_bias=False, max_axons=256,
            max_neurons=256, bias_row_splitting=True,
        )
        sources = mapper.map_fc(
            input_tensor_sources=np.array([_inp(i) for i in range(4)]),
            output_shape=np.array([3]),
            fc_weights=w,
            fc_biases=b,
            parameter_scale=torch.tensor(weight_scale),
            bias_scale=torch.tensor(bias_scale),
            name="fc",
        )
        graph = mapper.map(type("FakeRepr", (), {
            "map_to_ir": lambda self, m: np.asarray(sources, dtype=object)
        })())
        quantize_ir_graph(graph, bits, weight_quantization=True)
        verify_ir_graph_quantized(graph, bits)

        node = graph.get_neural_cores()[0]
        rows = node.core_matrix[4:, :]
        assert rows.shape[0] == k
        assert np.abs(node.core_matrix).max() <= q_max
        np.testing.assert_array_equal(
            rows.sum(axis=0), np.round(b.numpy().astype(np.float64) * weight_scale)
        )
