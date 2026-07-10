"""The two-scale bias grid must ride the IR mapping: perceptron ``bias_scale``
-> ``map_fc`` / ``register_weight_bank`` -> ``WeightBank`` / ``NeuralCore``.

A platform WITHOUT an on-chip bias register encodes the bias as an always-on
axon ROW of the core matrix, which must obey the ±q_max weight-register
contract on the weight grid — a two-scale (coarser) bias grid is not mappable
there and must fail loud (the WQ step's capability gate keeps recipes off this
path; the mapping assert is the backstop).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

SCALE_W = 250.0
SCALE_B = SCALE_W / 13


def _inp(idx):
    return IRSource(node_id=-2, index=idx)


def _map_fc(mapper, *, bias_scale, in_features=4, out_features=3):
    mapper.map_fc(
        input_tensor_sources=np.array([_inp(i) for i in range(in_features)]),
        output_shape=np.array([out_features]),
        fc_weights=torch.randn(out_features, in_features) * 0.05,
        fc_biases=torch.randn(out_features) * 0.5,
        parameter_scale=torch.tensor(SCALE_W),
        bias_scale=bias_scale,
        name="fc",
    )


class TestBiasScaleThreading:
    def test_map_fc_stamps_bias_scale_on_the_node(self):
        mapper = IRMapping(hardware_bias=True, max_axons=256, max_neurons=256)
        _map_fc(mapper, bias_scale=torch.tensor(SCALE_B))
        core = mapper.nodes[-1]
        assert core.hardware_bias is not None
        assert core.bias_scale is not None
        assert float(core.bias_scale) == pytest.approx(SCALE_B)

    def test_register_weight_bank_carries_bias_scale_to_bank_and_shared_core(self):
        mapper = IRMapping(hardware_bias=True, max_axons=256, max_neurons=256)
        bank_id = mapper.register_weight_bank(
            weights=torch.randn(3, 4) * 0.05,
            biases=torch.randn(3) * 0.5,
            parameter_scale=torch.tensor(SCALE_W),
            bias_scale=torch.tensor(SCALE_B),
        )
        bank = mapper._weight_banks[bank_id]
        assert bank.bias_scale is not None
        assert float(bank.bias_scale) == pytest.approx(SCALE_B)

        mapper.add_shared_neural_core(
            input_sources=np.array([_inp(i) for i in range(4)]),
            weight_bank_id=bank_id,
            name="shared",
        )
        core = mapper.nodes[-1]
        assert core.bias_scale is not None
        assert float(core.bias_scale) == pytest.approx(SCALE_B)

    def test_perceptron_mapper_threads_the_perceptron_bias_scale(self):
        p = Perceptron(3, 4, normalization=nn.Identity())
        p.set_activation_scale(1.0)
        p.set_parameter_scale(SCALE_W)
        p.set_bias_scale(SCALE_B)

        class _SourceStub:
            def map_to_ir(self, ir_mapping):
                # (1, in_features): PerceptronMapper transposes to one column.
                return np.array([[_inp(i) for i in range(4)]])

        mapper = IRMapping(hardware_bias=True, max_axons=256, max_neurons=256)
        PerceptronMapper(_SourceStub(), p)._map_to_ir(mapper)
        core = mapper.nodes[-1]
        assert core.bias_scale is not None
        assert float(core.bias_scale) == pytest.approx(SCALE_B)


class TestTwoScaleEndToEndExport:
    def test_napq_to_chip_ints_is_exact_across_the_seams(self):
        """NAPQ two-scale -> PerceptronMapper -> IRMapping -> quantize_ir_graph:
        the emitted ints reproduce the quantized effective parameters exactly
        (weights on ±q_max; bias = r * bias_int on the weight lattice)."""
        from mimarsinan.mapping.export.chip_quantize import (
            quantize_ir_graph,
            verify_ir_graph_quantized,
        )
        from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
            NormalizationAwarePerceptronQuantization,
        )
        from mimarsinan.transformations.perceptron.perceptron_transformer import (
            PerceptronTransformer,
        )

        bits = 5
        q_max = (2 ** (bits - 1)) - 1
        torch.manual_seed(1)
        p = Perceptron(3, 4, normalization=nn.Identity())
        p.set_activation_scale(1.0)
        with torch.no_grad():
            p.layer.weight.data.uniform_(-0.06, 0.06)
            p.layer.bias.data.copy_(torch.tensor([0.7, -0.4, 0.2]))
        NormalizationAwarePerceptronQuantization(
            bits=bits, device="cpu", rate=1.0, two_scale=True
        ).transform(p)
        eff_w = PerceptronTransformer().get_effective_weight(p).detach()
        eff_b = PerceptronTransformer().get_effective_bias(p).detach()

        class _SourceStub:
            def map_to_ir(self, ir_mapping):
                return np.array([[_inp(i) for i in range(4)]])

        mapper = IRMapping(hardware_bias=True, max_axons=256, max_neurons=256)
        sources = PerceptronMapper(_SourceStub(), p)._map_to_ir(mapper)
        graph = mapper.map(type("FakeRepr", (), {
            "map_to_ir": lambda self, m: np.asarray(sources, dtype=object)
        })())
        quantize_ir_graph(graph, bits, weight_quantization=True)
        verify_ir_graph_quantized(graph, bits)

        node = graph.get_neural_cores()[0]
        scale_w = float(node.threshold)
        assert scale_w == pytest.approx(float(p.parameter_scale), rel=1e-6)
        np.testing.assert_allclose(
            node.core_matrix.astype(np.float64),
            eff_w.numpy().T * scale_w,
            atol=1e-4,
        )
        assert np.abs(node.core_matrix).max() <= q_max
        ratio = round(float(p.parameter_scale) / float(p.bias_scale))
        np.testing.assert_allclose(
            node.hardware_bias.astype(np.float64),
            eff_b.numpy() * scale_w,
            atol=1e-4,
        )
        assert np.all(node.hardware_bias % ratio == 0), (
            "chip bias ints must sit on the r-multiple weight lattice"
        )


class TestParamEncodedBiasGuard:
    def test_two_scale_bias_rejected_without_hardware_bias(self):
        mapper = IRMapping(hardware_bias=False, max_axons=256, max_neurons=256)
        with pytest.raises(ValueError, match="bias"):
            _map_fc(mapper, bias_scale=torch.tensor(SCALE_B))

    def test_shared_grid_bias_still_maps_without_hardware_bias(self):
        mapper = IRMapping(hardware_bias=False, max_axons=256, max_neurons=256)
        _map_fc(mapper, bias_scale=torch.tensor(SCALE_W))
        core = mapper.nodes[-1]
        assert core.hardware_bias is None
        assert core.core_matrix.shape[0] == 4 + 1  # always-on bias row kept

    def test_absent_bias_scale_still_maps_without_hardware_bias(self):
        mapper = IRMapping(hardware_bias=False, max_axons=256, max_neurons=256)
        _map_fc(mapper, bias_scale=None)
        assert mapper.nodes[-1].hardware_bias is None

    def test_two_scale_bank_rejected_without_hardware_bias(self):
        mapper = IRMapping(hardware_bias=False, max_axons=256, max_neurons=256)
        with pytest.raises(ValueError, match="bias"):
            mapper.register_weight_bank(
                weights=torch.randn(3, 4) * 0.05,
                biases=torch.randn(3) * 0.5,
                parameter_scale=torch.tensor(SCALE_W),
                bias_scale=torch.tensor(SCALE_B),
            )
