"""Tests for IRMapping: neural core creation, tiling, and map_fc."""

import pytest
import torch
import numpy as np

from mimarsinan.mapping.ir import IRSource, NeuralCore, IRGraph
from mimarsinan.mapping.ir_mapping import IRMapping


class TestIRMappingBasic:
    def test_add_neural_core(self):
        m = IRMapping()
        sources = np.array([IRSource(-2, 0), IRSource(-2, 1)])
        w = torch.tensor([[1.0, 0.5], [0.3, 0.7]])
        b = torch.tensor([0.1, 0.2])
        out = m.add_neural_core(sources, w, b)

        assert len(m.nodes) == 1
        core = m.nodes[0]
        assert isinstance(core, NeuralCore)
        assert core.core_matrix.shape == (3, 2)
        assert len(out) == 2

    def test_add_compute_op(self):
        m = IRMapping()
        sources = np.array([IRSource(-2, i) for i in range(4)])
        out = m.add_compute_op(sources, "flatten", {})
        assert len(m.nodes) == 1
        assert m.nodes[0].op_type == "flatten"
        assert len(out.flatten()) == 4

    def test_node_ids_are_sequential(self):
        m = IRMapping()
        s1 = np.array([IRSource(-2, 0)])
        w = torch.tensor([[1.0]])
        m.add_neural_core(s1, w)
        m.add_compute_op(s1, "identity", {})
        m.add_neural_core(s1, w)

        ids = [n.id for n in m.nodes]
        assert ids == [0, 1, 2]


class TestIRMappingMapFC:
    def test_simple_fc_no_tiling(self):
        m = IRMapping(max_axons=64, max_neurons=64)
        sources = np.array([IRSource(-2, i) for i in range(8)])
        out_shape = np.array([4])
        w = torch.randn(4, 8)
        b = torch.randn(4)
        out = m.map_fc(sources, out_shape, w, b, name="fc1")

        assert len(m.nodes) == 1
        assert m.nodes[0].core_matrix.shape == (9, 4)
        assert len(out.flatten()) == 4

    def test_neuron_tiling(self):
        """When neurons > max_neurons, should split into multiple cores."""
        m = IRMapping(max_axons=64, max_neurons=4)
        sources = np.array([IRSource(-2, i) for i in range(8)])
        out_shape = np.array([10])
        w = torch.randn(10, 8)
        b = torch.randn(10)
        out = m.map_fc(sources, out_shape, w, b, name="fc_wide")

        assert len(m.nodes) >= 3
        assert len(out.flatten()) == 10

    def test_axon_tiling_disabled_raises(self):
        """When axons > max and tiling is off, should raise or use psum."""
        m = IRMapping(max_axons=4, max_neurons=64, allow_axon_tiling=False)
        sources = np.array([IRSource(-2, i) for i in range(10)])
        out_shape = np.array([4])
        w = torch.randn(4, 10)
        b = torch.randn(4)
        with pytest.raises((AssertionError, RuntimeError, ValueError)):
            m.map_fc(sources, out_shape, w, b, name="fc_too_wide")

    def test_axon_tiling_enabled(self):
        """With axon tiling, large axon counts should produce partial-sum cores."""
        m = IRMapping(max_axons=6, max_neurons=64, allow_axon_tiling=True)
        sources = np.array([IRSource(-2, i) for i in range(10)])
        out_shape = np.array([4])
        w = torch.randn(4, 10)
        b = torch.randn(4)
        out = m.map_fc(sources, out_shape, w, b, name="fc_tiled")

        assert len(out.flatten()) == 4
        assert len(m.nodes) > 1

    def test_map_produces_valid_graph(self):
        m = IRMapping(max_axons=32, max_neurons=32)
        sources = np.array([IRSource(-2, i) for i in range(8)])
        out_shape = np.array([4])
        w = torch.randn(4, 8)
        b = torch.randn(4)
        out = m.map_fc(sources, out_shape, w, b, name="fc")
        m.output_sources = out

        graph = IRGraph(nodes=m.nodes.copy(), output_sources=out)
        errors = graph.validate()
        assert errors == []
