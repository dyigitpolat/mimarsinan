"""Tests for IR: IRSource, IRNode, NeuralCore, ComputeOp, IRGraph."""

import pytest
import torch
import numpy as np

from mimarsinan.mapping.ir import (
    IRSource, IRGraph, NeuralCore, ComputeOp,
    soft_core_to_neural_core, neural_core_to_soft_core,
    ir_graph_to_soft_core_mapping, soft_core_mapping_to_ir_graph,
)


class TestIRSource:
    def test_off(self):
        s = IRSource(node_id=-1, index=0)
        assert s.is_off()
        assert not s.is_input()
        assert not s.is_always_on()

    def test_input(self):
        s = IRSource(node_id=-2, index=5)
        assert s.is_input()
        assert not s.is_off()

    def test_always_on(self):
        s = IRSource(node_id=-3, index=0)
        assert s.is_always_on()

    def test_node_reference(self):
        s = IRSource(node_id=7, index=3)
        assert not s.is_off()
        assert not s.is_input()
        assert not s.is_always_on()
        assert s.node_id == 7
        assert s.index == 3


class TestNeuralCore:
    def _make_core(self, axons=4, neurons=2, core_id=0):
        w = np.random.randn(axons, neurons).astype(np.float32)
        sources = np.array(
            [IRSource(node_id=-2, index=i) for i in range(axons)],
            dtype=object,
        )
        return NeuralCore(id=core_id, name=f"core_{core_id}",
                          input_sources=sources, core_matrix=w)

    def test_get_input_output_count(self):
        c = self._make_core(axons=5, neurons=3)
        assert c.get_input_count() == 5
        assert c.get_output_count() == 3

    def test_execute_relu_output(self):
        w = np.array([[1.0], [-1.0]], dtype=np.float32)
        sources = np.array([IRSource(-2, 0), IRSource(-2, 1)], dtype=object)
        core = NeuralCore(id=0, name="c", input_sources=sources, core_matrix=w)

        x = torch.tensor([[2.0, 3.0]])
        out = core.execute(x, {})
        # W^T @ x = [1, -1] . [2, 3] = 2 - 3 = -1 → relu → 0
        assert out.shape == (1, 1)
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_execute_positive_output(self):
        w = np.array([[1.0], [1.0]], dtype=np.float32)
        sources = np.array([IRSource(-2, 0), IRSource(-2, 1)], dtype=object)
        core = NeuralCore(id=0, name="c", input_sources=sources, core_matrix=w)

        x = torch.tensor([[2.0, 3.0]])
        out = core.execute(x, {})
        assert out.item() == pytest.approx(1.0, abs=1e-5)

    def test_always_on_source_acts_as_bias(self):
        w = np.array([[0.0], [5.0]], dtype=np.float32)
        sources = np.array([IRSource(-2, 0), IRSource(-3, 0)], dtype=object)
        core = NeuralCore(id=0, name="c", input_sources=sources, core_matrix=w)

        x = torch.tensor([[0.0]])
        out = core.execute(x, {})
        assert out.item() == pytest.approx(1.0, abs=1e-5)

    def test_off_source_contributes_zero(self):
        w = np.array([[1.0], [999.0]], dtype=np.float32)
        sources = np.array([IRSource(-2, 0), IRSource(-1, 0)], dtype=object)
        core = NeuralCore(id=0, name="c", input_sources=sources, core_matrix=w)

        x = torch.tensor([[3.0]])
        out = core.execute(x, {})
        assert out.item() == pytest.approx(1.0, abs=1e-5)

    def test_chained_cores(self):
        """Core1 feeds core2 via node references."""
        w1 = np.array([[1.0, 0.5]], dtype=np.float32)
        s1 = np.array([IRSource(-2, 0)], dtype=object)
        c1 = NeuralCore(id=0, name="c1", input_sources=s1, core_matrix=w1)

        w2 = np.array([[1.0], [1.0]], dtype=np.float32)
        s2 = np.array([IRSource(0, 0), IRSource(0, 1)], dtype=object)
        c2 = NeuralCore(id=1, name="c2", input_sources=s2, core_matrix=w2)

        x = torch.tensor([[2.0]])
        out1 = c1.execute(x, {})
        out2 = c2.execute(x, {0: out1})
        assert out2.shape == (1, 1)
        assert out2.item() > 0


class TestComputeOp:
    def test_flatten(self):
        sources = np.array([IRSource(-2, i) for i in range(6)], dtype=object)
        op = ComputeOp(id=0, name="flat", input_sources=sources,
                       op_type="flatten", input_shape=(2, 3))
        x = torch.randn(2, 6)
        out = op.execute_on_gathered(x)
        assert out.shape == (2, 6)

    def test_identity(self):
        sources = np.array([IRSource(-2, i) for i in range(4)], dtype=object)
        op = ComputeOp(id=0, name="id", input_sources=sources, op_type="identity")
        x = torch.randn(3, 4)
        out = op.execute_on_gathered(x)
        assert torch.allclose(x.view(3, -1), out)

    def test_add(self):
        sources = np.array([IRSource(-2, i) for i in range(4)], dtype=object)
        op = ComputeOp(id=0, name="add", input_sources=sources,
                       op_type="add", params={"half_size": 2})
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = op.execute_on_gathered(x)
        expected = torch.tensor([[4.0, 6.0]])
        assert torch.allclose(out, expected)

    def test_gelu(self):
        sources = np.array([IRSource(-2, i) for i in range(3)], dtype=object)
        op = ComputeOp(id=0, name="gelu", input_sources=sources, op_type="gelu")
        x = torch.tensor([[0.0, 1.0, -1.0]])
        out = op.execute_on_gathered(x)
        expected = torch.nn.functional.gelu(x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_max_pool2d(self):
        sources = np.array([IRSource(-2, i) for i in range(16)], dtype=object)
        op = ComputeOp(id=0, name="pool", input_sources=sources,
                       op_type="max_pool2d",
                       input_shape=(1, 4, 4),
                       params={"kernel_size": 2, "stride": 2, "padding": 0})
        x = torch.arange(16, dtype=torch.float32).unsqueeze(0)
        out = op.execute_on_gathered(x)
        assert out.shape == (1, 4)

    def test_unsupported_op_raises(self):
        sources = np.array([IRSource(-2, 0)], dtype=object)
        op = ComputeOp(id=0, name="bad", input_sources=sources, op_type="unknown_op")
        with pytest.raises(NotImplementedError, match="unknown_op"):
            op.execute_on_gathered(torch.tensor([[1.0]]))


class TestIRGraph:
    def test_get_neural_cores(self, tiny_ir_graph):
        cores = tiny_ir_graph.get_neural_cores()
        assert len(cores) == 2
        assert all(isinstance(c, NeuralCore) for c in cores)

    def test_get_compute_ops_empty(self, tiny_ir_graph):
        assert tiny_ir_graph.get_compute_ops() == []

    def test_get_node_by_id(self, tiny_ir_graph):
        node = tiny_ir_graph.get_node_by_id(0)
        assert node is not None
        assert node.name == "hidden"

    def test_get_node_by_id_missing(self, tiny_ir_graph):
        assert tiny_ir_graph.get_node_by_id(999) is None

    def test_validate_valid_graph(self, tiny_ir_graph):
        errors = tiny_ir_graph.validate()
        assert errors == []

    def test_validate_broken_reference(self):
        bad_sources = np.array([IRSource(node_id=99, index=0)], dtype=object)
        core = NeuralCore(id=0, name="c", input_sources=bad_sources,
                          core_matrix=np.ones((1, 1)))
        out = np.array([IRSource(0, 0)], dtype=object)
        graph = IRGraph(nodes=[core], output_sources=out)
        errors = graph.validate()
        assert len(errors) > 0
        assert "99" in errors[0]

    def test_empty_graph(self):
        g = IRGraph(nodes=[], output_sources=np.array([], dtype=object))
        assert g.get_neural_cores() == []
        assert g.validate() == []


class TestIRConversions:
    def test_ir_graph_to_soft_core_and_back(self, tiny_ir_graph):
        scm = ir_graph_to_soft_core_mapping(tiny_ir_graph)
        assert len(scm.cores) == 2

        rebuilt = soft_core_mapping_to_ir_graph(scm)
        assert len(rebuilt.nodes) == 2
        assert len(rebuilt.output_sources) == len(tiny_ir_graph.output_sources)

    def test_ir_graph_with_compute_op_rejects_conversion(self):
        sources = np.array([IRSource(-2, 0)], dtype=object)
        op = ComputeOp(id=0, name="op", input_sources=sources,
                       op_type="identity")
        graph = IRGraph(nodes=[op], output_sources=sources)
        with pytest.raises(ValueError, match="ComputeOp"):
            ir_graph_to_soft_core_mapping(graph)
