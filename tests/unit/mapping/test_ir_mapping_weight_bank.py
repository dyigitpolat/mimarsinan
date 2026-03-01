"""Tests for IRMapping weight-bank methods: register_weight_bank, add_shared_neural_core."""

import numpy as np
import torch

from mimarsinan.mapping.ir import IRSource, NeuralCore
from mimarsinan.mapping.ir_mapping import IRMapping


def _inp(idx):
    return IRSource(node_id=-2, index=idx)


class TestRegisterWeightBank:
    def test_returns_sequential_ids(self):
        ir = IRMapping()
        id0 = ir.register_weight_bank(weights=np.ones((2, 3)))
        id1 = ir.register_weight_bank(weights=np.ones((4, 5)))
        assert id0 == 0
        assert id1 == 1

    def test_bank_stores_transposed_matrix_with_bias(self):
        ir = IRMapping()
        w = np.random.randn(8, 12).astype(float)
        b = np.random.randn(8).astype(float)
        bank_id = ir.register_weight_bank(weights=w, biases=b)

        bank = ir._weight_banks[bank_id]
        assert bank.core_matrix.shape == (13, 8)  # 12 inputs + 1 bias row, 8 outputs
        np.testing.assert_allclose(bank.core_matrix[:12, :], w.T, atol=1e-12)
        np.testing.assert_allclose(bank.core_matrix[12, :], b, atol=1e-12)

    def test_bank_stores_transposed_matrix_no_bias(self):
        ir = IRMapping()
        w = np.random.randn(3, 5).astype(float)
        bank_id = ir.register_weight_bank(weights=w, biases=None)

        bank = ir._weight_banks[bank_id]
        assert bank.core_matrix.shape == (5, 3)
        np.testing.assert_allclose(bank.core_matrix, w.T, atol=1e-12)


class TestAddSharedNeuralCore:
    def test_creates_bank_backed_core(self):
        ir = IRMapping()
        w = np.random.randn(8, 12).astype(float)
        b = np.random.randn(8).astype(float)
        bank_id = ir.register_weight_bank(weights=w, biases=b)

        sources = np.array([_inp(i) for i in range(12)])
        out = ir.add_shared_neural_core(
            input_sources=sources,
            weight_bank_id=bank_id,
            has_bias=True,
            name="shared_core_0",
        )
        assert out.shape == (8,)

        node = ir.nodes[-1]
        assert isinstance(node, NeuralCore)
        assert node.core_matrix is None
        assert node.weight_bank_id == bank_id
        assert len(node.input_sources) == 13  # 12 inputs + 1 bias

    def test_get_output_count_works_without_graph(self):
        """Bank-backed cores must support get_output_count() without a graph.

        This is required by _flush_neural_segment in hybrid_hardcore_mapping
        which calls n.get_output_count() before the graph is available.
        """
        ir = IRMapping()
        w = np.random.randn(8, 12).astype(float)
        bank_id = ir.register_weight_bank(weights=w, biases=None)

        sources = np.array([_inp(i) for i in range(12)])
        ir.add_shared_neural_core(
            input_sources=sources, weight_bank_id=bank_id, has_bias=False,
        )
        node = ir.nodes[-1]
        assert node.get_output_count() == 8
        assert node.weight_row_slice == (0, 8)

    def test_no_bias(self):
        ir = IRMapping()
        bank_id = ir.register_weight_bank(weights=np.ones((2, 3)), biases=None)

        sources = np.array([_inp(i) for i in range(3)])
        out = ir.add_shared_neural_core(
            input_sources=sources, weight_bank_id=bank_id, has_bias=False,
        )
        assert out.shape == (2,)
        assert len(ir.nodes[-1].input_sources) == 3  # no extra bias axon


class TestMapProducesGraphWithBanks:
    def test_round_trip(self):
        ir = IRMapping()
        w = np.ones((2, 3), dtype=float)
        bank_id = ir.register_weight_bank(weights=w, biases=None)

        class FakeRepr:
            def map_to_ir(self_, ir_mapping):
                src = np.array([_inp(0), _inp(1), _inp(2)])
                out1 = ir_mapping.add_shared_neural_core(src, bank_id, has_bias=False, name="c0")
                out2 = ir_mapping.add_shared_neural_core(src, bank_id, has_bias=False, name="c1")
                return np.concatenate([out1, out2])

        graph = ir.map(FakeRepr())
        assert len(graph.weight_banks) == 1
        assert bank_id in graph.weight_banks
        cores = graph.get_neural_cores()
        assert len(cores) == 2
        for c in cores:
            assert c.has_weight_bank()
            resolved = c.get_core_matrix(graph)
            assert resolved.shape == (3, 2)

    def test_shared_and_owned_cores_coexist(self):
        ir = IRMapping()
        w = np.ones((2, 3), dtype=float)
        bank_id = ir.register_weight_bank(weights=w, biases=None)

        src = np.array([_inp(0), _inp(1), _inp(2)])

        ir.add_shared_neural_core(src, bank_id, has_bias=False, name="shared")
        ir.add_neural_core(src, weights=np.eye(2, 3), name="owned")

        assert len(ir.nodes) == 2
        assert ir.nodes[0].has_weight_bank()
        assert not ir.nodes[1].has_weight_bank()
        assert ir.nodes[1].core_matrix is not None
