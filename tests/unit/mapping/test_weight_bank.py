"""Tests for WeightBank, NeuralCore bank-backed mode, and materialization."""

import numpy as np
import pytest
import torch

from mimarsinan.mapping.ir import (
    IRGraph,
    IRSource,
    NeuralCore,
    WeightBank,
    ir_graph_to_soft_core_mapping,
    neural_core_to_soft_core,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _inp(idx):
    return IRSource(node_id=-2, index=idx)


def _on():
    return IRSource(node_id=-3, index=0)


def _make_weight_bank(axons=4, neurons=3, bank_id=0, seed=42):
    rng = np.random.RandomState(seed)
    mat = rng.randn(axons, neurons).astype(float)
    return WeightBank(
        id=bank_id,
        core_matrix=mat,
        activation_scale=torch.tensor(1.0),
        parameter_scale=torch.tensor(1.0),
        input_activation_scale=torch.tensor(1.0),
    )


# -----------------------------------------------------------------------
# WeightBank basics
# -----------------------------------------------------------------------
class TestWeightBankBasics:
    def test_creation(self):
        bank = _make_weight_bank()
        assert bank.id == 0
        assert bank.core_matrix.shape == (4, 3)

    def test_ir_graph_stores_banks(self):
        bank = _make_weight_bank()
        graph = IRGraph(nodes=[], output_sources=np.array([]), weight_banks={0: bank})
        assert graph.get_weight_bank(0) is bank
        assert graph.get_weight_bank(99) is None


# -----------------------------------------------------------------------
# NeuralCore owned vs bank-backed
# -----------------------------------------------------------------------
class TestNeuralCoreWeightModes:
    def test_owned_weight_core(self):
        mat = np.eye(3, dtype=float)
        core = NeuralCore(
            id=0, name="owned",
            input_sources=np.array([_inp(0), _inp(1), _inp(2)]),
            core_matrix=mat,
        )
        assert not core.has_weight_bank()
        assert core.get_core_matrix() is mat
        assert core.get_input_count() == 3
        assert core.get_output_count() == 3

    def test_bank_backed_core_full_bank(self):
        bank = _make_weight_bank(axons=4, neurons=3, bank_id=7)
        graph = IRGraph(nodes=[], output_sources=np.array([]), weight_banks={7: bank})

        core = NeuralCore(
            id=0, name="shared",
            input_sources=np.array([_inp(i) for i in range(4)]),
            core_matrix=None,
            weight_bank_id=7,
        )
        assert core.has_weight_bank()
        resolved = core.get_core_matrix(graph)
        np.testing.assert_array_equal(resolved, bank.core_matrix)

    def test_bank_backed_core_with_row_slice(self):
        bank = _make_weight_bank(axons=4, neurons=6, bank_id=1)
        graph = IRGraph(nodes=[], output_sources=np.array([]), weight_banks={1: bank})

        core = NeuralCore(
            id=0, name="sliced",
            input_sources=np.array([_inp(i) for i in range(4)]),
            core_matrix=None,
            weight_bank_id=1,
            weight_row_slice=(2, 5),
        )
        resolved = core.get_core_matrix(graph)
        assert resolved.shape == (4, 3)
        np.testing.assert_array_equal(resolved, bank.core_matrix[:, 2:5])

    def test_bank_backed_core_with_row_slice_output_count(self):
        core = NeuralCore(
            id=0, name="sliced",
            input_sources=np.array([_inp(0)]),
            core_matrix=None,
            weight_bank_id=1,
            weight_row_slice=(2, 5),
        )
        assert core.get_output_count() == 3

    def test_bank_backed_core_requires_graph(self):
        core = NeuralCore(
            id=0, name="no_graph",
            input_sources=np.array([_inp(0)]),
            core_matrix=None,
            weight_bank_id=99,
        )
        with pytest.raises(ValueError, match="no IRGraph"):
            core.get_core_matrix()

    def test_bank_backed_core_missing_bank(self):
        graph = IRGraph(nodes=[], output_sources=np.array([]), weight_banks={})
        core = NeuralCore(
            id=0, name="missing",
            input_sources=np.array([_inp(0)]),
            core_matrix=None,
            weight_bank_id=42,
        )
        with pytest.raises(KeyError, match="42"):
            core.get_core_matrix(graph)


# -----------------------------------------------------------------------
# IRGraph.validate with weight banks
# -----------------------------------------------------------------------
class TestIRGraphWeightBankValidation:
    def test_valid_bank_ref(self):
        bank = _make_weight_bank(axons=2, neurons=2, bank_id=0)
        core = NeuralCore(
            id=0, name="ok",
            input_sources=np.array([_inp(0), _inp(1)]),
            core_matrix=None,
            weight_bank_id=0,
        )
        graph = IRGraph(
            nodes=[core],
            output_sources=np.array([IRSource(node_id=0, index=0)]),
            weight_banks={0: bank},
        )
        assert graph.validate() == []

    def test_missing_bank_ref(self):
        core = NeuralCore(
            id=0, name="bad",
            input_sources=np.array([_inp(0)]),
            core_matrix=None,
            weight_bank_id=99,
        )
        graph = IRGraph(
            nodes=[core],
            output_sources=np.array([IRSource(node_id=0, index=0)]),
            weight_banks={},
        )
        errors = graph.validate()
        assert len(errors) == 1
        assert "99" in errors[0]


# -----------------------------------------------------------------------
# Materialization: neural_core_to_soft_core
# -----------------------------------------------------------------------
class TestWeightBankMaterialization:
    def test_owned_core(self):
        mat = np.eye(3, dtype=float)
        core = NeuralCore(
            id=0, name="owned",
            input_sources=np.array([_inp(0), _inp(1), _inp(2)]),
            core_matrix=mat,
        )
        sc = neural_core_to_soft_core(core)
        np.testing.assert_array_equal(sc.core_matrix, mat)

    def test_bank_backed_core(self):
        bank = _make_weight_bank(axons=4, neurons=3, bank_id=0)
        graph = IRGraph(
            nodes=[],
            output_sources=np.array([]),
            weight_banks={0: bank},
        )
        core = NeuralCore(
            id=0, name="shared",
            input_sources=np.array([_inp(i) for i in range(4)]),
            core_matrix=None,
            weight_bank_id=0,
        )
        sc = neural_core_to_soft_core(core, graph=graph)
        np.testing.assert_array_equal(sc.core_matrix, bank.core_matrix)

    def test_ir_graph_to_soft_core_mapping(self):
        bank = _make_weight_bank(axons=3, neurons=2, bank_id=0)
        n_pos = 4
        nodes = []
        all_out = []
        for i in range(n_pos):
            core = NeuralCore(
                id=i, name=f"c{i}",
                input_sources=np.array([_inp(j) for j in range(3)]),
                core_matrix=None,
                weight_bank_id=0,
            )
            nodes.append(core)
            all_out.extend([IRSource(node_id=i, index=k) for k in range(2)])

        graph = IRGraph(
            nodes=nodes,
            output_sources=np.array(all_out),
            weight_banks={0: bank},
        )
        scm = ir_graph_to_soft_core_mapping(graph)
        assert len(scm.cores) == n_pos
        for sc in scm.cores:
            np.testing.assert_array_equal(sc.core_matrix, bank.core_matrix)
