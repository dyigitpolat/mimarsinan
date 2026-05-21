"""Tests for ``IRGraph.remove_nodes``: the canonical whole-node deletion API.

``remove_nodes`` is the only public path for deleting NeuralCores from the
graph. It must:

- rewire every dangling reference (input_sources of any survivor, output
  sources of the graph) to ``IRSource(node_id=-1, index=0)`` ("off");
- refuse to remove a node when removal would empty out ``output_sources``
  (no surviving live output) -- raise ``ValueError``;
- treat ``psum_group_id`` and ``coalescing_group_id`` atomically: either
  every member of a partial-sum / coalescing group is removed, or none of
  them is (raise otherwise);
- drop weight banks that are no longer referenced by any surviving core.

Whole-graph invariants (``IRGraph.validate`` returns no errors) must hold
after every ``remove_nodes`` call.
"""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.mapping.ir import (
    ComputeOp,
    IRGraph,
    IRSource,
    NeuralCore,
    WeightBank,
)


def _src(specs):
    return np.array(
        [IRSource(node_id=nid, index=idx) for nid, idx in specs],
        dtype=object,
    )


def _two_core_chain(*, w0=None, w1=None, output_index=0):
    if w0 is None:
        w0 = np.array([[1.0]], dtype=np.float64)
    if w1 is None:
        w1 = np.array([[2.0]], dtype=np.float64)
    a = NeuralCore(
        id=0, name="A",
        input_sources=_src([(-2, 0)]),
        core_matrix=w0, threshold=1.0, latency=0,
    )
    b = NeuralCore(
        id=1, name="B",
        input_sources=_src([(0, 0)]),
        core_matrix=w1, threshold=1.0, latency=1,
    )
    return IRGraph(nodes=[a, b], output_sources=_src([(1, output_index)]))


class TestRewiringSemantics:
    def test_remove_nodes_drops_node_from_graph(self):
        graph = _two_core_chain()
        # Add a third independent live output so output_sources stays valid.
        c = NeuralCore(
            id=2, name="C",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        graph.nodes.append(c)
        graph.output_sources = np.append(
            graph.output_sources, _src([(2, 0)])
        )

        graph.remove_nodes([0])
        ids = [n.id for n in graph.nodes]
        assert 0 not in ids
        assert 1 in ids and 2 in ids
        assert graph.validate() == []

    def test_remove_nodes_rewires_consumers_to_off(self):
        """Every input_source pointing at a removed node must become OFF."""
        graph = _two_core_chain()
        # Keep at least one live output so removal of A does not leave the
        # graph without any output. Add a survivor C with its own output.
        c = NeuralCore(
            id=2, name="C",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        graph.nodes.append(c)
        graph.output_sources = np.append(
            graph.output_sources, _src([(2, 0)])
        )

        graph.remove_nodes([0])
        b = next(n for n in graph.nodes if n.id == 1)
        for src in b.input_sources.flatten():
            assert isinstance(src, IRSource)
            if src.node_id == 0:
                pytest.fail(
                    "B still references removed node A after remove_nodes()"
                )
        # The original (0, 0) edge should now be OFF.
        first = b.input_sources.flatten()[0]
        assert first.is_off()

    def test_remove_nodes_rewires_compute_op_input_sources(self):
        """ComputeOps that consumed a removed node's outputs are also rewired."""
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        relu = ComputeOp(
            id=1, name="relu",
            input_sources=_src([(0, 0)]),
            op_type="identity",
            input_shape=(1,), output_shape=(1,),
        )
        b = NeuralCore(
            id=2, name="B",
            input_sources=_src([(1, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=2,
        )
        # Keep a separate output path alive when removing A.
        c = NeuralCore(
            id=3, name="C",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        graph = IRGraph(
            nodes=[a, relu, b, c],
            output_sources=_src([(2, 0), (3, 0)]),
        )
        graph.remove_nodes([0])
        relu_node = next(n for n in graph.nodes if n.id == 1)
        first = relu_node.input_sources.flatten()[0]
        assert first.is_off()


class TestOutputProtection:
    def test_remove_nodes_raises_when_all_outputs_would_die(self):
        """Removing every node referenced by output_sources is forbidden."""
        graph = _two_core_chain()
        with pytest.raises(ValueError, match=r"output"):
            graph.remove_nodes([1])  # B is the only output target.

    def test_remove_nodes_drops_specific_output_entry_when_safe(self):
        """When output_sources references multiple cores and we remove one,
        the surviving entries are kept and the removed node's entries are
        rewired to OFF (not silently dropped, so output count is stable)."""
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        b = NeuralCore(
            id=1, name="B",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        graph = IRGraph(
            nodes=[a, b], output_sources=_src([(0, 0), (1, 0)])
        )
        graph.remove_nodes([0])
        flat = graph.output_sources.flatten()
        assert len(flat) == 2
        assert flat[0].is_off()
        assert flat[1].node_id == 1 and flat[1].index == 0


class TestPSumGroupAtomicity:
    def _build_psum_pair(self, *, both_dead: bool):
        """Two cores that share psum_group_id=42 (positive + negative parts).

        If ``both_dead``: caller will request both removed (atomic OK).
        If not: caller will only request one removed (atomic violation).
        """
        a = NeuralCore(
            id=0, name="A_pos",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
            psum_group_id=42, psum_role="partial_pos",
        )
        b = NeuralCore(
            id=1, name="A_neg",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[-1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
            psum_group_id=42, psum_role="partial_neg",
        )
        # Live output target unrelated to the psum pair.
        c = NeuralCore(
            id=2, name="C",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        return IRGraph(
            nodes=[a, b, c], output_sources=_src([(2, 0)])
        )

    def test_remove_nodes_atomic_per_psum_group_succeeds_when_full_group_removed(self):
        graph = self._build_psum_pair(both_dead=True)
        graph.remove_nodes([0, 1])  # full group {0, 1} removed
        ids = [n.id for n in graph.nodes]
        assert ids == [2]

    def test_remove_nodes_atomic_per_psum_group_raises_on_partial_removal(self):
        graph = self._build_psum_pair(both_dead=False)
        with pytest.raises(ValueError, match=r"psum"):
            graph.remove_nodes([0])  # leaves the negative half orphaned


class TestCoalescingGroupAtomicity:
    def _build_coalescing_pair(self):
        a = NeuralCore(
            id=0, name="A_master",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
            coalescing_group_id=7, coalescing_role="master",
        )
        b = NeuralCore(
            id=1, name="A_slave",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
            coalescing_group_id=7, coalescing_role="slave",
        )
        c = NeuralCore(
            id=2, name="C",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        return IRGraph(nodes=[a, b, c], output_sources=_src([(2, 0)]))

    def test_remove_nodes_atomic_per_coalescing_group_raises_on_partial_removal(self):
        graph = self._build_coalescing_pair()
        with pytest.raises(ValueError, match=r"coalescing"):
            graph.remove_nodes([1])

    def test_remove_nodes_atomic_per_coalescing_group_full_removal_ok(self):
        graph = self._build_coalescing_pair()
        graph.remove_nodes([0, 1])
        assert [n.id for n in graph.nodes] == [2]


class TestOrphanWeightBankCleanup:
    def _build_bank_graph(self):
        bank = WeightBank(
            id=11,
            core_matrix=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        )
        a = NeuralCore(
            id=0, name="A_bank",
            input_sources=_src([(-2, 0), (-2, 1)]),
            core_matrix=None, threshold=1.0, latency=0,
            weight_bank_id=11, weight_row_slice=(0, 1),
        )
        # B references the same bank -> bank still has consumers.
        b = NeuralCore(
            id=1, name="B_bank",
            input_sources=_src([(-2, 0), (-2, 1)]),
            core_matrix=None, threshold=1.0, latency=0,
            weight_bank_id=11, weight_row_slice=(1, 2),
        )
        # Output reaches B.
        return IRGraph(
            nodes=[a, b],
            output_sources=_src([(1, 0)]),
            weight_banks={11: bank},
        )

    def test_remove_nodes_keeps_bank_when_other_consumers_remain(self):
        graph = self._build_bank_graph()
        graph.remove_nodes([0])
        assert 11 in graph.weight_banks
        # B still references it.
        b = next(n for n in graph.nodes if n.id == 1)
        assert b.weight_bank_id == 11

    def test_remove_nodes_drops_orphan_bank_when_last_consumer_gone(self):
        # Build a graph where the bank has only one consumer.
        bank = WeightBank(
            id=99,
            core_matrix=np.array([[1.0]], dtype=np.float64),
        )
        a = NeuralCore(
            id=0, name="A_bank",
            input_sources=_src([(-2, 0)]),
            core_matrix=None, threshold=1.0, latency=0,
            weight_bank_id=99, weight_row_slice=(0, 1),
        )
        c = NeuralCore(
            id=1, name="C",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        graph = IRGraph(
            nodes=[a, c],
            output_sources=_src([(1, 0)]),
            weight_banks={99: bank},
        )
        graph.remove_nodes([0])
        assert 99 not in graph.weight_banks


class TestNoOpsAndIdempotence:
    def test_remove_nodes_empty_iterable_is_noop(self):
        graph = _two_core_chain()
        graph.remove_nodes([])
        assert [n.id for n in graph.nodes] == [0, 1]
        assert graph.validate() == []

    def test_remove_nodes_unknown_id_raises(self):
        """Defensive: removing an id not in the graph is a programmer error."""
        graph = _two_core_chain()
        with pytest.raises(ValueError, match=r"unknown|not in"):
            graph.remove_nodes([999])

    def test_remove_nodes_does_not_corrupt_input_for_compute_op_with_index(self):
        """A ComputeOp that referenced the removed node at a non-zero index
        should still produce well-formed off sources at the same axon slot."""
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0, 2.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        op = ComputeOp(
            id=1, name="op",
            input_sources=_src([(0, 1), (-3, 0)]),
            op_type="identity",
            input_shape=(2,), output_shape=(2,),
        )
        b = NeuralCore(
            id=2, name="B",
            input_sources=_src([(1, 0), (1, 1)]),
            core_matrix=np.array([[1.0], [1.0]], dtype=np.float64),
            threshold=1.0, latency=2,
        )
        c = NeuralCore(
            id=3, name="C",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        graph = IRGraph(
            nodes=[a, op, b, c],
            output_sources=_src([(2, 0), (3, 0)]),
        )
        graph.remove_nodes([0])
        op_node = next(n for n in graph.nodes if n.id == 1)
        flat = op_node.input_sources.flatten()
        assert flat[0].is_off()
        assert flat[1].is_always_on()
