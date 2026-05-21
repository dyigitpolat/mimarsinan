"""Tests for global (cross-core) pruning propagation.

`compute_global_pruned_sets` runs a bidirectional, recursive fixpoint over
the unified IR graph: pruning a neuron in core A propagates to consumer axons
in any neural core B (provided B is a NeuralCore — ComputeOps act as
barriers). Pruning all consumers of a neuron makes the neuron itself dead
unless it is in `output_sources`. Within-matrix propagation is delegated
to `compute_propagated_pruned_rows_cols`.

Exemptions:
- Axons fed by `IRSource(node_id=-2, ...)` (model input data) are never pruned.
- Neurons that appear in `IRGraph.output_sources` (model output logits) are
  never pruned.
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
from mimarsinan.mapping.pruning_graph_propagation import (
    compute_global_pruned_sets,
)


def _src(specs):
    return np.array(
        [IRSource(node_id=nid, index=idx) for nid, idx in specs],
        dtype=object,
    )


def _make_two_core_chain(
    *,
    w0: np.ndarray,
    w1: np.ndarray,
    output_indices: list[int] | None = None,
) -> IRGraph:
    """Build a minimal A -> B chain over the IR.

    A reads ``w0.shape[0] - 1`` inputs from model input (`-2`) plus a bias
    axon (``-3``); B reads each of A's neurons in order, plus a bias axon.
    Output sources take ``output_indices`` from B (default: all B neurons).
    """
    n_axons_a, n_neurons_a = w0.shape
    src0 = _src(
        [(-2, k) for k in range(n_axons_a - 1)] + [(-3, 0)]
    )
    a = NeuralCore(
        id=0, name="A", input_sources=src0, core_matrix=w0,
        threshold=1.0, latency=0,
    )
    n_axons_b, n_neurons_b = w1.shape
    if n_axons_b != n_neurons_a + 1:
        raise ValueError(
            "_make_two_core_chain expects B axons == A neurons + 1 (bias)"
        )
    src1 = _src([(0, j) for j in range(n_neurons_a)] + [(-3, 0)])
    b = NeuralCore(
        id=1, name="B", input_sources=src1, core_matrix=w1,
        threshold=1.0, latency=1,
    )
    if output_indices is None:
        output_indices = list(range(n_neurons_b))
    out_src = _src([(1, j) for j in output_indices])
    return IRGraph(nodes=[a, b], output_sources=out_src)


class TestComputeGlobalPrunedSetsAPI:
    """Contract for the `compute_global_pruned_sets` entry point."""

    def test_empty_graph_returns_empty_result(self):
        graph = IRGraph(nodes=[], output_sources=np.array([], dtype=object))
        res = compute_global_pruned_sets(
            graph,
            zero_threshold=1e-8,
            initial_per_node=None,
            initial_per_bank=None,
            exempt_rows_per_node={},
            exempt_cols_per_node={},
        )
        assert res.pruned_rows_per_node == {}
        assert res.pruned_cols_per_node == {}
        assert res.pruned_rows_per_bank == {}
        assert res.pruned_cols_per_bank == {}

    def test_returns_dicts_keyed_by_node_id(self):
        w0 = np.eye(3, 2, dtype=np.float64)
        w1 = np.array(
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            dtype=np.float64,
        )
        graph = _make_two_core_chain(w0=w0, w1=w1)
        res = compute_global_pruned_sets(
            graph,
            zero_threshold=1e-8,
            initial_per_node=None,
            initial_per_bank=None,
            exempt_rows_per_node={},
            exempt_cols_per_node={},
        )
        assert set(res.pruned_rows_per_node.keys()) == {0, 1}
        assert set(res.pruned_cols_per_node.keys()) == {0, 1}


class TestCrossCorePropagation:
    """Bidirectional, recursive pruning across NeuralCore boundaries."""

    def test_neuron_prune_propagates_to_consumer_axon(self):
        """Pruning A.col j flips B's axon reading (A,j) into the pruned set."""
        # A: 3 axons (2 inputs + bias), 2 neurons. Both columns have non-zero rows.
        w0 = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [0.5, 0.5],
            ],
            dtype=np.float64,
        )
        # B: 3 axons (2 from A + bias), 2 neurons.
        w1 = np.array(
            [
                [5.0, 6.0],
                [7.0, 8.0],
                [0.1, 0.1],
            ],
            dtype=np.float64,
        )
        graph = _make_two_core_chain(w0=w0, w1=w1)

        # Mark A.col 0 as initially pruned via the seed.
        res = compute_global_pruned_sets(
            graph,
            zero_threshold=1e-8,
            initial_per_node={0: (set(), {0})},
            initial_per_bank=None,
            exempt_rows_per_node={0: frozenset({0, 1}), 1: frozenset()},
            exempt_cols_per_node={0: frozenset(), 1: frozenset({0, 1})},
        )

        # Row 0 of B reads (A,0); since A.col 0 is pruned, B row 0 must be pruned.
        assert 0 in res.pruned_rows_per_node[1], (
            f"Expected B axon 0 to die (fed by pruned A neuron 0); got {res.pruned_rows_per_node[1]}"
        )
        # B row 1 reads (A,1) which is alive -> not pruned.
        assert 1 not in res.pruned_rows_per_node[1]

    def test_three_core_cascade_propagation(self):
        """A->B->C: A.col0 dead -> B.row0 dead -> B.col0 dies via within-matrix -> C.row0 dies."""
        # A: 3 axons, 2 neurons; col 0 will be seeded as pruned.
        w_a = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        # B: 3 axons (2 from A + bias), 2 neurons. Row 0 only contributes to col 0.
        # If row 0 dies, col 0 has only the bias row contribution -> all-zero -> col 0 dies.
        w_b = np.array(
            [
                [10.0, 0.0],
                [0.0, 11.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        # C: 3 axons (2 from B + bias), 2 neurons. Row 0 only feeds col 0.
        w_c = np.array(
            [
                [5.0, 0.0],
                [0.0, 6.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0), (-2, 1), (-3, 0)]),
            core_matrix=w_a, threshold=1.0, latency=0,
        )
        b = NeuralCore(
            id=1, name="B",
            input_sources=_src([(0, 0), (0, 1), (-3, 0)]),
            core_matrix=w_b, threshold=1.0, latency=1,
        )
        c = NeuralCore(
            id=2, name="C",
            input_sources=_src([(1, 0), (1, 1), (-3, 0)]),
            core_matrix=w_c, threshold=1.0, latency=2,
        )
        graph = IRGraph(
            nodes=[a, b, c],
            output_sources=_src([(2, 0), (2, 1)]),
        )
        res = compute_global_pruned_sets(
            graph,
            zero_threshold=1e-8,
            initial_per_node={0: (set(), {0})},
            initial_per_bank=None,
            exempt_rows_per_node={
                0: frozenset({0, 1}), 1: frozenset(), 2: frozenset(),
            },
            exempt_cols_per_node={
                0: frozenset(), 1: frozenset(), 2: frozenset({0, 1}),
            },
        )
        # A.col 0 stays in.
        assert 0 in res.pruned_cols_per_node[0]
        # B.row 0 dies because (A,0) is dead.
        assert 0 in res.pruned_rows_per_node[1]
        # B.col 0 dies because its only non-bias row (row 0) is now dead.
        assert 0 in res.pruned_cols_per_node[1]
        # C.row 0 dies because (B,0) is dead.
        assert 0 in res.pruned_rows_per_node[2]

    def test_unconsumed_neuron_is_pruned(self):
        """A.col j with no surviving consumer axon and not in output_sources is pruned."""
        # A has 3 neurons; only neurons 0 and 1 are read by B and outputs reference only B.
        w_a = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        w_b = np.array([[1.0], [1.0], [1.0]], dtype=np.float64)
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0), (-3, 0)]),
            core_matrix=w_a, threshold=1.0, latency=0,
        )
        # B reads A.0, A.1 (no one reads A.2)
        b = NeuralCore(
            id=1, name="B",
            input_sources=_src([(0, 0), (0, 1), (-3, 0)]),
            core_matrix=w_b, threshold=1.0, latency=1,
        )
        graph = IRGraph(nodes=[a, b], output_sources=_src([(1, 0)]))
        res = compute_global_pruned_sets(
            graph,
            zero_threshold=1e-8,
            initial_per_node=None,
            initial_per_bank=None,
            exempt_rows_per_node={0: frozenset({0}), 1: frozenset()},
            exempt_cols_per_node={0: frozenset(), 1: frozenset({0})},
        )
        assert 2 in res.pruned_cols_per_node[0], (
            f"A.col 2 has no consumer and is not in output_sources; should be pruned. "
            f"Got {res.pruned_cols_per_node[0]}"
        )
        assert 0 not in res.pruned_cols_per_node[0]
        assert 1 not in res.pruned_cols_per_node[0]


class TestComputeOpBarrier:
    """ComputeOps act as opaque consumers/producers and block cross-core propagation."""

    def test_compute_op_blocks_neuron_to_axon_propagation(self):
        """A -> ComputeOp -> B: pruning A.col 0 must NOT cascade to B's axons."""
        w_a = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float64,
        )
        op_src = _src([(0, 0), (0, 1)])
        op = ComputeOp(
            id=1, name="op", input_sources=op_src, op_type="identity",
        )
        w_b = np.array(
            [
                [5.0, 6.0],
                [7.0, 8.0],
                [0.1, 0.1],
            ],
            dtype=np.float64,
        )
        b = NeuralCore(
            id=2, name="B",
            input_sources=_src([(1, 0), (1, 1), (-3, 0)]),
            core_matrix=w_b, threshold=1.0, latency=1,
        )
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0), (-2, 1)]),
            core_matrix=w_a, threshold=1.0, latency=0,
        )
        graph = IRGraph(
            nodes=[a, op, b],
            output_sources=_src([(2, 0), (2, 1)]),
        )
        res = compute_global_pruned_sets(
            graph,
            zero_threshold=1e-8,
            initial_per_node={0: (set(), {0})},
            initial_per_bank=None,
            exempt_rows_per_node={0: frozenset({0, 1}), 2: frozenset()},
            exempt_cols_per_node={0: frozenset(), 2: frozenset({0, 1})},
        )
        # A.col 0 still pruned.
        assert 0 in res.pruned_cols_per_node[0]
        # ComputeOp barrier: B's axons unaffected.
        assert res.pruned_rows_per_node[2] == set(), (
            f"ComputeOp barrier should block cross-core propagation; "
            f"got pruned axons {res.pruned_rows_per_node[2]} in B"
        )

    def test_compute_op_keeps_producer_neurons_alive(self):
        """A neuron consumed only by a ComputeOp is considered to have a live consumer."""
        # A.col 0 has no NeuralCore consumer, only a ComputeOp.
        w_a = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        op_src = _src([(0, 0)])
        op = ComputeOp(
            id=1, name="op", input_sources=op_src, op_type="identity",
        )
        # B's input is from the ComputeOp, not directly from A.
        w_b = np.array([[1.0], [1.0]], dtype=np.float64)
        b = NeuralCore(
            id=2, name="B",
            input_sources=_src([(1, 0), (-3, 0)]),
            core_matrix=w_b, threshold=1.0, latency=1,
        )
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0), (-3, 0)]),
            core_matrix=w_a, threshold=1.0, latency=0,
        )
        graph = IRGraph(
            nodes=[a, op, b], output_sources=_src([(2, 0)]),
        )
        res = compute_global_pruned_sets(
            graph,
            zero_threshold=1e-8,
            initial_per_node=None,
            initial_per_bank=None,
            exempt_rows_per_node={0: frozenset({0}), 2: frozenset()},
            exempt_cols_per_node={0: frozenset(), 2: frozenset({0})},
        )
        # A.col 1 has no consumer at all (not in output_sources, not consumed) -> pruned.
        assert 1 in res.pruned_cols_per_node[0]
        # A.col 0 is consumed by ComputeOp -> kept alive.
        assert 0 not in res.pruned_cols_per_node[0]


class TestExemptions:
    """Model-level input-data axons and output-logit neurons must never be pruned."""

    def test_model_input_axon_never_pruned_even_when_zero(self):
        # All zeros, but axon 0 is fed from -2 (model input). Exempt rule keeps it.
        w = np.zeros((2, 1), dtype=np.float64)
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0), (-3, 0)]),
            core_matrix=w, threshold=1.0, latency=0,
        )
        graph = IRGraph(
            nodes=[a], output_sources=_src([(0, 0)]),
        )
        res = compute_global_pruned_sets(
            graph,
            zero_threshold=1e-8,
            initial_per_node=None,
            initial_per_bank=None,
            exempt_rows_per_node={0: frozenset({0})},
            exempt_cols_per_node={0: frozenset({0})},
        )
        assert 0 not in res.pruned_rows_per_node[0]

    def test_model_output_neuron_never_pruned_even_when_zero(self):
        w = np.zeros((2, 1), dtype=np.float64)
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0), (-3, 0)]),
            core_matrix=w, threshold=1.0, latency=0,
        )
        graph = IRGraph(
            nodes=[a], output_sources=_src([(0, 0)]),
        )
        res = compute_global_pruned_sets(
            graph,
            zero_threshold=1e-8,
            initial_per_node=None,
            initial_per_bank=None,
            exempt_rows_per_node={0: frozenset({0})},
            exempt_cols_per_node={0: frozenset({0})},
        )
        assert 0 not in res.pruned_cols_per_node[0]

    def test_seeded_pruned_index_with_exemption_stays_alive(self):
        """An exempt index must not be pruned even if seeded."""
        w = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0), (-3, 0)]),
            core_matrix=w, threshold=1.0, latency=0,
        )
        graph = IRGraph(
            nodes=[a], output_sources=_src([(0, 0), (0, 1)]),
        )
        res = compute_global_pruned_sets(
            graph,
            zero_threshold=1e-8,
            initial_per_node={0: ({0}, {0})},  # tries to prune exempt indices
            initial_per_bank=None,
            exempt_rows_per_node={0: frozenset({0})},
            exempt_cols_per_node={0: frozenset({0, 1})},
        )
        assert 0 not in res.pruned_rows_per_node[0]
        assert 0 not in res.pruned_cols_per_node[0]


class TestWeightBankPropagation:
    """Cross-core propagation through bank-backed cores must work over bank coordinates."""

    def test_bank_neuron_with_no_consumer_is_pruned(self):
        # A 4-neuron bank shared by a single node using the full slice.
        bank = WeightBank(
            id=0,
            core_matrix=np.array(
                [
                    [1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
        )
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0), (-2, 1), (-3, 0)]),
            core_matrix=None,
            weight_bank_id=0,
            weight_row_slice=(0, 4),
            threshold=1.0, latency=0,
        )
        # B reads only A.0, A.1, A.2, leaving A.3 with no consumer.
        w_b = np.array(
            [[1.0], [1.0], [1.0], [1.0]], dtype=np.float64,
        )
        b = NeuralCore(
            id=1, name="B",
            input_sources=_src([(0, 0), (0, 1), (0, 2), (-3, 0)]),
            core_matrix=w_b, threshold=1.0, latency=1,
        )
        graph = IRGraph(
            nodes=[a, b],
            output_sources=_src([(1, 0)]),
            weight_banks={0: bank},
        )
        res = compute_global_pruned_sets(
            graph,
            zero_threshold=1e-8,
            initial_per_node=None,
            initial_per_bank=None,
            exempt_rows_per_node={
                0: frozenset({0, 1}), 1: frozenset(),
            },
            exempt_cols_per_node={
                0: frozenset(), 1: frozenset({0}),
            },
        )
        # A.col 3 has no consumer -> pruned in the bank coordinate.
        assert 3 in res.pruned_cols_per_bank[0]
        assert 3 in res.pruned_cols_per_node[0]


class TestFixpointTermination:
    """Even worst-case dense connectivity must converge in finite iterations."""

    def test_dense_chain_terminates(self):
        rng = np.random.default_rng(0)
        sizes = [(5, 4), (5, 4), (5, 4)]
        nodes: list[NeuralCore] = []
        for i, (axons, neurons) in enumerate(sizes):
            mat = (rng.standard_normal((axons, neurons)) * 0.1).astype(np.float64)
            if i == 0:
                src = _src([(-2, k) for k in range(axons - 1)] + [(-3, 0)])
            else:
                src = _src(
                    [(i - 1, k) for k in range(axons - 1)] + [(-3, 0)]
                )
            nodes.append(
                NeuralCore(
                    id=i, name=f"core_{i}",
                    input_sources=src, core_matrix=mat,
                    threshold=1.0, latency=i,
                )
            )
        graph = IRGraph(
            nodes=nodes,
            output_sources=_src([(2, j) for j in range(sizes[-1][1])]),
        )
        # Should not hang or raise.
        res = compute_global_pruned_sets(
            graph,
            zero_threshold=1e-8,
            initial_per_node=None,
            initial_per_bank=None,
            exempt_rows_per_node={
                0: frozenset({0, 1, 2, 3}),
                1: frozenset(),
                2: frozenset(),
            },
            exempt_cols_per_node={
                0: frozenset(),
                1: frozenset(),
                2: frozenset({0, 1, 2, 3}),
            },
        )
        for nid in (0, 1, 2):
            assert nid in res.pruned_rows_per_node
            assert nid in res.pruned_cols_per_node
