"""W3 elimination ledger: per-kill attribution, propagation depth, reconciliation.

The ledger turns the cascade from "it runs" into "it is attributable"
(paper claims C1/P2; falsifiers F3' and F6): every eliminated row/column is
attributed to SEED vs CLOSURE-COUPLING vs EMERGENT-PROPAGATION vs
liveness-DEAD, carries its propagation depth (seed = 0; a kill caused by
depth-d structure is d+1), and the totals must reconcile against the
compacted shapes.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.pruning.elimination_ledger import (
    EliminationLedger,
    EliminationLedgerError,
    compute_elimination_arms,
    compute_elimination_ledger,
)
from mimarsinan.mapping.pruning.graph.propagation_mode import (
    ELIMINATION_PROPAGATION_CASCADE,
    ELIMINATION_PROPAGATION_CLOSURE,
    ELIMINATION_PROPAGATION_MASKED,
)
from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph

MASKED = ELIMINATION_PROPAGATION_MASKED
CLOSURE = ELIMINATION_PROPAGATION_CLOSURE
CASCADE = ELIMINATION_PROPAGATION_CASCADE


def _src(specs):
    return np.array(
        [IRSource(node_id=nid, index=idx) for nid, idx in specs],
        dtype=object,
    )


def make_emergent_chain():
    """A -> B -> C: seed A.col0 (d0) -> B.row0 (d1, coupling) -> B.col0
    (d2, emergent starvation) -> C.row0 (d3, emergent)."""
    w_a = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 1.0]], dtype=np.float64)
    w_b = np.array([[10.0, 0.0], [0.0, 11.0], [0.0, 1.0]], dtype=np.float64)
    w_c = np.array([[5.0, 0.0], [0.0, 6.0], [0.0, 1.0]], dtype=np.float64)
    a = NeuralCore(
        id=0, name="A", input_sources=_src([(-2, 0), (-2, 1), (-3, 0)]),
        core_matrix=w_a, threshold=1.0, latency=0,
    )
    b = NeuralCore(
        id=1, name="B", input_sources=_src([(0, 0), (0, 1), (-3, 0)]),
        core_matrix=w_b, threshold=1.0, latency=1,
    )
    c = NeuralCore(
        id=2, name="C", input_sources=_src([(1, 0), (1, 1), (-3, 0)]),
        core_matrix=w_c, threshold=1.0, latency=2,
    )
    graph = IRGraph(nodes=[a, b, c], output_sources=_src([(2, 0), (2, 1)]))
    seeds = {0: ([False, False, False], [True, False])}
    return graph, seeds


def _record_by_node(ledger: EliminationLedger, node_id: int):
    return next(r for r in ledger.per_node if r.node_id == node_id)


class TestDepthAndAttributionOnTheChain:
    def _ledger(self):
        graph, seeds = make_emergent_chain()
        return compute_elimination_ledger(
            graph, initial_pruned_per_node=seeds,
        )

    def test_exact_depths_on_the_hand_built_chain(self):
        ledger = self._ledger()
        a, b, c = (_record_by_node(ledger, i) for i in (0, 1, 2))
        assert a.col_depths == {0: 0}, "seed kill has depth 0"
        assert b.row_depths == {0: 1}, "consumer axon coupling is depth 1"
        assert b.col_depths == {0: 2}, "starved neuron is depth 2"
        assert c.row_depths == {0: 3}, "freed consumer row is depth 3"
        assert ledger.max_propagation_depth == 3

    def test_attribution_categories(self):
        ledger = self._ledger()
        a, b, c = (_record_by_node(ledger, i) for i in (0, 1, 2))
        assert a.counts.seed_cols == 1
        assert b.counts.closure_rows == 1
        assert b.counts.emergent_cols == 1
        assert c.counts.emergent_rows == 1
        assert ledger.seed_cols == 1
        assert ledger.closure_coupling_rows == 1
        assert ledger.emergent_propagation_cols == 1
        assert ledger.emergent_propagation_rows == 1
        assert ledger.liveness_dead_rows == 0
        assert ledger.liveness_dead_cols == 0

    def test_no_cores_deleted_no_bias_only_on_the_chain(self):
        ledger = self._ledger()
        assert ledger.cores_deleted == 0
        assert ledger.bias_only_collapses == 0

    def test_fixpoint_iteration_count_recorded(self):
        ledger = self._ledger()
        assert ledger.fixpoint_iterations >= 2

    def test_flat_dict_row(self):
        d = self._ledger().to_dict()
        assert d["mode"] == "cascade"
        assert d["seed_cols"] == 1
        assert d["closure_coupling_rows"] == 1
        assert d["emergent_propagation_rows"] == 1
        assert d["emergent_propagation_cols"] == 1
        assert d["max_propagation_depth"] == 3
        assert d["cores_deleted"] == 0
        assert d["bias_only_collapses"] == 0
        assert all(not isinstance(v, (list, dict, tuple)) for v in d.values())

    def test_summary_is_one_line(self):
        s = self._ledger().summary()
        assert "\n" not in s
        assert "cascade" in s


class TestModeRestrictedLedgers:
    def test_masked_ledger_has_only_seed_kills_at_depth_zero(self):
        graph, seeds = make_emergent_chain()
        ledger = compute_elimination_ledger(
            graph, initial_pruned_per_node=seeds,
            elimination_propagation=MASKED,
        )
        assert ledger.mode == MASKED
        assert ledger.seed_cols == 1
        assert ledger.closure_coupling_rows == 0
        assert ledger.emergent_propagation_rows == 0
        assert ledger.emergent_propagation_cols == 0
        assert ledger.max_propagation_depth == 0
        assert ledger.fixpoint_iterations == 0

    def test_closure_ledger_stops_at_depth_one(self):
        graph, seeds = make_emergent_chain()
        ledger = compute_elimination_ledger(
            graph, initial_pruned_per_node=seeds,
            elimination_propagation=CLOSURE,
        )
        assert ledger.mode == CLOSURE
        assert ledger.seed_cols == 1
        assert ledger.closure_coupling_rows == 1
        assert ledger.emergent_propagation_rows == 0
        assert ledger.emergent_propagation_cols == 0
        assert ledger.max_propagation_depth == 1


class TestReconciliationAgainstCompactedShapes:
    @pytest.mark.parametrize("mode", [MASKED, CLOSURE, CASCADE])
    def test_ledger_totals_reconcile_with_compaction(self, mode):
        graph, seeds = make_emergent_chain()
        ledger = compute_elimination_ledger(
            graph, initial_pruned_per_node=seeds,
            elimination_propagation=mode,
        )
        pruned = prune_ir_graph(
            copy.deepcopy(graph), initial_pruned_per_node=seeds,
            elimination_propagation=mode,
        )
        surviving = {n.id: n for n in pruned.nodes if isinstance(n, NeuralCore)}
        for record in ledger.per_node:
            if record.node_id not in surviving:
                continue
            node = surviving[record.node_id]
            killed_rows = record.counts.total_rows
            killed_cols = record.counts.total_cols
            expected_rows = record.n_axons - killed_rows
            if expected_rows == 0:
                expected_rows = 1  # BIAS_ONLY placeholder row
            assert node.core_matrix.shape == (
                expected_rows, record.n_neurons - killed_cols
            ), f"node {record.node_id} shape does not reconcile"
        deleted = {r.node_id for r in ledger.per_node} - set(surviving)
        assert len(deleted) == ledger.cores_deleted

    def test_pre_compaction_mask_seam_matches_ledger(self):
        """The ledger builds on the pre-compaction metadata seam: mask
        population counts must equal the ledger's per-node totals."""
        graph, seeds = make_emergent_chain()
        ledger = compute_elimination_ledger(
            graph, initial_pruned_per_node=seeds,
        )
        pruned = prune_ir_graph(
            copy.deepcopy(graph), initial_pruned_per_node=seeds,
            store_heatmap=True,
        )
        for node in pruned.nodes:
            if not isinstance(node, NeuralCore):
                continue
            record = _record_by_node(ledger, node.id)
            assert sum(node.pre_pruning_row_mask) == record.counts.total_rows
            assert sum(node.pre_pruning_col_mask) == record.counts.total_cols


class TestDeletedCoresAndLiveness:
    def test_fully_starved_cores_are_counted_deleted(self):
        """Seeding both A columns starves all of B; A and B are DEAD and
        deleted, C survives on its own data axon."""
        w_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        w_b = np.array([[10.0, 1.0], [1.0, 11.0]], dtype=np.float64)
        w_c = np.array([[5.0, 0.0], [0.0, 6.0], [1.0, 1.0]], dtype=np.float64)
        a = NeuralCore(
            id=0, name="A", input_sources=_src([(-2, 0), (-2, 1)]),
            core_matrix=w_a, threshold=1.0, latency=0,
        )
        b = NeuralCore(
            id=1, name="B", input_sources=_src([(0, 0), (0, 1)]),
            core_matrix=w_b, threshold=1.0, latency=1,
        )
        c = NeuralCore(
            id=2, name="C", input_sources=_src([(1, 0), (1, 1), (-2, 2)]),
            core_matrix=w_c, threshold=1.0, latency=2,
        )
        graph = IRGraph(nodes=[a, b, c], output_sources=_src([(2, 0), (2, 1)]))
        seeds = {0: ([False, False], [True, True])}
        ledger = compute_elimination_ledger(
            graph, initial_pruned_per_node=seeds,
        )
        assert ledger.cores_deleted == 2
        pruned = prune_ir_graph(
            copy.deepcopy(graph), initial_pruned_per_node=seeds,
        )
        assert {n.id for n in pruned.nodes} == {2}

    def test_bias_only_collapse_is_counted(self):
        """All of B's axons die but its non-zero bias keeps neurons alive
        under mvm semantics: B collapses to BIAS_ONLY, not DEAD."""
        w_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        w_b = np.array([[10.0, 1.0], [1.0, 11.0]], dtype=np.float64)
        w_c = np.array([[5.0, 0.0], [0.0, 6.0], [1.0, 1.0]], dtype=np.float64)
        a = NeuralCore(
            id=0, name="A", input_sources=_src([(-2, 0), (-2, 1)]),
            core_matrix=w_a, threshold=1.0, latency=0,
        )
        b = NeuralCore(
            id=1, name="B", input_sources=_src([(0, 0), (0, 1)]),
            core_matrix=w_b, threshold=1.0, latency=1,
            hardware_bias=np.array([0.5, 0.5]),
        )
        c = NeuralCore(
            id=2, name="C", input_sources=_src([(1, 0), (1, 1), (-2, 2)]),
            core_matrix=w_c, threshold=1.0, latency=2,
        )
        graph = IRGraph(nodes=[a, b, c], output_sources=_src([(2, 0), (2, 1)]))
        seeds = {0: ([False, False], [True, True])}
        ledger = compute_elimination_ledger(
            graph, initial_pruned_per_node=seeds,
            spiking_mode=INERT_SPIKING_MODE,
        )
        assert ledger.bias_only_collapses == 1
        assert ledger.cores_deleted == 1  # only A dies outright


class TestBankLedger:
    def _bank_graph(self):
        rng = np.random.default_rng(13)
        bank = WeightBank(
            id=0, core_matrix=rng.standard_normal((5, 4)).astype(np.float64)
        )
        nodes = []
        for tok in range(3):
            srcs = _src([(-2, tok * 4 + i) for i in range(4)] + [(-3, 0)])
            nodes.append(NeuralCore(
                id=tok, name=f"tok{tok}", input_sources=srcs,
                core_matrix=None, weight_bank_id=0, weight_row_slice=(0, 4),
                threshold=1.0, latency=0,
            ))
        head = NeuralCore(
            id=3, name="head",
            input_sources=_src(
                [(tok, j) for tok in range(3) for j in range(4)] + [(-3, 0)]
            ),
            core_matrix=rng.standard_normal((13, 2)).astype(np.float64),
            threshold=1.0, latency=1,
        )
        return IRGraph(
            nodes=nodes + [head],
            output_sources=_src([(3, 0), (3, 1)]),
            weight_banks={0: bank},
        )

    def test_per_bank_record_and_mask_seam_reconcile(self):
        graph = self._bank_graph()
        seeds = {0: ([False] * 5, [j == 1 for j in range(4)])}
        ledger = compute_elimination_ledger(
            graph, initial_pruned_per_bank=seeds,
        )
        bank_rec = next(r for r in ledger.per_bank if r.bank_id == 0)
        assert bank_rec.counts.seed_cols == 1
        assert bank_rec.counts.total_cols == 1
        pruned = prune_ir_graph(
            copy.deepcopy(graph), initial_pruned_per_bank=seeds,
        )
        tok = next(n for n in pruned.nodes if n.id == 0)
        assert sum(tok.pruned_col_mask) == bank_rec.counts.total_cols

    def test_head_rows_attributed_to_closure_coupling(self):
        graph = self._bank_graph()
        seeds = {0: ([False] * 5, [j == 1 for j in range(4)])}
        ledger = compute_elimination_ledger(
            graph, initial_pruned_per_bank=seeds,
        )
        head = _record_by_node(ledger, 3)
        assert head.counts.closure_rows == 3, (
            "one dead axon per token instance, each one hop from the seed"
        )


class TestLedgerContract:
    def test_empty_graph_yields_empty_ledger(self):
        graph = IRGraph(nodes=[], output_sources=np.array([], dtype=object))
        ledger = compute_elimination_ledger(graph)
        assert ledger.per_node == ()
        assert ledger.cores_deleted == 0
        assert ledger.to_dict()["total_rows_eliminated"] == 0

    def test_unknown_mode_rejected(self):
        graph, seeds = make_emergent_chain()
        with pytest.raises(ValueError, match="bogus"):
            compute_elimination_ledger(
                graph, initial_pruned_per_node=seeds,
                elimination_propagation="bogus",
            )


class TestPrecomputedArmsConsumeTheirOwnInputs:
    """[W6c] Handing the ledger a precomputed :class:`EliminationArms` used to
    make it SILENTLY IGNORE its own ``zero_threshold`` /
    ``initial_pruned_per_*`` — the arms had already consumed theirs, and the
    ledger's defaults leaked into the liveness pass. Every such input is now
    either consumed from the arms or refused."""

    def _arms(self, **kwargs):
        graph, seeds = make_emergent_chain()
        return graph, seeds, compute_elimination_arms(
            graph, initial_pruned_per_node=seeds, **kwargs
        )

    def test_a_contradicting_zero_threshold_fails_loud(self):
        graph, _, arms = self._arms(zero_threshold=1e-3)
        with pytest.raises(EliminationLedgerError, match="zero_threshold"):
            compute_elimination_ledger(graph, zero_threshold=1e-8, arms=arms)

    def test_a_contradicting_spiking_mode_fails_loud(self):
        graph, _, arms = self._arms()
        with pytest.raises(EliminationLedgerError, match="spiking_mode"):
            compute_elimination_ledger(graph, spiking_mode="if", arms=arms)

    def test_a_second_seed_set_fails_loud(self):
        graph, seeds, arms = self._arms()
        with pytest.raises(
            EliminationLedgerError, match="initial_pruned_per_node"
        ):
            compute_elimination_ledger(
                graph, initial_pruned_per_node=seeds, arms=arms
            )

    def test_an_echoed_input_is_accepted(self):
        """Restating exactly what the arms ran with is not a contradiction."""
        graph, _, arms = self._arms(zero_threshold=1e-3)
        ledger = compute_elimination_ledger(graph, zero_threshold=1e-3, arms=arms)
        assert ledger.per_node

    def test_the_arms_threshold_is_the_one_actually_used(self):
        """Same arms either way: passing them must not change one number."""
        graph, seeds, arms = self._arms(zero_threshold=1e-3)
        shared = compute_elimination_ledger(graph, arms=arms)
        standalone = compute_elimination_ledger(
            graph, initial_pruned_per_node=seeds, zero_threshold=1e-3,
        )
        assert shared.to_dict() == standalone.to_dict()
