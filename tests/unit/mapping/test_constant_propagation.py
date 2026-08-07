"""[W4b-2] graph-level constant propagation: arms, folds, certificate, ledger.

The three DoD barriers the zero-only cascade cannot cross are pinned here as
hand-built dyadic vehicles: a SIGMOID chain (act(0) != 0), a RESIDUAL JOIN
(multi-input, opaque), and a BIAS_ONLY 1x1 core (a pure constant producer).
Each is measured off-vs-full, certified bit-exactly, and mutated to prove the
certificate can actually fail.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.pruning.certificate import (
    CascadeCertificateError,
    CascadeCertificatePreconditionError,
    certify_cascade_equivalence,
)
from mimarsinan.mapping.pruning.elimination_ledger import (
    compute_elimination_ledger,
)
from mimarsinan.mapping.pruning.graph import (
    compute_global_pruned_sets,
    constant_line_values,
    reanalyze_constant_folding,
)
from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph
from mimarsinan.mapping.pruning.ir_pruning_helpers import (
    _boundary_policy_exemptions,
)
from mimarsinan.mapping.pruning.liveness_transfer import (
    COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY,
)

from unit.mapping.constant_vehicles import (
    bias_only_collapse_graph,
    gelu_execution_exact_graph,
    residual_join_graph,
    sigmoid_chain_graph,
)

STEM_DEAD = {0: ([False] * 3, [True] * 4)}


def _arm(graph, *, folding, seeds=None, mode="cascade", **kw):
    exempt_rows, exempt_cols = _boundary_policy_exemptions(graph)
    return compute_global_pruned_sets(
        graph,
        initial_per_node=seeds,
        exempt_rows_per_node=exempt_rows,
        exempt_cols_per_node=exempt_cols,
        mode=mode,
        elimination_constant_folding=folding,
        spiking_mode=kw.pop("spiking_mode", INERT_SPIKING_MODE),
        **kw,
    )


def _kills(result) -> int:
    return (
        sum(len(s) for s in result.pruned_rows_per_node.values())
        + sum(len(s) for s in result.pruned_cols_per_node.values())
        + sum(len(s) for s in result.pruned_rows_per_bank.values())
        + sum(len(s) for s in result.pruned_cols_per_bank.values())
    )


def _seed_sets(graph, seeds):
    return {nid: ({i for i, m in enumerate(r) if m},
                  {j for j, m in enumerate(c) if m})
            for nid, (r, c) in (seeds or {}).items()}


class TestSigmoidBarrierNowPropagates:
    """sigmoid(0) = 0.5 — the classic non-zero-preserving barrier."""

    def _run(self, folding):
        graph = sigmoid_chain_graph()
        return _arm(
            graph, folding=folding, seeds=_seed_sets(graph, STEM_DEAD)
        )

    def test_zero_only_cascade_cannot_cross_the_sigmoid(self):
        result = self._run("off")
        assert result.pruned_rows_per_node[2] == set(), (
            "W4b-1 leaves a non-zero-preserving activation opaque"
        )

    def test_constant_fold_kills_every_consumer_row(self):
        result = self._run("full")
        assert result.pruned_rows_per_node[2] == {0, 1, 2, 3}
        assert result.constant_folds.folded_rows[2] == {
            j: 0.5 for j in range(4)
        }

    def test_the_folded_program_is_bit_exact(self):
        report = certify_cascade_equivalence(
            sigmoid_chain_graph(), initial_pruned_per_node=STEM_DEAD,
            batches=3, batch_size=8,
        )
        assert report.passed is True
        assert report.max_abs_delta == 0.0
        assert report.pruned_cells < report.reference_cells

    def test_without_folding_the_certificate_still_refuses_the_sigmoid(self):
        with pytest.raises(
            CascadeCertificatePreconditionError, match="Sigmoid"
        ):
            certify_cascade_equivalence(
                sigmoid_chain_graph(), initial_pruned_per_node=STEM_DEAD,
                batches=1, batch_size=2, elimination_constant_folding="off",
            )


class TestResidualJoinFolds:
    """A multi-input join is opaque in both directions until it is constant."""

    def _run(self, folding):
        graph = residual_join_graph()
        return _arm(
            graph, folding=folding, seeds=_seed_sets(graph, STEM_DEAD)
        )

    def test_zero_only_cascade_stops_at_the_join(self):
        assert self._run("off").pruned_rows_per_node[3] == set()

    def test_join_folds_when_every_branch_is_constant(self):
        result = self._run("full")
        assert result.pruned_rows_per_node[3] == {0, 1, 2, 3}
        assert set(result.constant_folds.folded_rows[3]) == {0, 1, 2, 3}

    def test_the_folded_join_is_bit_exact(self):
        report = certify_cascade_equivalence(
            residual_join_graph(), initial_pruned_per_node=STEM_DEAD,
            batches=3, batch_size=8,
        )
        assert report.passed is True
        assert report.pruned_cells < report.reference_cells


class TestBiasOnlyCollapse:
    """The PI's BIAS_ONLY enhancement falls out of the lattice, unaided."""

    def test_the_bias_core_constant_folds_into_its_consumer(self):
        result = _arm(bias_only_collapse_graph(), folding="full")
        assert result.constant_folds.folded_rows[2] == {0: 0.25}
        assert 0 in result.pruned_cols_per_node[0], (
            "the bias core's only neuron is orphaned once the fold frees "
            "its reader"
        )

    def test_zero_only_cascade_keeps_the_bias_core_alive(self):
        result = _arm(bias_only_collapse_graph(), folding="off")
        assert result.pruned_rows_per_node[2] == set()
        assert result.pruned_cols_per_node[0] == set()

    def test_prune_ir_graph_deletes_the_collapsed_core(self):
        graph = bias_only_collapse_graph()
        prune_ir_graph(graph, spiking_mode=INERT_SPIKING_MODE)
        assert 0 not in {n.id for n in graph.nodes}
        head = next(n for n in graph.nodes if n.id == 2)
        assert head.core_matrix.shape == (3, 2), (
            "the freed axon row is compacted away too"
        )

    def test_the_collapse_is_bit_exact(self):
        report = certify_cascade_equivalence(
            bias_only_collapse_graph(), batches=3, batch_size=8,
        )
        assert report.passed is True
        assert report.pruned_cells < report.reference_cells


class TestArmSemantics:
    """masked never folds; closure folds one hop; cascade folds to fixpoint."""

    def _arms(self, make, seeds):
        graph = make()
        return {
            mode: _arm(
                graph, folding="full", seeds=_seed_sets(graph, seeds),
                mode=mode,
            )
            for mode in ("masked", "closure", "cascade")
        }

    def test_masked_never_folds(self):
        arms = self._arms(sigmoid_chain_graph, STEM_DEAD)
        assert arms["masked"].constant_folds.folded_rows == {}
        assert constant_line_values(arms["masked"]) == {}

    def test_closure_folds_exactly_one_hop(self):
        arms = self._arms(sigmoid_chain_graph, STEM_DEAD)
        assert set(arms["closure"].constant_folds.folded_rows[2]) == {
            0, 1, 2, 3
        }

    def test_closure_does_not_discover_constant_columns(self):
        """A column becoming constant is EMERGENT deadness; closure must not
        find it, so only the cascade records core-output CONST lines."""
        arms = self._arms(bias_only_collapse_graph, None)
        assert constant_line_values(arms["closure"]) == {}
        assert (0, 0) in constant_line_values(arms["cascade"])

    @pytest.mark.parametrize(
        "make,seeds",
        [
            (sigmoid_chain_graph, STEM_DEAD),
            (residual_join_graph, STEM_DEAD),
            (bias_only_collapse_graph, None),
        ],
    )
    def test_masked_subseteq_closure_subseteq_cascade_setwise(self, make, seeds):
        arms = self._arms(make, seeds)
        for attr in (
            "pruned_rows_per_node", "pruned_cols_per_node",
            "pruned_rows_per_bank", "pruned_cols_per_bank",
        ):
            for key, lo in getattr(arms["masked"], attr).items():
                assert lo <= getattr(arms["closure"], attr).get(key, set())
            for key, lo in getattr(arms["closure"], attr).items():
                assert lo <= getattr(arms["cascade"], attr).get(key, set())

    @pytest.mark.parametrize(
        "make,seeds",
        [
            (sigmoid_chain_graph, STEM_DEAD),
            (residual_join_graph, STEM_DEAD),
            (bias_only_collapse_graph, None),
        ],
    )
    def test_const_lines_are_ordered_setwise_too(self, make, seeds):
        """CONST lines compare as a SET of (port, value) pairs: descent is
        one-way, so a weaker arm's lines must appear verbatim in a stronger
        arm's — same port AND same value, never a re-valued line."""
        arms = self._arms(make, seeds)
        masked = set(constant_line_values(arms["masked"]).items())
        closure = set(constant_line_values(arms["closure"]).items())
        cascade = set(constant_line_values(arms["cascade"]).items())
        assert masked <= closure <= cascade


class TestKillSwitchesAreByteIdentical:
    @pytest.mark.parametrize(
        "make,seeds",
        [
            (sigmoid_chain_graph, STEM_DEAD),
            (residual_join_graph, STEM_DEAD),
            (bias_only_collapse_graph, None),
            (gelu_execution_exact_graph, None),
        ],
    )
    def test_off_reproduces_the_zero_only_cascade(self, make, seeds):
        graph = make()
        before = copy.deepcopy(graph)
        off = _arm(graph, folding="off", seeds=_seed_sets(graph, seeds))
        assert off.constant_folds.folded_rows == {}
        assert off.constant_folds.deltas == {}
        for a, b in zip(before.nodes, graph.nodes):
            if isinstance(a, NeuralCore) and a.core_matrix is not None:
                assert np.array_equal(a.core_matrix, b.core_matrix)

    def test_identity_only_transfers_disable_folding(self):
        graph = sigmoid_chain_graph()
        result = _arm(
            graph, folding="full", seeds=_seed_sets(graph, STEM_DEAD),
            computeop_liveness_transfers=(
                COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY
            ),
        )
        assert result.constant_folds.folded_rows == {}
        assert result.pruned_rows_per_node[2] == set()

    def test_spiking_domain_folds_zeros_only(self):
        """CONST(0) is exact everywhere; CONST(c != 0) is not, so a LIF
        deployment folds the join's zeros and refuses sigmoid's 0.5."""
        graph = sigmoid_chain_graph()
        result = _arm(
            graph, folding="full", seeds=_seed_sets(graph, STEM_DEAD),
            spiking_mode="lif",
        )
        assert result.constant_folds.folded_rows == {}
        assert result.pruned_rows_per_node[2] == set()

    def test_default_path_leaves_weights_untouched_when_nothing_folds(self):
        graph = sigmoid_chain_graph()
        before = copy.deepcopy(graph)
        prune_ir_graph(graph, spiking_mode="lif")
        for a, b in zip(before.nodes, graph.nodes):
            if isinstance(a, NeuralCore) and a.core_matrix is not None:
                assert np.array_equal(a.core_matrix, b.core_matrix)


class TestCertificateTripsOnFoldMutation:
    """A certificate that cannot fail is not a certificate."""

    def test_corrupted_fold_coefficient_trips(self, monkeypatch):
        import mimarsinan.mapping.pruning.graph.constant_folding as cf

        original = cf.ConstantFoldState.record_fold

        def corrupted(self, node_id, row, constant, row_weights):
            return original(self, node_id, row, constant * 2.0, row_weights)

        monkeypatch.setattr(cf.ConstantFoldState, "record_fold", corrupted)
        with pytest.raises(CascadeCertificateError, match="differ"):
            certify_cascade_equivalence(
                sigmoid_chain_graph(), initial_pruned_per_node=STEM_DEAD,
                batches=3, batch_size=8,
            )

    def test_corrupted_constant_evaluation_trips(self, monkeypatch):
        import mimarsinan.mapping.pruning.liveness_transfer.constant_transfer as ct

        original = ct.derive_constant_outputs

        def corrupted(op, transfer, in_values):
            return {
                o: v + 0.25 for o, v in original(op, transfer, in_values).items()
            }

        monkeypatch.setattr(ct, "derive_constant_outputs", corrupted)
        with pytest.raises(CascadeCertificateError, match="differ"):
            certify_cascade_equivalence(
                sigmoid_chain_graph(), initial_pruned_per_node=STEM_DEAD,
                batches=3, batch_size=8,
            )

    def test_corrupted_carrier_target_trips(self, monkeypatch):
        """Folding onto the WRONG column of the carrier must be caught."""
        import mimarsinan.mapping.pruning.graph.constant_folding as cf

        original = cf.ConstantFoldState.record_fold

        def corrupted(self, node_id, row, constant, row_weights):
            return original(
                self, node_id, row, constant, np.roll(
                    np.asarray(row_weights, dtype=np.float64), 1
                )
            )

        monkeypatch.setattr(cf.ConstantFoldState, "record_fold", corrupted)
        with pytest.raises(CascadeCertificateError, match="differ"):
            certify_cascade_equivalence(
                residual_join_graph(), initial_pruned_per_node=STEM_DEAD,
                batches=3, batch_size=8,
            )


class TestCertificateRefusesGridBreakingConstants:
    def test_gelu_of_a_nonzero_constant_folds_but_never_certifies(self):
        """Execution-exact, grid-breaking: the fold happens (it is exactly what
        the deployment computes) and the CERTIFICATE is what refuses — the
        documented honesty, moved to the gate that can actually see the grid."""
        import torch.nn as nn

        from mimarsinan.mapping.pruning.certificate.dyadic_grid import is_on_grid
        from mimarsinan.mapping.pruning.liveness_transfer import (
            derive_constant_outputs,
        )
        from mimarsinan.mapping.pruning.liveness_transfer.transfer_types import (
            OPAQUE_TRANSFER,
        )
        from unit.mapping.test_constant_lattice import _op

        folded = derive_constant_outputs(
            _op(nn.GELU().eval()), OPAQUE_TRANSFER, [0.5] * 4
        )
        assert set(folded) == {0, 1, 2, 3}
        assert not is_on_grid(list(folded.values()))

    def test_the_certificate_refuses_the_execution_exact_vehicle(self):
        with pytest.raises(
            CascadeCertificatePreconditionError, match="off-grid"
        ):
            certify_cascade_equivalence(
                gelu_execution_exact_graph(), batches=1, batch_size=2,
            )

    def test_identity_carries_the_non_dyadic_constant_one_line_further(self):
        """The op class the fp64-vs-fp32 gate refused outright: an Identity
        relaying a constant the deployment carries exactly but the dyadic grid
        does not."""
        from mimarsinan.mapping.pruning.certificate.dyadic_grid import is_on_grid

        result = _arm(gelu_execution_exact_graph(), folding="full")
        lines = constant_line_values(result)
        relay = next(
            n for n in gelu_execution_exact_graph().nodes if n.name == "relay"
        )
        assert (relay.id, 0) in lines
        assert not is_on_grid([lines[(relay.id, 0)]])


class TestLedgerAttributionAndReplay:
    def test_constant_fold_is_its_own_attribution_category(self):
        ledger = compute_elimination_ledger(
            sigmoid_chain_graph(), initial_pruned_per_node=STEM_DEAD,
            spiking_mode=INERT_SPIKING_MODE,
        )
        record = next(r for r in ledger.per_node if r.node_id == 2)
        assert record.counts.constant_rows == 4
        assert record.counts.closure_rows == 0
        assert record.counts.emergent_rows == 0
        assert ledger.constant_fold_rows == 4
        assert ledger.to_dict()["constant_fold_rows"] == 4

    def test_depth_replay_traverses_the_folds_and_reconciles(self):
        """The replay RAISES on divergence, so a green ledger IS the proof
        that the replay walked the same folds as production."""
        ledger = compute_elimination_ledger(
            residual_join_graph(), initial_pruned_per_node=STEM_DEAD,
            spiking_mode=INERT_SPIKING_MODE,
        )
        head = next(r for r in ledger.per_node if r.node_id == 3)
        assert set(head.row_depths) == {0, 1, 2, 3}
        assert all(d >= 1 for d in head.row_depths.values())

    def test_bias_only_collapse_is_recorded_as_a_deleted_core(self):
        ledger = compute_elimination_ledger(
            bias_only_collapse_graph(), spiking_mode=INERT_SPIKING_MODE,
        )
        assert ledger.cores_deleted == 1
        assert ledger.constant_fold_rows == 1

    def test_grid_certifiable_folds_are_counted_apart_from_exact_only(self):
        """[W4b-2] every fold is classified with the SSOT grid predicate: a
        certifiable fold and an execution-exact-only one both land, and the
        ledger says which is which instead of hiding the refusal."""
        certifiable = compute_elimination_ledger(
            sigmoid_chain_graph(), initial_pruned_per_node=STEM_DEAD,
            spiking_mode=INERT_SPIKING_MODE,
        )
        assert certifiable.constant_fold_rows == 4
        assert certifiable.constant_fold_rows_grid_certifiable == 4
        assert certifiable.constant_fold_rows_execution_exact_only == 0

        exact_only = compute_elimination_ledger(
            gelu_execution_exact_graph(), spiking_mode=INERT_SPIKING_MODE,
        )
        assert exact_only.constant_fold_rows == 1
        assert exact_only.constant_fold_rows_grid_certifiable == 0
        assert exact_only.constant_fold_rows_execution_exact_only == 1
        row = exact_only.to_dict()
        assert row["constant_fold_rows_grid_certifiable"] == 0
        assert row["constant_fold_rows_execution_exact_only"] == 1
        assert "grid_certifiable" in exact_only.summary()

    def test_the_split_partitions_the_constant_fold_category(self):
        ledger = compute_elimination_ledger(
            gelu_execution_exact_graph(), spiking_mode=INERT_SPIKING_MODE,
        )
        assert ledger.constant_fold_rows == (
            ledger.constant_fold_rows_grid_certifiable
            + ledger.constant_fold_rows_execution_exact_only
        )

    def test_ledger_totals_stay_a_partition(self):
        ledger = compute_elimination_ledger(
            sigmoid_chain_graph(), initial_pruned_per_node=STEM_DEAD,
            spiking_mode=INERT_SPIKING_MODE,
        )
        row = ledger.to_dict()
        assert row["total_rows_eliminated"] == (
            row["seed_rows"] + row["closure_coupling_rows"]
            + row["emergent_propagation_rows"] + row["liveness_dead_rows"]
            + row["constant_fold_rows"]
        )


class TestRetrospectiveReanalysis:
    def test_reharvest_reports_the_delta_without_mutating_the_artifact(self):
        graph = sigmoid_chain_graph()
        before = copy.deepcopy(graph)
        report = reanalyze_constant_folding(
            graph, initial_pruned_per_node=STEM_DEAD,
            spiking_mode=INERT_SPIKING_MODE,
        )
        assert report.additional_kills > 0
        assert report.folded_rows == 4
        assert report.cores_with_folds == 1
        assert report.to_dict()["additional_kills"] == report.additional_kills
        for a, b in zip(before.nodes, graph.nodes):
            if isinstance(a, NeuralCore) and a.core_matrix is not None:
                assert np.array_equal(a.core_matrix, b.core_matrix)

    def test_reharvest_is_inert_in_the_spiking_domain(self):
        report = reanalyze_constant_folding(
            sigmoid_chain_graph(), initial_pruned_per_node=STEM_DEAD,
            spiking_mode="lif",
        )
        assert report.additional_kills == 0
        assert report.folded_rows == 0
