"""W3c asymmetric per-instance orphan on a shared weight bank.

Two bank-backed instances (A, B) share the SAME ``weight_row_slice`` of one
physical WeightBank. Admitted seeds kill everything feeding instance A's view
of physical column 0 while instance B keeps LIVE rows into that same column.

The shared-bank rule (see ``check_shared_bank_union_rule``): a bank column may
be eliminated only if it is dead for ALL sharing instances. Propagation-driven
deadness discovered in ONE instance's view (starvation or consumer-orphaning)
must therefore stay per-instance and must NOT kill the physical column that
the other sharer still drives — killing it drops live signal and breaks the
value function.

These tests pin both the structural semantics (bank column 0 survives, the
deadness stays asymmetric per node) and the end-to-end W1 certificate
(bit-exact value parity through the deployed executor) on this scenario, for
every propagation arm, plus ledger/replay reconciliation.
"""

import numpy as np
import pytest

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.pruning.certificate import (
    certify_cascade_equivalence,
    check_shared_bank_union_rule,
)
from mimarsinan.mapping.pruning.elimination_ledger import (
    compute_elimination_ledger,
)
from mimarsinan.mapping.pruning.graph.propagation_mode import (
    ELIMINATION_PROPAGATION_CASCADE,
    ELIMINATION_PROPAGATION_MODES,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_core import (
    compute_global_pruned_sets,
)
from mimarsinan.mapping.pruning.ir_pruning_helpers import (
    _boundary_policy_exemptions,
)

U_A, U_B, INST_A, INST_B, HEAD = 0, 1, 2, 3, 4


def _srcs(specs):
    return np.array([IRSource(node_id=n, index=i) for n, i in specs], dtype=object)


def _dyadic(rng, shape, fraction_bits=4, span=8):
    ints = rng.integers(-span, span + 1, size=shape).astype(np.float64)
    return np.ldexp(ints, -fraction_bits)


def _asymmetric_shared_bank_graph(seed=17):
    """Two owned feeders -> two bank-backed sharers (SAME slice) -> owned head.

    Bank (2x2) is diagonal, so physical column j is fed exclusively by
    physical row j. U_A feeds instance A, U_B feeds instance B: killing
    U_A's neuron 0 starves column 0 in A's view ONLY, while B still drives
    the same physical column through its live axon row 0.
    """
    rng = np.random.default_rng(seed)
    bank = WeightBank(
        id=0,
        core_matrix=np.array([[0.5, 0.0], [0.0, 0.75]], dtype=np.float64),
    )
    u_a = NeuralCore(
        id=U_A, name="u_a", input_sources=_srcs([(-2, 0), (-2, 1)]),
        core_matrix=np.array([[0.25, 0.0], [0.0, 0.5]], dtype=np.float64),
        threshold=1.0, latency=0,
    )
    u_b = NeuralCore(
        id=U_B, name="u_b", input_sources=_srcs([(-2, 2), (-2, 3)]),
        core_matrix=np.array([[0.5, 0.0], [0.0, 0.25]], dtype=np.float64),
        threshold=1.0, latency=0,
    )
    inst_a = NeuralCore(
        id=INST_A, name="inst_a", input_sources=_srcs([(U_A, 0), (U_A, 1)]),
        core_matrix=None, weight_bank_id=0, weight_row_slice=(0, 2),
        threshold=1.0, latency=1,
    )
    inst_b = NeuralCore(
        id=INST_B, name="inst_b", input_sources=_srcs([(U_B, 0), (U_B, 1)]),
        core_matrix=None, weight_bank_id=0, weight_row_slice=(0, 2),
        threshold=1.0, latency=1,
    )
    head = NeuralCore(
        id=HEAD, name="head",
        input_sources=_srcs(
            [(INST_A, 0), (INST_A, 1), (INST_B, 0), (INST_B, 1), (-3, 0)]
        ),
        core_matrix=_dyadic(rng, (5, 2)), threshold=1.0, latency=2,
    )
    # Row 2 of the head reads (B, 0): force it nonzero so dropping the shared
    # bank column for B provably changes the model's value function.
    head.core_matrix[2, :] = [0.5, -0.25]
    return IRGraph(
        nodes=[u_a, u_b, inst_a, inst_b, head],
        output_sources=_srcs([(HEAD, 0), (HEAD, 1)]),
        weight_banks={0: bank},
    )


# Seeds that create the asymmetry.
STARVATION_SEEDS = {U_A: ([False, False], [True, False])}   # kill U_A neuron 0
ORPHAN_SEEDS = {HEAD: ([True, False, False, False, False], [False, False])}


def _fixpoint(graph, seeds_sets, mode=ELIMINATION_PROPAGATION_CASCADE):
    exempt_rows, exempt_cols = _boundary_policy_exemptions(graph)
    return compute_global_pruned_sets(
        graph,
        initial_per_node=seeds_sets,
        exempt_rows_per_node=exempt_rows,
        exempt_cols_per_node=exempt_cols,
        mode=mode,
    )


class TestAsymmetricStarvationStructure:
    """Instance A's rows into column 0 die; B keeps live rows into column 0."""

    def test_bank_column_survives_for_the_live_sharer(self):
        graph = _asymmetric_shared_bank_graph()
        result = _fixpoint(graph, {U_A: (set(), {0})})
        assert result.pruned_cols_per_node[INST_A] == {0}, (
            "instance A's view of column 0 is starved and must die per-node"
        )
        assert result.pruned_cols_per_node[INST_B] == set(), (
            "instance B still drives column 0 with live rows; killing its "
            "local view drops live signal"
        )
        assert 0 not in result.pruned_cols_per_bank[0], (
            "the PHYSICAL bank column may die only when dead for ALL sharers"
        )
        assert 0 not in result.pruned_rows_per_bank[0], (
            "bank row 0 still carries B's live signal into column 0"
        )
        check_shared_bank_union_rule(graph, result)

    def test_live_sharer_chain_stays_untouched(self):
        graph = _asymmetric_shared_bank_graph()
        result = _fixpoint(graph, {U_A: (set(), {0})})
        assert result.pruned_rows_per_node[INST_B] == set()
        assert result.pruned_cols_per_node[U_B] == set()
        assert result.pruned_rows_per_node[HEAD] == {0}, (
            "only the head row reading the genuinely dead (A, 0) may die"
        )

    def test_symmetric_deadness_still_reaches_the_bank(self):
        """When BOTH sharers lose column 0 the physical column must die —
        the fix must not weaken legitimate shared reclamation."""
        graph = _asymmetric_shared_bank_graph()
        result = _fixpoint(graph, {U_A: (set(), {0}), U_B: (set(), {0})})
        assert result.pruned_cols_per_node[INST_A] == {0}
        assert result.pruned_cols_per_node[INST_B] == {0}
        assert 0 in result.pruned_cols_per_bank[0]
        assert 0 in result.pruned_rows_per_bank[0]
        check_shared_bank_union_rule(graph, result)


class TestAsymmetricOrphanCertificate:
    """The W1 certificate must stay green: bit-exact value parity."""

    def test_certificate_green_on_asymmetric_starvation(self):
        graph = _asymmetric_shared_bank_graph()
        report = certify_cascade_equivalence(
            graph, initial_pruned_per_node=STARVATION_SEEDS,
            batches=2, batch_size=4,
        )
        assert report.passed is True

    def test_certificate_green_on_asymmetric_consumer_orphan(self):
        """Variant: A's column 0 dies by ORPHANING (its only consumer row is
        seeded dead) instead of starvation; B's column 0 keeps a live reader."""
        graph = _asymmetric_shared_bank_graph()
        report = certify_cascade_equivalence(
            graph, initial_pruned_per_node=ORPHAN_SEEDS,
            batches=2, batch_size=4,
        )
        assert report.passed is True

    @pytest.mark.parametrize("mode", ELIMINATION_PROPAGATION_MODES)
    def test_certificate_green_under_every_arm(self, mode):
        graph = _asymmetric_shared_bank_graph()
        report = certify_cascade_equivalence(
            graph, initial_pruned_per_node=STARVATION_SEEDS,
            batches=2, batch_size=4, elimination_propagation=mode,
        )
        assert report.passed is True

    def test_certificate_green_on_symmetric_bank_death(self):
        graph = _asymmetric_shared_bank_graph()
        report = certify_cascade_equivalence(
            graph,
            initial_pruned_per_node={
                U_A: ([False, False], [True, False]),
                U_B: ([False, False], [True, False]),
            },
            batches=2, batch_size=4,
        )
        assert report.passed is True
        assert report.pruned_cells < report.reference_cells


class TestAsymmetricOrphanLedger:
    """Arm ordering + depth-replay reconciliation on the asymmetric vehicle
    (compute_elimination_ledger fails loud on either violation)."""

    @pytest.mark.parametrize("mode", ELIMINATION_PROPAGATION_MODES)
    def test_ledger_reconciles(self, mode):
        graph = _asymmetric_shared_bank_graph()
        ledger = compute_elimination_ledger(
            graph, initial_pruned_per_node=STARVATION_SEEDS,
            elimination_propagation=mode,
        )
        assert ledger.mode == mode

    def test_arm_ordering_invariant(self):
        graph = _asymmetric_shared_bank_graph()
        results = {
            m: _fixpoint(graph, {U_A: (set(), {0})}, mode=m)
            for m in ELIMINATION_PROPAGATION_MODES
        }
        masked, closure, cascade = (
            results[m] for m in ELIMINATION_PROPAGATION_MODES
        )
        for lo, hi in ((masked, closure), (closure, cascade)):
            for nid in lo.pruned_rows_per_node:
                assert lo.pruned_rows_per_node[nid] <= hi.pruned_rows_per_node[nid]
                assert lo.pruned_cols_per_node[nid] <= hi.pruned_cols_per_node[nid]
            for bid in lo.pruned_rows_per_bank:
                assert lo.pruned_rows_per_bank[bid] <= hi.pruned_rows_per_bank[bid]
                assert lo.pruned_cols_per_bank[bid] <= hi.pruned_cols_per_bank[bid]
