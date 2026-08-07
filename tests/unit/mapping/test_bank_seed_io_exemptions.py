"""W3e: ``initial_per_bank`` seeds on model-I/O-exempt bank structure are inert.

``build_global_pruning_context`` unions ``initial_per_bank`` seeds into
``ctx.bank_pruned_rows/cols`` WITHOUT exemption filtering (unlike the
``initial_per_node`` arm). That asymmetry is safe — NOT-A-DEFECT — because of
a downstream guard these tests pin:

  1. ``_refresh_bank_pruning`` (pruning_graph_refresh.py) derives per-bank
     exemptions from every referencing core's model-I/O exemptions offset
     through ``weight_row_slice`` (the same derivation as the cascade-step
     kernel ``_bank_causal_kills``), and
  2. ``compute_propagated_pruned_rows_cols`` (pruning_propagation.py, the
     "Exempt indices are never added at init" mask strip) removes exempt
     coordinates from the INITIAL seed masks, after which
  3. ``_refresh_bank_pruning`` REPLACES ``bank_pruned_rows/cols[bank_id]``
     with the scrubbed result — the illegal seed is gone, not merely ignored.

Every propagation arm invokes a bank refresh before any result escapes
(masked/closure via ``run_bank_alias_fixpoint``, cascade via the fixpoint
loop), and nothing consumes ``ctx.bank_pruned_*`` between seed-union and the
first refresh, so the unfiltered seed can never reach physical elimination.

If seed-time acceptance is ever changed to bypass or reorder that refresh
(e.g. union-only bank updates), these tests fail: an I/O-exempt bank seed
would drop model-output-feeding or model-input-fed physical structure.
"""

import copy

import numpy as np
import pytest

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.pruning.certificate import certify_cascade_equivalence
from mimarsinan.mapping.pruning.elimination_ledger import (
    compute_elimination_ledger,
)
from mimarsinan.mapping.pruning.graph.propagation_mode import (
    ELIMINATION_PROPAGATION_MODES,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_core import (
    compute_global_pruned_sets,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_modes import run_masked
from mimarsinan.mapping.pruning.graph.pruning_graph_seeding import (
    build_global_pruning_context,
)
from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph
from mimarsinan.mapping.pruning.ir_pruning_helpers import (
    _boundary_policy_exemptions,
    _collect_initial_seeds,
)

def _srcs(specs):
    return np.array(
        [IRSource(node_id=n, index=i) for n, i in specs], dtype=object
    )


def _output_side_graph():
    """u (owned) -> inst (bank-backed); BOTH inst neurons are model outputs.

    Bank is diagonal, so seeding bank column 0 would (absent the guard)
    starve bank row 0 and physically drop the structure feeding output 0.
    """
    bank = WeightBank(
        id=0, core_matrix=np.array([[0.5, 0.0], [0.0, 0.75]])
    )
    u = NeuralCore(
        id=0, name="u", input_sources=_srcs([(-2, 0), (-2, 1)]),
        core_matrix=np.array([[0.25, 0.0], [0.0, 0.5]]),
        threshold=1.0, latency=0,
    )
    inst = NeuralCore(
        id=1, name="inst", input_sources=_srcs([(0, 0), (0, 1)]),
        core_matrix=None, weight_bank_id=0, weight_row_slice=(0, 2),
        threshold=1.0, latency=1,
    )
    return IRGraph(
        nodes=[u, inst], output_sources=_srcs([(1, 0), (1, 1)]),
        weight_banks={0: bank},
    )


def _input_side_graph():
    """inst (bank-backed) fed DIRECTLY by model input axons -> head (outputs).

    Both bank rows correspond to model-input axons of the sharer, so they are
    I/O-exempt through the derived per-bank exemption; bank columns are NOT
    exempt (inst neurons are interior).
    """
    bank = WeightBank(
        id=0, core_matrix=np.array([[0.5, 0.0], [0.0, 0.75]])
    )
    inst = NeuralCore(
        id=0, name="inst", input_sources=_srcs([(-2, 0), (-2, 1)]),
        core_matrix=None, weight_bank_id=0, weight_row_slice=(0, 2),
        threshold=1.0, latency=0,
    )
    head = NeuralCore(
        id=1, name="head", input_sources=_srcs([(0, 0), (0, 1)]),
        core_matrix=np.array([[0.25, 0.5], [0.0, 0.5]]),
        threshold=1.0, latency=1,
    )
    return IRGraph(
        nodes=[inst, head], output_sources=_srcs([(1, 0), (1, 1)]),
        weight_banks={0: bank},
    )


# Bank seeds in the bank's own coordinates (True = pruned).
SEED_OUTPUT_COL = {0: ([False, False], [True, False])}
SEED_INPUT_ROW = {0: ([True, False], [False, False])}
SEED_MIXED = {0: ([True, False], [True, False])}  # exempt row + legit col


def _run_sets(graph, seed_bank, mode):
    exempt_rows, exempt_cols = _boundary_policy_exemptions(graph)
    _, seed_per_bank = _collect_initial_seeds(graph, None, seed_bank)
    return compute_global_pruned_sets(
        graph,
        initial_per_bank=seed_per_bank,
        exempt_rows_per_node=exempt_rows,
        exempt_cols_per_node=exempt_cols,
        mode=mode,
    )


class TestOutputSideBankColumnSeed:
    """A bank column feeding a model output must survive its own seed."""

    @pytest.mark.parametrize("mode", ELIMINATION_PROPAGATION_MODES)
    def test_exempt_bank_column_seed_is_scrubbed(self, mode):
        graph = _output_side_graph()
        result = _run_sets(graph, SEED_OUTPUT_COL, mode)
        assert result.pruned_cols_per_bank[0] == set(), (
            "bank column 0 feeds model output (inst, 0); the seed must be "
            "scrubbed by the derived per-bank exemption, exactly as the "
            "cascade-step kernel would refuse creating the kill"
        )
        assert result.pruned_rows_per_bank[0] == set(), (
            "no within-bank starvation may leak from the refused seed"
        )
        assert result.pruned_cols_per_node[1] == set()
        assert result.pruned_rows_per_node[1] == set()
        assert result.pruned_cols_per_node[0] == set(), (
            "the refused bank seed must not orphan the upstream feeder"
        )

    def test_prune_ir_graph_keeps_output_structure_physically(self):
        graph = _output_side_graph()
        original_bank = graph.weight_banks[0].core_matrix.copy()
        pruned = prune_ir_graph(
            copy.deepcopy(graph), initial_pruned_per_bank=SEED_OUTPUT_COL
        )
        outs = [
            (s.node_id, s.index) for s in pruned.output_sources.flatten()
        ]
        assert outs == [(1, 0), (1, 1)], (
            "model output width/wiring must survive an exempt bank seed"
        )
        np.testing.assert_array_equal(
            pruned.weight_banks[0].core_matrix, original_bank
        )
        inst = next(n for n in pruned.nodes if n.id == 1)
        assert inst.pruned_col_mask == [False, False]
        assert inst.pruned_row_mask == [False, False]

    def test_certificate_bit_exact_and_nothing_reclaimed(self):
        graph = _output_side_graph()
        report = certify_cascade_equivalence(
            graph, initial_pruned_per_bank=SEED_OUTPUT_COL,
            batches=2, batch_size=4,
        )
        assert report.passed is True
        assert report.pruned_cells == report.reference_cells, (
            "the whole seed is I/O-exempt; nothing may be reclaimed"
        )

    @pytest.mark.parametrize("mode", ELIMINATION_PROPAGATION_MODES)
    def test_ledger_reconciles(self, mode):
        graph = _output_side_graph()
        ledger = compute_elimination_ledger(
            graph, initial_pruned_per_bank=SEED_OUTPUT_COL,
            elimination_propagation=mode,
        )
        assert ledger.mode == mode
        bank_record = next(b for b in ledger.per_bank if b.bank_id == 0)
        assert bank_record.counts.seed_cols == 0, (
            "a scrubbed seed must not be attributed as a SEED kill"
        )


class TestInputSideBankRowSeed:
    """A bank row fed by a model-input axon must survive its own seed."""

    @pytest.mark.parametrize("mode", ELIMINATION_PROPAGATION_MODES)
    def test_exempt_bank_row_seed_is_scrubbed(self, mode):
        graph = _input_side_graph()
        result = _run_sets(graph, SEED_INPUT_ROW, mode)
        assert result.pruned_rows_per_bank[0] == set(), (
            "bank row 0 is a model-input axon of the sharer; the seed must "
            "be scrubbed by the derived per-bank row exemption"
        )
        assert result.pruned_cols_per_bank[0] == set()
        assert result.pruned_rows_per_node[0] == set()
        assert result.pruned_cols_per_node[0] == set()
        assert result.pruned_rows_per_node[1] == set()

    def test_certificate_bit_exact_and_nothing_reclaimed(self):
        graph = _input_side_graph()
        report = certify_cascade_equivalence(
            graph, initial_pruned_per_bank=SEED_INPUT_ROW,
            batches=2, batch_size=4,
        )
        assert report.passed is True
        assert report.pruned_cells == report.reference_cells

    @pytest.mark.parametrize("mode", ELIMINATION_PROPAGATION_MODES)
    def test_ledger_reconciles(self, mode):
        graph = _input_side_graph()
        ledger = compute_elimination_ledger(
            graph, initial_pruned_per_bank=SEED_INPUT_ROW,
            elimination_propagation=mode,
        )
        assert ledger.mode == mode


class TestGuardDoesNotOverScrub:
    """The exemption scrub is surgical: legit bank seeds still reclaim."""

    def test_mixed_seed_keeps_exempt_row_and_kills_legit_column(self):
        graph = _input_side_graph()
        result = _run_sets(graph, SEED_MIXED, "cascade")
        assert result.pruned_rows_per_bank[0] == set(), (
            "the exempt input-axon row half of the seed must be scrubbed"
        )
        assert result.pruned_cols_per_bank[0] == {0}, (
            "the non-exempt column half of the same seed must still be "
            "admitted and reclaimed"
        )
        assert result.pruned_cols_per_node[0] == {0}
        assert result.pruned_rows_per_node[1] == {0}, (
            "the head row reading the legitimately dead (inst, 0) cascades"
        )

    def test_certificate_green_with_real_reclamation(self):
        graph = _input_side_graph()
        report = certify_cascade_equivalence(
            graph, initial_pruned_per_bank=SEED_MIXED,
            batches=2, batch_size=4,
        )
        assert report.passed is True
        assert report.pruned_cells < report.reference_cells, (
            "the mixed seed's legitimate column must physically reclaim cells"
        )


class TestGuardMechanismPin:
    """Pin the guard itself: seed-time acceptance IS unfiltered, and the
    FIRST bank refresh (run in every arm before any result escapes) replaces
    the bank kill sets with the exemption-scrubbed fixpoint of
    ``compute_propagated_pruned_rows_cols`` (its "never added at init" strip
    over the derived per-bank exemptions). If either half changes — filtering
    moves to seed time (fine, update this test) or the refresh stops
    replacing/scrubbing (regression) — this fails loudly."""

    def _seeded_ctx(self, graph, seed_bank):
        exempt_rows, exempt_cols = _boundary_policy_exemptions(graph)
        _, seed_per_bank = _collect_initial_seeds(graph, None, seed_bank)
        return build_global_pruning_context(
            graph,
            zero_threshold=1e-8,
            initial_per_node=None,
            initial_per_bank=seed_per_bank,
            exempt_rows_per_node=exempt_rows,
            exempt_cols_per_node=exempt_cols,
        )

    def test_output_col_seed_accepted_then_scrubbed_by_first_refresh(self):
        ctx = self._seeded_ctx(_output_side_graph(), SEED_OUTPUT_COL)
        assert 0 in ctx.bank_pruned_cols[0], (
            "seed-time acceptance is unfiltered today; if this assertion "
            "fails because filtering moved to seed time, the guard moved "
            "earlier — update this pin, the invariant below must still hold"
        )
        run_masked(ctx)
        assert ctx.bank_pruned_cols[0] == set(), (
            "the first bank refresh must scrub the I/O-exempt bank column "
            "seed via the derived per-bank exemption"
        )

    def test_input_row_seed_accepted_then_scrubbed_by_first_refresh(self):
        ctx = self._seeded_ctx(_input_side_graph(), SEED_INPUT_ROW)
        assert 0 in ctx.bank_pruned_rows[0]
        run_masked(ctx)
        assert ctx.bank_pruned_rows[0] == set(), (
            "the first bank refresh must scrub the input-axon bank row seed"
        )
