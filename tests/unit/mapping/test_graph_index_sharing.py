"""One graph index per run: build the immutable structure once, not per arm.

Measured (real 4,925-core ViT, 2026-08-07): a single context build spends
93.1 s on the ComputeOp transfer index and 6.8 s on the consumer index — both
pure functions of (graph, transfer policy) — and the ledger builds a fresh
context per arm plus one for the replay, so ~400-500 s of the 851 s analysis
is rebuilding identical structure.

Sharing must be bit-identical (it removes work, never changes facts) and must
REFUSE a mismatched index rather than silently analyzing under the wrong
policy.
"""

import pytest

from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE
from mimarsinan.mapping.pruning.elimination_ledger.arm_runs import (
    compute_elimination_arms,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_seeding import (
    build_global_pruning_context,
    build_graph_index,
)

import unit.mapping.constant_vehicles as vehicles

VEHICLES = [
    "bias_only_collapse_graph", "sigmoid_chain_graph",
    "residual_join_graph", "gelu_execution_exact_graph",
]


def _ctx(graph, *, index=None, transfers="full"):
    return build_global_pruning_context(
        graph, zero_threshold=1e-8,
        initial_per_node=None, initial_per_bank=None,
        exempt_rows_per_node=None, exempt_cols_per_node=None,
        computeop_liveness_transfers=transfers,
        elimination_constant_folding="full",
        spiking_mode=INERT_SPIKING_MODE,
        graph_index=index,
    )


def _state(ctx):
    from mimarsinan.mapping.pruning.graph.pruning_graph_core import (
        _run_cascade_fixpoint,
    )
    _run_cascade_fixpoint(ctx)
    return {
        "rows": {n: sorted(s) for n, s in sorted(ctx.pruned_rows.items())},
        "cols": {n: sorted(s) for n, s in sorted(ctx.pruned_cols.items())},
        "lattice": {k: float(v).hex()
                    for k, v in (ctx.constants.lattice.values or {}).items()},
    }


class TestSharedIndexIsBitIdentical:
    @pytest.mark.parametrize("maker", VEHICLES)
    def test_shared_index_changes_nothing(self, maker):
        graph = getattr(vehicles, maker)()
        baseline = _state(_ctx(graph))
        index = build_graph_index(graph, computeop_liveness_transfers="full")
        first = _state(_ctx(graph, index=index))
        second = _state(_ctx(graph, index=index))   # reused across contexts
        assert first == baseline, f"{maker}: shared index diverged"
        assert second == baseline, f"{maker}: index reuse diverged"


class TestTheIndexIsBuiltOncePerRun:
    def test_arms_do_not_rebuild_the_transfer_index(self, monkeypatch):
        import mimarsinan.mapping.pruning.graph.pruning_graph_types as types

        calls = []
        real = types.build_computeop_transfer_index

        def counting(*a, **k):
            calls.append(1)
            return real(*a, **k)

        monkeypatch.setattr(types, "build_computeop_transfer_index", counting)
        arms = compute_elimination_arms(
            vehicles.gelu_execution_exact_graph(),
            elimination_constant_folding="full",
            computeop_liveness_transfers="full",
            spiking_mode=INERT_SPIKING_MODE,
        )
        assert arms.graph_index is not None
        assert len(calls) <= 1, (
            f"the transfer index was built {len(calls)} times; one run must "
            f"build it once and share it across arms and the replay"
        )


class TestAMismatchedIndexIsRefused:
    """A silently wrong index would analyze under the wrong policy."""

    def test_policy_mismatch_raises(self):
        graph = vehicles.gelu_execution_exact_graph()
        index = build_graph_index(graph, computeop_liveness_transfers="identity_only")
        with pytest.raises(ValueError, match="transfer policy"):
            _ctx(graph, index=index, transfers="full")

    def test_foreign_graph_raises(self):
        index = build_graph_index(
            vehicles.sigmoid_chain_graph(), computeop_liveness_transfers="full",
        )
        with pytest.raises(ValueError, match="different graph"):
            _ctx(vehicles.gelu_execution_exact_graph(), index=index)
