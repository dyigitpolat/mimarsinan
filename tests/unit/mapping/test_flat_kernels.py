"""Per-operator differential gate: flat kernels vs the reference kernels.

Phase 2 of docs/elimination_flat_engine_plan.md. Every flat kernel must equal
its reference twin ON EVERY CORE for every adversarial topology and for
randomized monotone prune states -- the states a wave loop can actually reach.
No tolerance anywhere: these are set predicates, and the sets must be equal.
"""

import numpy as np
import pytest

from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE
from mimarsinan.mapping.pruning.graph.analysis_dependencies import (
    PortPlan,
    _dead_axons_from_plan,
    _orphans_from_plan,
)
from mimarsinan.mapping.pruning.graph.flat import (
    build_flat_state,
    flat_cross_core_dead_axons,
    flat_orphan_neurons,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_seeding import (
    build_global_pruning_context,
)

from unit.mapping.adversarial_topologies import ADVERSARIAL_TOPOLOGIES, build_topology


def _ctx(name):
    graph, seeds = build_topology(name)
    return build_global_pruning_context(
        graph, zero_threshold=1e-8, initial_per_node=seeds, initial_per_bank=None,
        exempt_rows_per_node=None, exempt_cols_per_node=None,
        computeop_liveness_transfers="full", elimination_constant_folding="off",
        spiking_mode=INERT_SPIKING_MODE,
    )


def _random_state(ctx, state, rng):
    """A random MONOTONE-reachable prune state, mirrored into both forms."""
    row_dead = np.zeros(state.n_rows, dtype=bool)
    col_dead = np.zeros(state.n_cols, dtype=bool)
    pruned_rows = {nid: set() for nid in state.node_ids}
    pruned_cols = {nid: set() for nid in state.node_ids}
    for k, nid in enumerate(state.node_ids):
        rows = state.rows_of(k)
        cols = state.cols_of(k)
        n_r, n_c = rows.stop - rows.start, cols.stop - cols.start
        for i in rng.choice(n_r, size=max(0, n_r // 3), replace=False):
            row_dead[rows.start + i] = True
            pruned_rows[nid].add(int(i))
        for j in rng.choice(n_c, size=max(0, n_c // 3), replace=False):
            col_dead[cols.start + j] = True
            pruned_cols[nid].add(int(j))
    return row_dead, col_dead, pruned_rows, pruned_cols


@pytest.mark.parametrize("name", sorted(ADVERSARIAL_TOPOLOGIES))
def test_flat_kernels_equal_reference_on_every_core(name):
    ctx = _ctx(name)
    state = build_flat_state(ctx)
    rng = np.random.default_rng(7)

    for trial in range(3):
        row_dead, col_dead, pruned_rows, pruned_cols = _random_state(ctx, state, rng)

        flat_axons = flat_cross_core_dead_axons(state, col_dead)
        flat_orphans = flat_orphan_neurons(state, row_dead)

        for k, node in enumerate(ctx.neural_cores):
            mat = ctx.base_node_matrix(node)
            n_neurons = int(mat.shape[1]) if mat is not None else 0
            plan = PortPlan(node, n_neurons, ctx.computeop_transfers,
                            ctx.consumer_axons, ctx.model_output_neurons)

            ref_axons = _dead_axons_from_plan(plan, pruned_cols)
            rows = state.rows_of(k)
            got_axons = {int(i) for i in np.flatnonzero(flat_axons[rows])}
            assert got_axons == ref_axons, (
                f"{name} trial {trial} core {node.id}: dead-axon sets differ; "
                f"flat-only={sorted(got_axons - ref_axons)[:5]} "
                f"ref-only={sorted(ref_axons - got_axons)[:5]}"
            )

            ref_orphans = _orphans_from_plan(plan, pruned_rows)
            cols = state.cols_of(k)
            got_orphans = {int(j) for j in np.flatnonzero(flat_orphans[cols])}
            assert got_orphans == ref_orphans, (
                f"{name} trial {trial} core {node.id}: orphan sets differ; "
                f"flat-only={sorted(got_orphans - ref_orphans)[:5]} "
                f"ref-only={sorted(ref_orphans - got_orphans)[:5]}"
            )


class TestTheDifferentialCanFail:
    """Mutation: corrupt the flat state and prove the gate rejects it."""

    def test_a_dropped_consumer_edge_is_caught(self):
        ctx = _ctx("fan_out")
        state = build_flat_state(ctx)
        rng = np.random.default_rng(3)
        row_dead, col_dead, pruned_rows, _ = _random_state(ctx, state, rng)
        # sabotage: pretend one consumed port has no consumers at all
        if not state.consumer_port.size:
            pytest.skip("topology exposes no consumer edges")
        sabotaged = FlatStateProxy(state)
        orphans = flat_orphan_neurons(sabotaged, row_dead)
        k = int(np.searchsorted(
            state.col_base, int(sabotaged.dropped_port), side="right") - 1)
        node = ctx.neural_cores[k]
        mat = ctx.base_node_matrix(node)
        plan = PortPlan(node, int(mat.shape[1]) if mat is not None else 0,
                        ctx.computeop_transfers, ctx.consumer_axons,
                        ctx.model_output_neurons)
        ref = _orphans_from_plan(plan, pruned_rows)
        cols = state.cols_of(k)
        got = {int(j) for j in np.flatnonzero(orphans[cols])}
        assert got != ref, "the sabotage must be visible or the gate proves nothing"


class FlatStateProxy:
    """A FlatState with one consumed port silently converted to orphan-now."""

    def __init__(self, state):
        self._s = state
        self.dropped_port = int(state.consumer_port[0])
        self.orphan_now = np.concatenate(
            [state.orphan_now, np.asarray([self.dropped_port])])
        self.consumer_port = state.consumer_port[1:]
        self.consumer_offsets = state.consumer_offsets[1:].copy()
        self.consumer_rows = state.consumer_rows[
            int(state.consumer_offsets[1]):]
        self.consumer_offsets -= int(state.consumer_offsets[1])

    def __getattr__(self, item):
        return getattr(self._s, item)
