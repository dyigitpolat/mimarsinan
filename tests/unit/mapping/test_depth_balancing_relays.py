"""Depth-balancing relays for unequal-depth intra-segment fan-in (C5, V6).

An intra-segment edge with depth gap > 1 both DROPS the shallow branch's early
spikes and RE-READS its stale final buffer (lif_step.py window semantics).
The exact mapper-level remedy inserts identity relay cores on the shallow
branch; under the strict "<" comparator a relay's identity weight must exceed
theta (exact-theta is silent — the integer-lattice dead-relay hazard, V9).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pytest

from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.latency.depth_balancing import (
    DeadRelayError,
    UnbalancedFanInError,
    assert_intra_segment_gap1,
    assert_relays_alive,
    find_intra_segment_depth_gaps,
    insert_depth_balancing_relays,
    relay_weight,
)
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_identity_hybrid_mapping,
)
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

T = 8
N = 2


def _core(node_id, name, sources, matrix, latency=None):
    return NeuralCore(
        id=node_id,
        name=name,
        input_sources=np.asarray(sources, dtype=object),
        core_matrix=np.asarray(matrix, dtype=np.float64),
        threshold=1.0,
        latency=latency,
        parameter_scale=torch.tensor(0.0),
    )


def _join_graph(*, deep_w=0.4, shallow_w=0.35) -> IRGraph:
    """A(2ch) -> B1 -> B2 -> C plus the direct shallow edge A -> C (gap 3)."""
    eye = np.eye(N)
    a = _core(0, "A", [IRSource(-2, i) for i in range(N)], eye)
    b1 = _core(1, "B1", [IRSource(0, i) for i in range(N)], eye)
    b2 = _core(2, "B2", [IRSource(1, i) for i in range(N)], eye)
    c = _core(
        3, "C",
        [IRSource(2, i) for i in range(N)] + [IRSource(0, i) for i in range(N)],
        np.vstack([eye * deep_w, eye * shallow_w]),
    )
    return IRGraph(
        nodes=[a, b1, b2, c],
        output_sources=np.asarray([IRSource(3, i) for i in range(N)], dtype=object),
    )


def _chain_graph() -> IRGraph:
    eye = np.eye(N)
    a = _core(0, "A", [IRSource(-2, i) for i in range(N)], eye)
    b = _core(1, "B", [IRSource(0, i) for i in range(N)], eye * 0.5)
    return IRGraph(
        nodes=[a, b],
        output_sources=np.asarray([IRSource(1, i) for i in range(N)], dtype=object),
    )


class TestGapDetection:
    def test_join_graph_has_one_gap3_violation(self):
        violations = find_intra_segment_depth_gaps(_join_graph())
        assert [(v.producer_id, v.consumer_id, v.gap) for v in violations] == [
            (0, 3, 3)
        ]

    def test_chain_graph_is_clean(self):
        assert find_intra_segment_depth_gaps(_chain_graph()) == []

    def test_gap1_guard_raises_loud_on_join(self):
        with pytest.raises(UnbalancedFanInError, match="gap"):
            assert_intra_segment_gap1(_join_graph())

    def test_gap1_guard_passes_on_chain(self):
        assert_intra_segment_gap1(_chain_graph())

    def test_zero_weight_edges_are_dead(self):
        """A gap edge whose consumer columns are all zero is not live."""
        graph = _join_graph(shallow_w=0.0)
        assert find_intra_segment_depth_gaps(graph) == []


class TestRelayInsertion:
    def test_inserts_gap_minus_one_relays_and_rewires(self):
        graph = _join_graph()
        inserted = insert_depth_balancing_relays(graph, thresholding_mode="<=")
        assert inserted == 2
        assert find_intra_segment_depth_gaps(graph) == []
        assert_intra_segment_gap1(graph)
        assert graph.validate() == []
        relays = [n for n in graph.nodes if getattr(n, "_depth_relay", False)]
        assert len(relays) == 2
        # The consumer's shallow axons now reference the LAST relay.
        consumer = graph.get_node_by_id(3)
        srcs = [s.node_id for s in consumer.input_sources.flatten()]
        assert srcs[:N] == [2] * N
        assert srcs[N:] == [relays[-1].id] * N

    def test_noop_on_gap_free_graph(self):
        graph = _chain_graph()
        nodes_before = list(graph.nodes)
        inserted = insert_depth_balancing_relays(graph, thresholding_mode="<")
        assert inserted == 0
        assert graph.nodes == nodes_before

    def test_idempotent(self):
        graph = _join_graph()
        assert insert_depth_balancing_relays(graph, thresholding_mode="<=") == 2
        assert insert_depth_balancing_relays(graph, thresholding_mode="<=") == 0

    def test_relay_weight_exceeds_theta_under_strict_comparator(self):
        assert relay_weight("<") > 1.0
        assert relay_weight("<=") == 1.0
        graph = _join_graph()
        insert_depth_balancing_relays(graph, thresholding_mode="<")
        assert_relays_alive(graph, thresholding_mode="<")


class TestDeadRelayGuard:
    def test_exact_theta_relay_is_dead_under_strict(self):
        graph = _join_graph()
        insert_depth_balancing_relays(graph, thresholding_mode="<")
        # Snap a relay's weight onto the threshold lattice (the hazard).
        relay = next(n for n in graph.nodes if getattr(n, "_depth_relay", False))
        relay.core_matrix = np.eye(N, dtype=np.float64)
        with pytest.raises(DeadRelayError, match="never fires"):
            assert_relays_alive(graph, thresholding_mode="<")
        # The same weight is alive under the inclusive comparator.
        assert_relays_alive(graph, thresholding_mode="<=")

    def test_relays_survive_chip_quantization(self):
        """threshold=scale stays a float below the integer relay weight, so the
        quantized relay stays alive under '<' (no lattice snap today)."""
        graph = _join_graph()
        insert_depth_balancing_relays(graph, thresholding_mode="<")
        quantize_ir_graph(graph, 8, weight_quantization=True)
        assert_relays_alive(graph, thresholding_mode="<")

    def test_no_relays_is_a_noop(self):
        assert_relays_alive(_chain_graph(), thresholding_mode="<")


class TestJoinCountExactness:
    """With relays the join core's counts equal the commutation target
    clamp(F(Q), 0, T); without them the stale-read/head-drop corrupts them."""

    def _counts(self, graph, rate: float) -> torch.Tensor:
        hybrid = build_identity_hybrid_mapping(ir_graph=graph)
        flow = SpikingHybridCoreFlow(
            input_shape=(N,),
            hybrid_mapping=hybrid,
            simulation_length=T,
            preprocessor=nn.Identity(),
            firing_mode="Default",
            spike_mode="Uniform",
            thresholding_mode="<=",
            spiking_mode="lif",
            cycle_accurate_lif_forward=True,
        ).eval()
        x = torch.full((1, N), rate, dtype=torch.float32)
        with torch.no_grad():
            return flow(x)[0].to(torch.float64)

    def _target(self, rate: float) -> torch.Tensor:
        n = round(rate * T)
        q = torch.full((N,), (0.4 + 0.35) * n, dtype=torch.float64)
        return torch.clamp(torch.floor(q), 0.0, float(T))

    def test_balanced_join_matches_commutation_target(self):
        graph = _join_graph()
        insert_depth_balancing_relays(graph, thresholding_mode="<=")
        for rate in (0.25, 0.5, 0.75, 1.0):
            got = self._counts(graph, rate)
            assert torch.equal(got, self._target(rate)), (
                f"rate={rate}: relayed join {got.tolist()} != "
                f"target {self._target(rate).tolist()}"
            )

    def test_unbalanced_join_violates_commutation_target(self):
        graph = _join_graph()
        mismatched = [
            rate for rate in (0.25, 0.5, 0.75)
            if not torch.equal(self._counts(graph, rate), self._target(rate))
        ]
        assert mismatched, (
            "the un-relayed join unexpectedly matched the commutation target "
            "on every rate — the V6 window hazard no longer reproduces?"
        )
