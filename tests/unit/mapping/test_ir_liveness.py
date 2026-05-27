"""Liveness analysis for IR nodes.

``compute_liveness`` classifies every NeuralCore as ``LIVE`` / ``BIAS_ONLY``
/ ``DEAD`` post-pruning. The contract:

- A node is ``DEAD`` when either
  * **no neuron is reachable from any model output** (every column would be
    pruned by the global propagation solver), OR
  * **the core cannot emit any spike** (every axon is off-source AND
    ``hardware_bias`` cannot exceed the threshold within ``simulation_steps``
    LIF integration cycles).
- A node is ``BIAS_ONLY`` when it is reachable AND every axon is dead but
  ``hardware_bias`` can fire at least one surviving neuron.
- Otherwise the node is ``LIVE``.

DEAD wins over BIAS_ONLY: a bias-only emitter that nothing downstream
consumes is dead, not bias-only.

ComputeOps are barriers, never neural-core-classified themselves.
"""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.mapping.ir import (
    ComputeOp,
    IRGraph,
    IRSource,
    NeuralCore,
)
from mimarsinan.mapping.pruning.ir_liveness import (
    NodeLiveness,
    LivenessResult,
    compute_liveness,
)


def _src(specs):
    return np.array(
        [IRSource(node_id=nid, index=idx) for nid, idx in specs],
        dtype=object,
    )


# Default integration window used in tests; large enough to make subthreshold
# bias exposable as DEAD.
T_DEFAULT = 4


class TestApiContract:
    def test_returns_livenessresult_with_per_node_and_reasons(self):
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        graph = IRGraph(nodes=[a], output_sources=_src([(0, 0)]))
        result = compute_liveness(graph, simulation_steps=T_DEFAULT)
        assert isinstance(result, LivenessResult)
        assert 0 in result.per_node
        assert isinstance(result.per_node[0], NodeLiveness)
        assert isinstance(result.reasons[0], str)
        assert result.reasons[0]  # non-empty

    def test_compute_op_nodes_are_not_classified(self):
        """ComputeOps are barriers, not neural cores: they have no liveness
        entry. Their presence in the graph must not crash classification."""
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0), (-2, 1)]),
            core_matrix=np.array([[1.0], [1.0]], dtype=np.float64),
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
        graph = IRGraph(nodes=[a, relu, b], output_sources=_src([(2, 0)]))
        result = compute_liveness(graph, simulation_steps=T_DEFAULT)
        assert 0 in result.per_node and 2 in result.per_node
        assert 1 not in result.per_node


class TestLocalEmission:
    """Local "can this core ever fire" emission analysis."""

    def test_zero_matrix_zero_bias_is_dead(self):
        """No live axon weight + zero/None bias -> cannot emit -> DEAD."""
        # A is the dead candidate. B keeps the graph valid (output reaches
        # something live) so we exercise A's deadness in isolation.
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.zeros((1, 1), dtype=np.float64),
            threshold=1.0, latency=0,
        )
        b = NeuralCore(
            id=1, name="B",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        graph = IRGraph(nodes=[a, b], output_sources=_src([(1, 0)]))
        result = compute_liveness(graph, simulation_steps=T_DEFAULT)
        assert result.per_node[0] == NodeLiveness.DEAD
        assert result.per_node[1] == NodeLiveness.LIVE

    def test_zero_matrix_strong_bias_with_live_consumer_is_bias_only(self):
        """All axons off + bias above threshold + reachable -> BIAS_ONLY.

        A has no live axons but a strong ``hardware_bias`` that fires its
        single neuron in a single integration step. B consumes A so A is
        reachable. A must be classified BIAS_ONLY.
        """
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-1, 0)]),  # off
            core_matrix=np.zeros((1, 1), dtype=np.float64),
            hardware_bias=np.array([2.0], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        b = NeuralCore(
            id=1, name="B",
            input_sources=_src([(0, 0), (-3, 0)]),
            core_matrix=np.array([[1.0], [0.0]], dtype=np.float64),
            threshold=1.0, latency=1,
        )
        graph = IRGraph(nodes=[a, b], output_sources=_src([(1, 0)]))
        result = compute_liveness(graph, simulation_steps=T_DEFAULT)
        assert result.per_node[0] == NodeLiveness.BIAS_ONLY
        assert "bias" in result.reasons[0].lower()

    def test_zero_matrix_subthreshold_bias_is_dead(self):
        """Bias too weak to fire within simulation_steps -> DEAD.

        ``threshold = 10.0``, ``simulation_steps = 4``, ``bias = 1.0``: at
        most ``4 * 1.0 < 10.0`` membrane potential, never crosses threshold.
        """
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-1, 0)]),
            core_matrix=np.zeros((1, 1), dtype=np.float64),
            hardware_bias=np.array([1.0], dtype=np.float64),
            threshold=10.0, latency=0,
        )
        b = NeuralCore(
            id=1, name="B",
            input_sources=_src([(0, 0), (-3, 0)]),
            core_matrix=np.array([[1.0], [0.0]], dtype=np.float64),
            threshold=1.0, latency=1,
        )
        graph = IRGraph(nodes=[a, b], output_sources=_src([(1, 0)]))
        result = compute_liveness(graph, simulation_steps=T_DEFAULT)
        assert result.per_node[0] == NodeLiveness.DEAD

    def test_zero_matrix_strong_bias_without_consumer_is_dead(self):
        """DEAD wins: bias-only emitter with no reachable consumer is DEAD,
        not BIAS_ONLY. The neuron's spikes have nowhere to go."""
        # A is bias-driven; nothing references A; output_sources references B.
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-1, 0)]),
            core_matrix=np.zeros((1, 1), dtype=np.float64),
            hardware_bias=np.array([5.0], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        b = NeuralCore(
            id=1, name="B",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        graph = IRGraph(nodes=[a, b], output_sources=_src([(1, 0)]))
        result = compute_liveness(graph, simulation_steps=T_DEFAULT)
        assert result.per_node[0] == NodeLiveness.DEAD


class TestReachabilityFixpoint:
    def test_unreachable_normal_core_is_dead(self):
        """A core with non-zero weights but no consumer of its neurons is
        DEAD. Reuses the global pruning solver's reachability result."""
        # A produces 2 neurons; nothing references A; B is the live output path.
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0), (-3, 0)]),
            core_matrix=np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        b = NeuralCore(
            id=1, name="B",
            input_sources=_src([(-2, 1)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        graph = IRGraph(nodes=[a, b], output_sources=_src([(1, 0)]))
        result = compute_liveness(graph, simulation_steps=T_DEFAULT)
        assert result.per_node[0] == NodeLiveness.DEAD
        assert result.per_node[1] == NodeLiveness.LIVE

    def test_chain_propagation_dead_producer_kills_consumer_when_only_input(self):
        """Producer A is DEAD (zero matrix, no bias). B reads only from A.

        After propagation B's axons are all dead and B has no bias -> B is
        also DEAD (pure dead-math). The output_sources is exempted on a
        third core C so propagation does not refuse to converge.
        """
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.zeros((1, 2), dtype=np.float64),
            threshold=1.0, latency=0,
        )
        b = NeuralCore(
            id=1, name="B",
            input_sources=_src([(0, 0), (0, 1)]),
            core_matrix=np.array([[1.0], [1.0]], dtype=np.float64),
            threshold=1.0, latency=1,
        )
        # C is the live output target.
        c = NeuralCore(
            id=2, name="C",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        graph = IRGraph(nodes=[a, b, c], output_sources=_src([(2, 0)]))
        result = compute_liveness(graph, simulation_steps=T_DEFAULT)
        assert result.per_node[0] == NodeLiveness.DEAD
        assert result.per_node[1] == NodeLiveness.DEAD
        assert result.per_node[2] == NodeLiveness.LIVE

    def test_compute_op_barrier_does_not_short_circuit_reachability(self):
        """A -> ComputeOp -> B (output). The ComputeOp persistently consumes
        A's neurons, so A stays reachable even though propagation otherwise
        treats column-level pruning conservatively across the barrier.
        """
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0), (-2, 1)]),
            core_matrix=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        relu = ComputeOp(
            id=1, name="relu",
            input_sources=_src([(0, 0), (0, 1)]),
            op_type="identity",
            input_shape=(2,), output_shape=(2,),
        )
        b = NeuralCore(
            id=2, name="B",
            input_sources=_src([(1, 0), (1, 1)]),
            core_matrix=np.array([[1.0], [1.0]], dtype=np.float64),
            threshold=1.0, latency=2,
        )
        graph = IRGraph(nodes=[a, relu, b], output_sources=_src([(2, 0)]))
        result = compute_liveness(graph, simulation_steps=T_DEFAULT)
        assert result.per_node[0] == NodeLiveness.LIVE
        assert result.per_node[2] == NodeLiveness.LIVE


class TestOutputExemption:
    def test_output_exempt_neuron_keeps_core_live(self):
        """A neuron that appears in ``IRGraph.output_sources`` is exempt from
        column pruning and the core that owns it is LIVE.

        Build a core whose only consumer is the model output. With no other
        live consumer, the propagation would normally prune everything; the
        output exemption must keep it live.
        """
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        graph = IRGraph(nodes=[a], output_sources=_src([(0, 0)]))
        result = compute_liveness(graph, simulation_steps=T_DEFAULT)
        assert result.per_node[0] == NodeLiveness.LIVE


class TestEnumValues:
    def test_enum_values_match_plan(self):
        assert NodeLiveness.LIVE.value == "live"
        assert NodeLiveness.BIAS_ONLY.value == "bias_only"
        assert NodeLiveness.DEAD.value == "dead"


class TestEdgeCases:
    def test_empty_graph_returns_empty_result(self):
        graph = IRGraph(nodes=[], output_sources=np.array([], dtype=object))
        result = compute_liveness(graph, simulation_steps=T_DEFAULT)
        assert result.per_node == {}
        assert result.reasons == {}

    def test_simulation_steps_required_positive(self):
        """``simulation_steps`` must be a positive int; 0 / negative makes
        the bias-vs-threshold check meaningless."""
        a = NeuralCore(
            id=0, name="A",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        graph = IRGraph(nodes=[a], output_sources=_src([(0, 0)]))
        with pytest.raises(ValueError):
            compute_liveness(graph, simulation_steps=0)
        with pytest.raises(ValueError):
            compute_liveness(graph, simulation_steps=-1)
