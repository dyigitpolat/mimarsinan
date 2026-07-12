"""Dry-run placement feasibility: run the REAL hybrid packer, no GPU, no sim.

``estimate_cores_needed`` is a SOUND LOWER bound — it admits any config whose
ideal diagonal packing fits the budget. But the greedy packer keeps each
threshold group (a softcore's perceptron index, a STRUCTURAL/weight-independent
property) on its own hard cores, so a config can pass the lower bound yet crash
late with ``RuntimeError("No more hard cores available")``. ``dryrun_pack_feasible``
runs the actual packer and gives the early, definitive verdict the lower bound
cannot — catching the per-perceptron fragmentation the bound ignores.
"""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.verification.capacity import (
    PackFeasibility,
    dryrun_pack_feasible,
    estimate_cores_needed,
)


def _src(specs):
    return np.array(
        [IRSource(node_id=nid, index=idx) for nid, idx in specs],
        dtype=object,
    )


def _core(node_id, name, in_count, out_count, perceptron_index=None):
    """A NeuralCore of ``in_count`` axons × ``out_count`` neurons.

    ``perceptron_index`` becomes the softcore's threshold group: distinct indices
    force the packer onto separate hard cores (it cannot mix groups on one core).
    """
    core = NeuralCore(
        id=node_id,
        name=name,
        input_sources=_src([(-2, i) for i in range(in_count)]),
        core_matrix=np.ones((in_count, out_count), dtype=np.float64),
        threshold=1.0,
        latency=0,
    )
    if perceptron_index is not None:
        core.perceptron_index = perceptron_index
    return core


def _cores(max_axons, max_neurons, count):
    return [{"max_axons": max_axons, "max_neurons": max_neurons, "count": count,
             "has_bias": True}]


class TestFeasiblePack:
    def test_single_fitting_core_is_feasible_one_hard_core(self):
        graph = IRGraph(nodes=[_core(0, "A", 4, 4)], output_sources=_src([(0, 0)]))
        pf = dryrun_pack_feasible(graph, {"cores": _cores(8, 8, 16)})
        assert isinstance(pf, PackFeasibility)
        assert pf.feasible is True
        assert pf.hard_cores == 1
        assert pf.overflowing_segment is None
        assert pf.error is None

    def test_copacking_uses_fewer_hard_cores_than_softcores(self):
        """Cores in the SAME threshold group (one perceptron index — a conv layer's
        spatial positions) co-pack along the diagonal, not one-per-softcore."""
        nodes = [_core(i, f"C{i}", 2, 3, perceptron_index=0) for i in range(10)]
        graph = IRGraph(nodes=nodes, output_sources=_src([(9, 0)]))
        pf = dryrun_pack_feasible(graph, {"cores": _cores(8, 8, 64)})
        assert pf.feasible is True
        assert pf.hard_cores < 10  # co-packed, not one-per-softcore
        assert pf.hard_cores >= 1


class TestInfeasiblePack:
    def test_oversized_softcore_is_infeasible_naming_segment(self):
        huge = _core(0, "huge", 80, 80)
        graph = IRGraph(nodes=[huge], output_sources=_src([(0, 0)]))
        pf = dryrun_pack_feasible(graph, {"cores": _cores(8, 8, 2)})
        assert pf.feasible is False
        assert pf.hard_cores is None
        assert pf.overflowing_segment == "neural_segment_final"
        assert "No more hard cores available" in pf.error


class TestThresholdGroupFragmentationGap:
    """The reason the dry-run exists: the SOUND lower bound admits, the packer rejects."""

    def _distinct_group_graph(self, n):
        nodes = [_core(i, f"P{i}", 2, 2, perceptron_index=i) for i in range(n)]
        return IRGraph(nodes=nodes, output_sources=_src([(n - 1, 0)]))

    def test_lower_bound_admits_but_packer_rejects(self):
        # 8 cores, each a distinct threshold group, on a 4-core budget.
        # Lower bound mixes groups → ceil(16 axons / 8) = 2 ≤ 4 → ADMITS.
        # Real packer keeps each group separate → 8 cores > 4 → REJECTS.
        graph = self._distinct_group_graph(8)
        constraints = {"cores": _cores(8, 8, 4)}

        est = estimate_cores_needed(graph, constraints)
        assert est.feasible is True  # the loose lower bound is fooled
        assert est.cores_needed <= 4

        pf = dryrun_pack_feasible(graph, constraints)
        assert pf.feasible is False  # the real packer is not
        assert pf.overflowing_segment is not None
        assert "No more hard cores available" in pf.error

    def test_same_groups_fit_where_distinct_groups_do_not(self):
        """Control: the SAME 8 cores in ONE group co-pack and fit the 4-core budget."""
        nodes = [_core(i, f"S{i}", 2, 2, perceptron_index=0) for i in range(8)]
        graph = IRGraph(nodes=nodes, output_sources=_src([(7, 0)]))
        pf = dryrun_pack_feasible(graph, {"cores": _cores(8, 8, 4)})
        assert pf.feasible is True
        assert pf.hard_cores <= 4


class TestPerChannelThetaPacking:
    """[R3/S2] the per-channel-theta packing prerequisite: threshold groups are
    keyed on perceptron IDENTITY (softcore_spec perceptron_index), and the SCM
    threshold stays 1.0 in effective coordinates — a per-channel activation_scale
    vector must therefore change NOTHING in the packing verdict."""

    def _with_per_channel_theta(self, graph):
        import torch

        for node in graph.nodes:
            n_out = node.core_matrix.shape[1]
            node.activation_scale = torch.linspace(0.5, 2.0, n_out)
        return graph

    def _distinct_group_graph(self, n):
        nodes = [_core(i, f"P{i}", 2, 2, perceptron_index=i) for i in range(n)]
        return IRGraph(nodes=nodes, output_sources=_src([(n - 1, 0)]))

    def test_vector_theta_verdict_identical_to_scalar(self):
        constraints = {"cores": _cores(8, 8, 64)}
        scalar = dryrun_pack_feasible(self._distinct_group_graph(8), constraints)
        vector = dryrun_pack_feasible(
            self._with_per_channel_theta(self._distinct_group_graph(8)),
            constraints,
        )
        assert scalar.feasible and vector.feasible
        assert vector.hard_cores == scalar.hard_cores

    def test_vector_theta_does_not_multiply_threshold_groups(self):
        # Same-group co-packing must survive per-channel theta: if the packer
        # keyed groups on theta VALUES, these 8 same-index cores would
        # fragment onto >4 hard cores and reject.
        import torch

        nodes = [_core(i, f"S{i}", 2, 2, perceptron_index=0) for i in range(8)]
        for node in nodes:
            node.activation_scale = torch.linspace(0.5, 2.0, 2)
        graph = IRGraph(nodes=nodes, output_sources=_src([(7, 0)]))
        pf = dryrun_pack_feasible(graph, {"cores": _cores(8, 8, 4)})
        assert pf.feasible is True
        assert pf.hard_cores <= 4

    def test_mixer_shaped_graph_with_vector_theta_packs(self):
        from unit.mapping.test_identity_hybrid_mapping import (
            _make_mini_mixer_ir_graph,
        )
        import torch

        constraints = {"cores": _cores(64, 64, 32)}
        ir_scalar, _ = _make_mini_mixer_ir_graph()
        scalar = dryrun_pack_feasible(ir_scalar, constraints)

        ir_vector, _ = _make_mini_mixer_ir_graph()
        for node in ir_vector.get_neural_cores():
            node.activation_scale = torch.linspace(
                0.5, 2.0, node.get_output_count()
            )
        vector = dryrun_pack_feasible(ir_vector, constraints)

        assert scalar.feasible and vector.feasible
        assert vector.hard_cores == scalar.hard_cores


class TestEmptyAndEdge:
    def test_non_capacity_error_propagates_not_swallowed_as_infeasible(self):
        """A structural error (empty graph → no stages) is NOT mislabeled a capacity
        rejection — it propagates so the scheduler's non-fatal wrapper admits the job
        rather than the dry-run wrongly dropping it as infeasible."""
        graph = IRGraph(nodes=[], output_sources=_src([(-1, 0)]))
        with pytest.raises(ValueError):
            dryrun_pack_feasible(graph, {"cores": _cores(8, 8, 4)})

    def test_pack_feasibility_is_frozen(self):
        graph = IRGraph(nodes=[_core(0, "A", 2, 2)], output_sources=_src([(0, 0)]))
        pf = dryrun_pack_feasible(graph, {"cores": _cores(8, 8, 4)})
        with pytest.raises(Exception):
            pf.feasible = False
