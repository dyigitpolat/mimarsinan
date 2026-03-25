"""Contract tests: unified schedule partitioner produces identical results for
identical shapes regardless of whether the input is NeuralCore or LayoutSoftCoreSpec.

Also tests effective_core_budget and snapshot schedule metadata.
"""

import copy
import numpy as np
import pytest

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, ComputeOp
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.schedule_partitioner import (
    effective_core_budget,
    estimate_passes_for_layout,
    partition_segment_into_passes,
)
from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping


def _make_core(node_id, input_from=None, n_in=4, n_out=2):
    if input_from is None:
        sources = np.array(
            [IRSource(-2, i) for i in range(n_in)] + [IRSource(-3, 0)],
            dtype=object,
        )
    else:
        sources = np.array(
            [IRSource(input_from, i) for i in range(n_in)] + [IRSource(-3, 0)],
            dtype=object,
        )
    w = np.ones((n_in + 1, n_out), dtype=np.float32) * 0.1
    return NeuralCore(id=node_id, name=f"c{node_id}", input_sources=sources, core_matrix=w)


# ---------------------------------------------------------------------------
# effective_core_budget
# ---------------------------------------------------------------------------

class TestEffectiveCoreBudget:
    def test_single_type_full_budget(self):
        cores_config = [{"max_axons": 32, "max_neurons": 32, "count": 100}]
        assert effective_core_budget(cores_config) == 100

    def test_heterogeneous_80_percent(self):
        cores_config = [
            {"max_axons": 32, "max_neurons": 32, "count": 50},
            {"max_axons": 64, "max_neurons": 64, "count": 50},
        ]
        assert effective_core_budget(cores_config) == 80

    def test_empty(self):
        assert effective_core_budget([]) == 0


# ---------------------------------------------------------------------------
# Contract: same shapes → same pass count from both paths
# ---------------------------------------------------------------------------

class TestUnifiedPartitionerConsistency:
    """For identical shapes and budget, partition_segment_into_passes (NeuralCore)
    and estimate_passes_for_layout (LayoutSoftCoreSpec) must produce the same
    number of passes per segment."""

    def test_simple_segment_same_passes(self):
        """Single segment, small budget: both paths produce same pass count."""
        n_cores = 6
        n_in, n_out = 4, 2
        budget = 3

        cores = [_make_core(i, n_in=n_in, n_out=n_out) for i in range(n_cores)]
        neural_passes = partition_segment_into_passes(cores, budget)

        specs = [
            LayoutSoftCoreSpec(
                input_count=n_in + 1, output_count=n_out,
                segment_id=0, latency_tag=0,
            )
            for _ in range(n_cores)
        ]
        layout_n, layout_passes = estimate_passes_for_layout(specs, budget)

        assert len(neural_passes) == layout_n, (
            f"NeuralCore path: {len(neural_passes)} passes, "
            f"Layout path: {layout_n} passes"
        )

    def test_multi_latency_same_passes(self):
        """Two latency levels, tight budget: both paths agree."""
        budget = 2

        c0 = _make_core(0, n_in=4, n_out=2)
        c1 = _make_core(1, n_in=4, n_out=2)
        c2 = _make_core(2, input_from=0, n_in=2, n_out=2)
        c3 = _make_core(3, input_from=1, n_in=2, n_out=2)
        cores = [c0, c1, c2, c3]

        neural_passes = partition_segment_into_passes(cores, budget)

        specs = [
            LayoutSoftCoreSpec(input_count=5, output_count=2, segment_id=0, latency_tag=1),
            LayoutSoftCoreSpec(input_count=5, output_count=2, segment_id=0, latency_tag=1),
            LayoutSoftCoreSpec(input_count=3, output_count=2, segment_id=0, latency_tag=2),
            LayoutSoftCoreSpec(input_count=3, output_count=2, segment_id=0, latency_tag=2),
        ]
        layout_n, layout_passes = estimate_passes_for_layout(specs, budget)

        assert len(neural_passes) == layout_n

    def test_hw_cost_aware_same_passes(self):
        """With hw cost estimation, both paths agree."""
        budget = 4
        hw_kwargs = dict(max_hw_axons=256, max_hw_neurons=256, allow_coalescing=True)

        cores = [_make_core(i, n_in=512, n_out=10) for i in range(4)]
        neural_passes = partition_segment_into_passes(cores, budget, **hw_kwargs)

        specs = [
            LayoutSoftCoreSpec(
                input_count=513, output_count=10,
                segment_id=0, latency_tag=0,
            )
            for _ in range(4)
        ]
        layout_n, _ = estimate_passes_for_layout(specs, budget, **hw_kwargs)

        assert len(neural_passes) == layout_n

    def test_single_core_fits_both_say_one_pass(self):
        cores = [_make_core(0)]
        assert len(partition_segment_into_passes(cores, 10)) == 1

        specs = [LayoutSoftCoreSpec(input_count=5, output_count=2, segment_id=0, latency_tag=0)]
        n, _ = estimate_passes_for_layout(specs, 10)
        assert n == 1


# ---------------------------------------------------------------------------
# Snapshot schedule metadata
# ---------------------------------------------------------------------------

class TestSnapshotScheduleMetadata:
    """snapshot_hard_core_mapping should include schedule indices when scheduling is on."""

    def test_snapshot_includes_schedule_indices(self):
        from mimarsinan.gui.snapshot.builders import snapshot_hard_core_mapping

        c0 = _make_core(0, n_in=4, n_out=2)
        c1 = _make_core(1, n_in=4, n_out=2)
        c2 = _make_core(2, n_in=4, n_out=2)
        c3 = _make_core(3, n_in=4, n_out=2)
        c4 = _make_core(4, n_in=4, n_out=2)
        c5 = _make_core(5, n_in=4, n_out=2)

        ir_graph = IRGraph(
            nodes=[c0, c1, c2, c3, c4, c5],
            output_sources=np.array([IRSource(5, 0), IRSource(5, 1)], dtype=object),
            weight_banks={},
        )

        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir_graph,
            cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 2, "has_bias": True}],
            allow_scheduling=True,
        )

        has_scheduled_stages = any(
            s.schedule_pass_index is not None for s in hm.stages
        )
        if not has_scheduled_stages:
            pytest.skip("No multi-pass needed for this config")

        snap = snapshot_hard_core_mapping(hm)
        for stage in snap["stages"]:
            if stage["kind"] == "neural" and "schedule_pass_index" in stage:
                assert isinstance(stage["schedule_pass_index"], int)
                assert isinstance(stage["schedule_segment_index"], int)
                return

        pytest.fail("No neural stage with schedule metadata found in snapshot")

    def test_snapshot_no_schedule_when_single_pool(self):
        from mimarsinan.gui.snapshot.builders import snapshot_hard_core_mapping

        cores = [_make_core(i) for i in range(3)]
        ir_graph = IRGraph(
            nodes=cores,
            output_sources=np.array([IRSource(2, 0), IRSource(2, 1)], dtype=object),
            weight_banks={},
        )

        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir_graph,
            cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 100, "has_bias": True}],
            allow_scheduling=False,
        )

        snap = snapshot_hard_core_mapping(hm)
        for stage in snap["stages"]:
            if stage["kind"] == "neural":
                assert "schedule_pass_index" not in stage
                assert "schedule_segment_index" not in stage
