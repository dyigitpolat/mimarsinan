"""Segment mapping is single-pass-or-crash.

After removing latency-group pass splitting from the hard-core mapper, the
contract is:

* Each neural segment identified by the layout mapper (runs of NeuralCores
  between ComputeOp sync barriers) becomes **exactly one** ``HybridStage``
  of ``kind="neural"``.
* If the segment's combined cores cannot pack onto the available hardware
  pool, ``build_hybrid_hard_core_mapping`` propagates the packer's
  ``RuntimeError`` — no silent sub-pass splitting.
"""

import numpy as np
import pytest

from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore


def _make_core(node_id, input_from=None, n_in=4, n_out=2, latency=None):
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
    core = NeuralCore(
        id=node_id, name=f"c{node_id}",
        input_sources=sources, core_matrix=w,
    )
    if latency is not None:
        core.latency = latency
    return core


def _single_chain_ir(n_cores: int) -> IRGraph:
    """Chain of ``n_cores`` NeuralCores with increasing latency (0..n-1)."""
    cores = []
    for i in range(n_cores):
        cores.append(
            _make_core(
                i,
                input_from=(cores[-1].id if cores else None),
                latency=i,
            )
        )
    out_sources = np.array(
        [IRSource(cores[-1].id, j) for j in range(2)], dtype=object,
    )
    return IRGraph(nodes=cores, output_sources=out_sources)


def _graph_with_compute_op_barriers(n_per_seg: int, n_segments: int) -> IRGraph:
    """``n_segments`` chains of ``n_per_seg`` cores, with ComputeOp sync barriers
    between them.  Used to verify that ComputeOps still produce segment
    boundaries even though latency alone no longer does.
    """
    nodes = []
    next_id = 0
    prev_output_id = None
    for s in range(n_segments):
        for i in range(n_per_seg):
            nodes.append(
                _make_core(
                    next_id,
                    input_from=prev_output_id if (s == 0 and i == 0) and prev_output_id is None
                               else (nodes[-1].id if nodes else None),
                    latency=i,
                )
            )
            next_id += 1
            prev_output_id = nodes[-1].id
        if s < n_segments - 1:
            # ComputeOp barrier
            op = ComputeOp(
                id=next_id,
                name=f"barrier_{s}",
                input_sources=np.array([IRSource(prev_output_id, j) for j in range(2)], dtype=object),
                op_type="flatten",
                output_shape=(2,),
            )
            nodes.append(op)
            prev_output_id = op.id
            next_id += 1
    out_sources = np.array([IRSource(prev_output_id, j) for j in range(2)], dtype=object)
    return IRGraph(nodes=nodes, output_sources=out_sources)


class TestSinglePassSegments:
    """Segments always produce exactly one neural stage."""

    def test_single_segment_many_latencies_one_stage(self):
        """A 6-core chain with 6 distinct latencies must pack as ONE stage."""
        ir = _single_chain_ir(n_cores=6)
        cores_config = [{"max_axons": 16, "max_neurons": 16, "count": 10}]
        mapping = build_hybrid_hard_core_mapping(
            ir_graph=ir,
            cores_config=cores_config,
            allow_neuron_splitting=False,
            allow_scheduling=True,
        )
        neural_stages = [s for s in mapping.stages if s.kind == "neural"]
        assert len(neural_stages) == 1, (
            f"Expected 1 neural stage for a single no-barrier segment, "
            f"got {len(neural_stages)}: "
            f"{[s.name for s in neural_stages]}"
        )

    def test_barrier_separated_segments_each_one_stage(self):
        """Two segments separated by a ComputeOp → exactly two neural stages."""
        ir = _graph_with_compute_op_barriers(n_per_seg=4, n_segments=2)
        cores_config = [{"max_axons": 16, "max_neurons": 16, "count": 10}]
        mapping = build_hybrid_hard_core_mapping(
            ir_graph=ir,
            cores_config=cores_config,
            allow_neuron_splitting=False,
            allow_scheduling=True,
        )
        neural_stages = [s for s in mapping.stages if s.kind == "neural"]
        compute_stages = [s for s in mapping.stages if s.kind == "compute"]
        assert len(neural_stages) == 2, (
            f"Expected 2 neural stages (one per barrier-separated segment), "
            f"got {len(neural_stages)}"
        )
        assert len(compute_stages) == 1, (
            f"Expected 1 compute stage (the barrier), got {len(compute_stages)}"
        )


class TestCapacityDrivenSyncBarriers:
    """When a segment's latency chain exceeds the hardware pool, the
    builder inserts implicit sync barriers — additional
    ``HybridStage(kind="neural")`` entries — so each stage packs in one
    pass.  This is the SimpleMLP-style case: 3 Perceptrons on 2 cores
    should map to 2 + 1 layers across two neural stages.
    """

    def test_three_layer_chain_on_two_cores_splits_into_two_stages(self):
        ir = _single_chain_ir(n_cores=3)  # latencies 0, 1, 2
        cores_config = [{"max_axons": 16, "max_neurons": 16, "count": 2}]
        mapping = build_hybrid_hard_core_mapping(
            ir_graph=ir,
            cores_config=cores_config,
            allow_neuron_splitting=False,
            allow_scheduling=True,
        )
        neural_stages = [s for s in mapping.stages if s.kind == "neural"]
        assert len(neural_stages) == 2, (
            f"Expected 2 neural stages for 3-core chain on 2-core pool, "
            f"got {len(neural_stages)}"
        )
        # Stage 1 should hold the first two latency groups (2 cores), stage
        # 2 should hold the third.  Sizes check against that.
        s1_cores = len(neural_stages[0].hard_core_mapping.cores)
        s2_cores = len(neural_stages[1].hard_core_mapping.cores)
        assert s1_cores == 2 and s2_cores == 1, (
            f"Expected 2 + 1 core distribution, got {s1_cores} + {s2_cores}"
        )

    def test_fits_in_one_pass_when_pool_is_big_enough(self):
        """Control: 3-core chain on a 4-core pool is still one stage."""
        ir = _single_chain_ir(n_cores=3)
        cores_config = [{"max_axons": 16, "max_neurons": 16, "count": 4}]
        mapping = build_hybrid_hard_core_mapping(
            ir_graph=ir,
            cores_config=cores_config,
            allow_neuron_splitting=False,
            allow_scheduling=True,
        )
        neural_stages = [s for s in mapping.stages if s.kind == "neural"]
        assert len(neural_stages) == 1


class TestWizardCapacityEstimator:
    """The wizard's ``estimate_passes_for_layout_validated`` must mirror the
    hard-core mapper's capacity splitting, so a config that the mapper
    can deploy is reported as *feasible* by the wizard and surfaces the
    expanded pass count."""

    def test_estimator_matches_hard_core_mapper_split(self):
        from mimarsinan.mapping.layout.layout_types import (
            LayoutHardCoreType, LayoutSoftCoreSpec,
        )
        from mimarsinan.mapping.schedule_partitioner import (
            estimate_passes_for_layout_validated,
        )

        # Three-layer chain emulating SimpleMLP: latency 0, 1, 2, same seg.
        softcores = [
            LayoutSoftCoreSpec(
                input_count=4, output_count=4,
                threshold_group_id=i, latency_tag=i, segment_id=0,
                name=f"p{i}",
            )
            for i in range(3)
        ]
        # 2-core pool (both core types equal to simplify).
        core_types = [LayoutHardCoreType(max_axons=16, max_neurons=16, count=2)]

        n_passes, pass_lists, ok = estimate_passes_for_layout_validated(
            softcores, max_cores_per_pass=2,
            max_hw_axons=16, max_hw_neurons=16,
            allow_coalescing=False, allow_splitting=False,
            core_types=core_types,
        )
        assert ok, "Capacity-split should rescue 3-on-2 into 2+1 sub-segments"
        assert n_passes == 2, f"Expected 2 passes post-split, got {n_passes}"
        assert [len(p) for p in pass_lists] == [2, 1]

    def test_estimator_reports_infeasible_when_softcore_exceeds_every_hw_type(self):
        """Genuine infeasibility: a softcore whose dimensions exceed all
        core types (and no splitting/coalescing is enabled).  Halving
        cannot help because the unit softcore itself does not fit."""
        from mimarsinan.mapping.layout.layout_types import (
            LayoutHardCoreType, LayoutSoftCoreSpec,
        )
        from mimarsinan.mapping.schedule_partitioner import (
            estimate_passes_for_layout_validated,
        )

        softcores = [
            LayoutSoftCoreSpec(
                input_count=64, output_count=4,
                threshold_group_id=0, latency_tag=0, segment_id=0,
                name="oversized",
            )
        ]
        core_types = [LayoutHardCoreType(max_axons=16, max_neurons=16, count=5)]
        _, _, ok = estimate_passes_for_layout_validated(
            softcores, max_cores_per_pass=5,
            max_hw_axons=16, max_hw_neurons=16,
            allow_coalescing=False, allow_splitting=False,
            core_types=core_types,
        )
        assert not ok, "Softcore too wide for every hw type must flag infeasible"

    def test_estimator_rescues_large_latency_group_via_halving(self):
        """Scheduling rescues a single latency group by halving: 10 cores
        on a 2-core pool fit when spread across passes."""
        from mimarsinan.mapping.layout.layout_types import (
            LayoutHardCoreType, LayoutSoftCoreSpec,
        )
        from mimarsinan.mapping.schedule_partitioner import (
            estimate_passes_for_layout_validated,
        )
        softcores = [
            LayoutSoftCoreSpec(
                input_count=4, output_count=4,
                threshold_group_id=i, latency_tag=0, segment_id=0,
                name=f"p{i}",
            )
            for i in range(10)
        ]
        core_types = [LayoutHardCoreType(max_axons=16, max_neurons=16, count=2)]
        n_passes, pass_lists, ok = estimate_passes_for_layout_validated(
            softcores, max_cores_per_pass=2,
            max_hw_axons=16, max_hw_neurons=16,
            allow_coalescing=False, allow_splitting=False,
            core_types=core_types,
        )
        assert ok, "Halving within a latency group should rescue infeasibility"
        assert n_passes > 1, "Expect a multi-pass schedule for oversized group"
