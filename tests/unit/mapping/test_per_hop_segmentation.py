"""Per-hop neural segmentation (C3): count-exact re-timing at every hop.

Splitting a deep single-segment chain into per-hop neural segments makes every
hop boundary a decode/re-encode: the transcode is count-preserving
(``round((c/T)*T) = c``) and RESETS arrival timing, killing the back-loading
deficit (+1.9pp at chain9 S=4). Mixer-class vehicles already get this at their
ComputeOp boundaries; this is the mapping-level option for chains.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.layout.segmentation import (
    HostSegment,
    NeuralSegment,
    partition_ir_graph,
)
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

T = 8
N = 2


def _core(node_id, name, sources, matrix, **kwargs):
    return NeuralCore(
        id=node_id,
        name=name,
        input_sources=np.asarray(sources, dtype=object),
        core_matrix=np.asarray(matrix, dtype=np.float64),
        threshold=1.0,
        parameter_scale=torch.tensor(0.0),
        **kwargs,
    )


def _chain_graph(weights=(1.0, 0.6, 0.9)) -> IRGraph:
    nodes = []
    for depth, w in enumerate(weights):
        src = -2 if depth == 0 else depth - 1
        nodes.append(
            _core(depth, f"L{depth}", [IRSource(src, i) for i in range(N)],
                  np.eye(N) * w)
        )
    return IRGraph(
        nodes=nodes,
        output_sources=np.asarray(
            [IRSource(len(weights) - 1, i) for i in range(N)], dtype=object,
        ),
    )


class TestPartitionFlagOff:
    def test_default_partition_is_single_maximal_run(self):
        segments = partition_ir_graph(_chain_graph())
        assert len(segments) == 1
        assert isinstance(segments[0], NeuralSegment)
        assert len(segments[0].nodes) == 3
        assert segments[0].label == "neural_segment_final"

    def test_per_hop_false_is_byte_identical(self):
        graph = _chain_graph()
        default = partition_ir_graph(graph)
        explicit = partition_ir_graph(graph, per_hop=False)
        assert [type(s) for s in default] == [type(s) for s in explicit]
        assert [
            [n.id for n in s.nodes] for s in default
        ] == [[n.id for n in s.nodes] for s in explicit]


class TestPartitionPerHop:
    def test_chain_splits_into_one_segment_per_hop(self):
        segments = partition_ir_graph(_chain_graph(), per_hop=True)
        assert len(segments) == 3
        assert all(isinstance(s, NeuralSegment) for s in segments)
        assert [[n.id for n in s.nodes] for s in segments] == [[0], [1], [2]]
        assert [s.label for s in segments] == [
            "neural_segment_final_hop0",
            "neural_segment_final_hop1",
            "neural_segment_final_hop2",
        ]

    def test_same_depth_fan_in_stays_grouped(self):
        """Two parallel depth-1 cores share one hop segment."""
        eye = np.eye(N)
        a = _core(0, "A", [IRSource(-2, i) for i in range(N)], eye)
        b1 = _core(1, "B1", [IRSource(0, i) for i in range(N)], eye)
        b2 = _core(2, "B2", [IRSource(0, i) for i in range(N)], eye)
        c = _core(
            3, "C",
            [IRSource(1, i) for i in range(N)] + [IRSource(2, i) for i in range(N)],
            np.vstack([eye, eye]) * 0.4,
        )
        graph = IRGraph(
            nodes=[a, b1, b2, c],
            output_sources=np.asarray([IRSource(3, i) for i in range(N)], dtype=object),
        )
        segments = partition_ir_graph(graph, per_hop=True)
        assert [[n.id for n in s.nodes] for s in segments] == [[0], [1, 2], [3]]

    def test_compute_op_barriers_are_preserved(self):
        eye = np.eye(N)
        a = _core(0, "A", [IRSource(-2, i) for i in range(N)], eye)
        op = ComputeOp(
            id=1, name="host_op",
            input_sources=np.asarray([IRSource(0, i) for i in range(N)], dtype=object),
            op_type="identity", params={"module": nn.Identity()},
        )
        b = _core(2, "B", [IRSource(1, i) for i in range(N)], eye * 0.5)
        graph = IRGraph(
            nodes=[a, op, b],
            output_sources=np.asarray([IRSource(2, i) for i in range(N)], dtype=object),
        )
        segments = partition_ir_graph(graph, per_hop=True)
        assert [type(s) for s in segments] == [NeuralSegment, HostSegment, NeuralSegment]

    def test_coalescing_groups_keep_segment_whole(self):
        """Partial-sum / coalescing groups must not be split across boundaries
        (membrane transfer is not a decode/re-encode); the segment stays whole."""
        graph = _chain_graph()
        graph.nodes[1].coalescing_group_id = 7
        segments = partition_ir_graph(graph, per_hop=True)
        assert len(segments) == 1
        assert len(segments[0].nodes) == 3


class TestPerHopBuildAndExecution:
    def _flow(self, hybrid) -> SpikingHybridCoreFlow:
        return SpikingHybridCoreFlow(
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

    def _build(self, per_hop: bool):
        cores = [{"max_axons": 16, "max_neurons": 16, "count": 32}]
        return build_hybrid_hard_core_mapping(
            ir_graph=_chain_graph(),
            cores_config=cores,
            per_hop_neural_segments=per_hop,
        )

    def test_per_hop_build_has_one_neural_stage_per_hop(self):
        hybrid = self._build(per_hop=True)
        assert [s.kind for s in hybrid.stages] == ["neural"] * 3
        baseline = self._build(per_hop=False)
        assert [s.kind for s in baseline.stages] == ["neural"]

    def test_constant_drive_counts_match_unsplit_chain(self):
        """For constant drives (Theorem-2-exact both ways) the per-hop split is
        count-exact: the boundary transcode preserves every count."""
        split = self._flow(self._build(per_hop=True))
        whole = self._flow(self._build(per_hop=False))
        for rate in (0.25, 0.5, 0.75, 1.0):
            x = torch.full((1, N), rate, dtype=torch.float32)
            with torch.no_grad():
                out_split = split(x)
                out_whole = whole(x)
            assert torch.equal(out_split, out_whole), (
                f"rate={rate}: per-hop {out_split.tolist()} != "
                f"single-segment {out_whole.tolist()}"
            )


def _flow_for(hybrid, input_size=N) -> SpikingHybridCoreFlow:
    return SpikingHybridCoreFlow(
        input_shape=(input_size,),
        hybrid_mapping=hybrid,
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode="Default",
        spike_mode="Uniform",
        thresholding_mode="<=",
        spiking_mode="lif",
        cycle_accurate_lif_forward=True,
    ).eval()


def _counts_and_logits(flow, x):
    captured: list[torch.Tensor] = []
    flow.stage_count_recorder = (
        lambda stage, counts: captured.append(counts.detach().clone())
    )
    try:
        with torch.no_grad():
            out = flow(x)
    finally:
        flow.stage_count_recorder = None
    return captured, out


def _fan_in_graph() -> IRGraph:
    """Depth-0 → two parallel depth-1 cores → depth-2 join (rhythm-sensitive)."""
    eye = np.eye(N)
    a = _core(0, "A", [IRSource(-2, i) for i in range(N)], eye * 0.9)
    b1 = _core(1, "B1", [IRSource(0, i) for i in range(N)], eye * 0.7)
    b2 = _core(2, "B2", [IRSource(0, i) for i in range(N)], eye * 0.4)
    c = _core(
        3, "C",
        [IRSource(1, i) for i in range(N)] + [IRSource(2, i) for i in range(N)],
        np.vstack([eye, eye]) * 0.6,
    )
    return IRGraph(
        nodes=[a, b1, b2, c],
        output_sources=np.asarray([IRSource(3, i) for i in range(N)], dtype=object),
    )


class TestRetimedLevelStages:
    """[fused mapping] The mapping stays fused (one honest neural stage);
    when re-timing is armed the stage carries per-level execution stages —
    structurally the split builder's hop stages — and every executor runs
    them through its existing per-stage path. Counts and logits must be
    bit-equal to the split build."""

    CORES = [{"max_axons": 16, "max_neurons": 16, "count": 32}]

    def _build(self, graph, *, per_hop=False, retimed=False):
        return build_hybrid_hard_core_mapping(
            ir_graph=graph,
            cores_config=self.CORES,
            per_hop_neural_segments=per_hop,
            retimed_level_stages=retimed,
        )

    def test_fused_build_carries_level_stages(self):
        hybrid = self._build(_chain_graph(), retimed=True)
        assert [s.kind for s in hybrid.stages] == ["neural"]
        levels = hybrid.stages[0].retimed_level_stages
        assert levels is not None and len(levels) == 3
        assert [ls.name for ls in levels] == [
            "neural_segment_final_hop0",
            "neural_segment_final_hop1",
            "neural_segment_final_hop2",
        ]
        assert [[s.node_id for s in ls.output_map] for ls in levels] == [
            [0], [1], [2],
        ]

    def test_unarmed_build_attaches_nothing(self):
        hybrid = self._build(_chain_graph())
        assert hybrid.stages[0].retimed_level_stages is None

    def test_single_level_segment_attaches_nothing(self):
        graph = _chain_graph(weights=(1.0,))
        hybrid = self._build(graph, retimed=True)
        assert hybrid.stages[0].retimed_level_stages is None

    def test_coalescing_groups_refuse_levels_like_the_split(self):
        graph = _chain_graph()
        graph.nodes[1].coalescing_group_id = 7
        hybrid = self._build(graph, retimed=True)
        assert hybrid.stages[0].retimed_level_stages is None

    def _assert_differential(self, graph, inputs):
        split = _flow_for(self._build(graph, per_hop=True))
        fused = _flow_for(self._build(graph, retimed=True))
        for x in inputs:
            split_counts, split_out = _counts_and_logits(split, x)
            fused_counts, fused_out = _counts_and_logits(fused, x)
            assert len(split_counts) == len(fused_counts)
            for hop, (a, b) in enumerate(zip(split_counts, fused_counts)):
                assert torch.equal(a, b), (
                    f"hop {hop}: split {a.tolist()} != fused-levels {b.tolist()}"
                )
            assert torch.equal(split_out, fused_out)

    def _random_inputs(self, n=4, batch=3):
        gen = torch.Generator().manual_seed(7)
        return [
            torch.randint(0, T + 1, (batch, N), generator=gen).float() / T
            for _ in range(n)
        ]

    def test_chain_counts_and_logits_bit_equal_split(self):
        self._assert_differential(_chain_graph(), self._random_inputs())

    def test_fan_in_counts_and_logits_bit_equal_split(self):
        self._assert_differential(_fan_in_graph(), self._random_inputs())

    def test_hardware_bias_chain_bit_equal_split(self):
        graph = _chain_graph(weights=(0.8, 0.5))
        for node in graph.nodes:
            node.hardware_bias = np.full(N, 0.125, dtype=np.float64)
        self._assert_differential(graph, self._random_inputs())

    def test_recording_runs_per_level_with_hop_names(self):
        from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import (
            enumerate_execution_stages,
        )

        hybrid = self._build(_chain_graph(), retimed=True)
        fused = _flow_for(hybrid)
        x = torch.full((1, N), 0.5, dtype=torch.float32)
        out, record = fused.forward_with_recording(x)
        names = [seg.stage_name for seg in record.segments.values()]
        assert names == [
            "neural_segment_final_hop0",
            "neural_segment_final_hop1",
            "neural_segment_final_hop2",
        ]
        # reference-driven runners share the recorded index arithmetic.
        expected_indices = [
            idx for idx, _stage, exec_stage
            in enumerate_execution_stages(hybrid.stages)
            if exec_stage.kind == "neural"
        ]
        assert sorted(record.segments.keys()) == expected_indices
        split = _flow_for(self._build(_chain_graph(), per_hop=True))
        out_split, record_split = split.forward_with_recording(x)
        assert torch.equal(out, out_split)
        fused_out_counts = [
            seg.seg_output_spike_count.tolist()
            for seg in record.segments.values()
        ]
        split_out_counts = [
            seg.seg_output_spike_count.tolist()
            for seg in record_split.segments.values()
        ]
        assert fused_out_counts == split_out_counts
