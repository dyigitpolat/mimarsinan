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
