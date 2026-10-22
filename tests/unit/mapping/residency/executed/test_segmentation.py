"""The builder's segments under the key: the executed plan runs on the shape-only plane's segments
(``depth_partition``, so both planes plan one program) only where they keep the class of every neural-to-neural edge
of the reference segmentation (``partition_ir_graph``, which the identity and U1 builders execute). Where an edge
changes class the key refuses by name, because a raster streamed inside a segment and a transcode at a segment
boundary are different functions; where only cores move, every core reads the same producers through the same kind
of boundary, and the executed plan equals the identity run exactly."""

from __future__ import annotations

import pytest
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.layout.segmentation import NeuralSegment, partition_ir_graph
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.packing.schedule_residency_plan import depth_partition
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.mapping.support.schedule.pass_carry import carried_outputs_by_stage

from . import vehicles as V

PLANS = [{"step": "floor"}, {"step": "stateless"}]


def _groups(segments):
    return [sorted(int(n.id) for n in s.nodes) for s in segments if isinstance(s, NeuralSegment)]


class TestAnEdgeThatChangesClassRefusesByName:
    @pytest.mark.parametrize("count", [2, 4, 500])
    @pytest.mark.parametrize("plan", PLANS)
    def test_the_key_refuses_naming_the_moved_cores_and_the_edge(self, count, plan):
        ir = V.ir_of(V.branchy_repr("edge_class"))
        a, _b, c, _d, _out = [int(n.id) for n in ir.get_neural_cores()]
        assert _groups(partition_ir_graph(ir)) != _groups(depth_partition(ir))
        with pytest.raises(ResidencyPlanError, match="change class") as caught:
            V.build(ir, count, plan)
        assert isinstance(caught.value, ToolchainRefusal)
        message = str(caught.value)
        assert f"{a}->{c}" in message and f"moves cores [{c}]" in message
        assert f"{a}->{c} (a segment boundary in the reference, streamed re-segmented)" in message

    @pytest.mark.parametrize("count", [2, 3, 500])
    @pytest.mark.parametrize("plan", PLANS)
    def test_a_streamed_edge_that_becomes_a_segment_boundary_refuses_by_name(self, count, plan):
        """The other direction: only P -> C changes, streamed in the reference and a boundary re-segmented (a run
        that admitted it would differ from the identity run by max |delta| 1.0)."""
        ir = V.ir_of(V.branchy_repr("intra_to_cross"))
        _q, p, c, _out = [int(n.id) for n in ir.get_neural_cores()]
        assert _groups(partition_ir_graph(ir)) != _groups(depth_partition(ir))
        with pytest.raises(ResidencyPlanError, match="change class") as caught:
            V.build(ir, count, plan)
        assert isinstance(caught.value, ToolchainRefusal)
        message = str(caught.value)
        assert f"{p}->{c} (streamed in the reference, a segment boundary re-segmented)" in message
        assert f"moves cores [{p}]" in message


class TestCoresThatMoveWithEveryEdgeClassKept:
    def test_the_two_segmentations_group_the_cores_differently(self):
        ir = V.ir_of(V.branchy_repr("cores_only"))
        assert _groups(partition_ir_graph(ir)) != _groups(depth_partition(ir))

    @pytest.mark.parametrize("count", [3, 4, 500])
    @pytest.mark.parametrize("plan", PLANS)
    def test_the_executed_plan_equals_the_identity_run_exactly(self, count, plan):
        ir = V.ir_of(V.branchy_repr("cores_only"))
        identity = V.lif_flow(build_identity_hybrid_mapping(ir_graph=ir))
        mapping = V.build(ir, count, plan)
        if count == 3:
            assert carried_outputs_by_stage(mapping.stages)                  # a cut inside a segment carries
        executed = V.lif_flow(mapping)
        outputs = []
        for batch in range(4):
            x = torch.rand(6, V.IN, generator=torch.Generator().manual_seed(batch))
            with torch.no_grad():
                want = identity(x)
                assert torch.equal(executed(x), want)
            outputs.append(want)
        assert float(torch.cat(outputs).std(0).max()) > 0                   # the vehicle discriminates

    @pytest.mark.parametrize("count", [3, 4])
    def test_both_planes_plan_the_re_segmented_program(self, count):
        rep = V.branchy_repr("cores_only")
        specs, census, program = V.layout_plane(rep)
        ir = V.ir_of(rep)
        hybrid = V.build(ir, count, {"step": "floor"})
        frag = V.fragments(specs, census, count, {"step": "floor"}, program=program)
        assert V.built_pass_layers(hybrid, ir) == V.fragment_pass_layers(frag, specs)
        assert ([s.schedule_resident_cores for s in V.neural_passes(hybrid)]
                == [p.schedule_resident_cores for p in frag.pass_programs])
