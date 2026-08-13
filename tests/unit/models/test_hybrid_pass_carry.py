"""A pass boundary inside a streamed segment is semantically invisible.

The load-bearing claim of multi-pass scheduling under streamed LIF: cutting a segment
into passes must not change what it computes. That is a test, not an argument — the
same model on a grid that fits and on a grid that forces a cut must agree bit for bit.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

T = 8


def _lif(width, fan_in, scale):
    p = Perceptron(width, fan_in, normalization=nn.Identity())
    p.base_activation = LIFActivation(T=T, activation_scale=torch.tensor(1.0))
    p.activation = p.base_activation
    with torch.no_grad():
        p.layer.weight.copy_(torch.rand(width, fan_in) * scale)
        if p.layer.bias is not None:
            p.layer.bias.zero_()
    return p


def _deep_lif_ir(seed: int = 0):
    """Encode -> three LIF hops with mid-range firing, so the segment has latency
    groups to cut between AND real rhythm on the wires between them.

    Weights are set explicitly: untrained random ones saturate every neuron, and a
    saturated raster IS its own uniform re-encode, so such a vehicle cannot witness
    the carry at all.
    """
    torch.manual_seed(seed)
    inp = InputMapper((8,))
    enc = _lif(8, 8, 0.5)
    enc.is_encoding_layer = True
    enc.use_cycle_accurate_trains = True
    node = PerceptronMapper(inp, enc)
    for width in (8, 8, 4):
        node = PerceptronMapper(node, _lif(width, 8, 0.35))
    repr_ = ModelRepresentation(node)
    mark_encoding_layers(repr_)
    return IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=16, max_neurons=16,
    ).map(repr_)


def _flow(hybrid):
    return SpikingHybridCoreFlow(
        (8,), hybrid, simulation_length=T, spiking_mode="lif",
        cycle_accurate_lif_forward=True,
    )


def _fused(ir):
    return build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 16, "max_neurons": 16, "count": 16}],
    )


def _scheduled(ir, count: int):
    """A grid too small for one program, so the segment must be cut into passes."""
    strategy = MappingStrategy.resolve(ChipCapabilities(allow_scheduling=True))
    return build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 16, "max_neurons": 16, "count": count}],
        strategy=strategy,
    )


def _passes(hybrid):
    return sum(
        1 for s in hybrid.stages
        if getattr(s, "kind", None) == "neural"
        and getattr(s, "schedule_pass_index", None) is not None
    )


class TestTheRasterAgreesWithTheCounts:
    """The carry is recorded from the same fires the counts accumulate, so the
    raster summed over its window must BE the counts. A carry that failed this
    would hand the next pass spikes the producer never emitted."""

    def test_the_published_raster_sums_to_the_segment_counts(self):
        from mimarsinan.models.spiking.hybrid.executors.packed_cycle import (
            run_neural_segment_packed,
        )
        from mimarsinan.models.spiking.cycle_policy import cycle_neuron_policy

        hybrid = _fused(_deep_lif_ir())
        flow = _flow(hybrid)
        stage = next(s for s in hybrid.stages if getattr(s, "kind", None) == "neural")
        device = torch.device("cpu")
        seg = flow._get_segment_tensors(stage, device)
        seg.setdefault("latency", None)
        if seg["latency"] is None:
            from mimarsinan.mapping.latency.chip import ChipLatency

            seg["latency"] = int(ChipLatency(stage.hard_core_mapping).calculate())
        policy = cycle_neuron_policy("lif", "cascaded", "Default")
        train = (torch.rand(T, 2, len(stage.input_map) and
                            max(s.offset + s.size for s in stage.input_map)) > 0.5
                 ).float()
        sink: list = []
        with torch.no_grad():
            counts = run_neural_segment_packed(
                flow, train, seg=seg, stage=stage, T=T, batch_size=2,
                device=device, policy=policy, output_train=sink,
            )
        assert sink, "a requested carry must be produced"
        assert sink[0].shape == (T, 2, counts.shape[1])
        assert torch.equal(sink[0].sum(dim=0), counts)


class TestPassingIsSemanticallyInvisible:
    def test_a_scheduled_run_reproduces_the_fused_run_exactly(self):
        ir = _deep_lif_ir()
        fused, scheduled = _fused(ir), _scheduled(ir, count=2)
        assert _passes(scheduled) >= 2, (
            f"the grid must force a cut for this test to mean anything; got "
            f"{_passes(scheduled)} pass(es)")
        x = torch.rand(3, 8)
        with torch.no_grad():
            assert torch.equal(_flow(fused)(x), _flow(scheduled)(x))

    def test_more_passes_still_reproduce_the_fused_run(self):
        ir = _deep_lif_ir(seed=7)
        fused, scheduled = _fused(ir), _scheduled(ir, count=1)
        x = torch.rand(3, 8)
        with torch.no_grad():
            assert torch.equal(_flow(fused)(x), _flow(scheduled)(x))

    def test_the_cut_really_happened(self):
        """Guard the guard: if the grid stopped forcing a cut, the equivalence
        tests above would pass vacuously."""
        from mimarsinan.mapping.support.schedule.pass_cut import (
            carried_outputs_by_stage,
        )

        scheduled = _scheduled(_deep_lif_ir(), count=2)
        carried = carried_outputs_by_stage(scheduled.stages)
        assert carried, "no wire crosses a pass boundary; the test proves nothing"

    def test_the_carry_is_load_bearing(self, monkeypatch):
        """Collapsing the pass boundary to counts CHANGES the result — which is
        exactly why streamed scheduling was locked off. Without this, the
        equivalence tests above could pass with the carry disabled."""
        from mimarsinan.models.spiking.hybrid.stage_io import HybridStageIOMixin

        ir = _deep_lif_ir(seed=3)
        fused, scheduled = _fused(ir), _scheduled(ir, count=2)
        x = torch.rand(8, 8)
        with torch.no_grad():
            reference = _flow(fused)(x)
        monkeypatch.setattr(
            HybridStageIOMixin, "_publish_carried_trains",
            staticmethod(lambda *a, **k: None),
        )
        with torch.no_grad():
            collapsed = _flow(scheduled)(x)
        assert not torch.equal(reference, collapsed), (
            "the uniform re-encode reproduced the raster on this vehicle, so it "
            "cannot witness the carry; pick a vehicle with non-uniform rhythm")
