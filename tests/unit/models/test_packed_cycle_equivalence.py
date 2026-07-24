"""[cert-plan W1] the packed (stage-flat) cycle executor is bit-equal to the
per-core reference loop: same per-cycle LIF physics, different tensor layout.
Integer-grid weights make charge sums layout-exact, so any count difference
is a semantics defect, not float noise."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

T = 8


def _int_grid_(module: nn.Linear, scale: int = 3) -> None:
    with torch.no_grad():
        module.weight.copy_((module.weight * scale).round())
        if module.bias is not None:
            module.bias.copy_((module.bias * scale).round())


def _chain(theta: float = 3.0, hardware_bias: bool = False, per_hop: bool = False):
    torch.manual_seed(11)
    inp = InputMapper((8,))
    p1 = Perceptron(6, 8, normalization=nn.Identity())
    p1.is_encoding_layer = True
    p1.set_activation_scale(1.0)
    lif1 = LIFActivation(T=T, activation_scale=p1.activation_scale)
    lif1.use_cycle_accurate_trains = True
    p1.base_activation = lif1
    p1.activation = lif1
    p2 = Perceptron(5, 6, normalization=nn.Identity())
    p2.set_activation_scale(theta)
    p2.per_input_scales = torch.full((6,), 1.0)
    lif2 = LIFActivation(T=T, activation_scale=p2.activation_scale)
    lif2.use_cycle_accurate_trains = True
    p2.base_activation = lif2
    p2.activation = lif2
    p3 = Perceptron(4, 5, normalization=nn.Identity())
    p3.set_activation_scale(theta)
    p3.per_input_scales = torch.full((5,), float(theta))
    lif3 = LIFActivation(T=T, activation_scale=p3.activation_scale)
    lif3.use_cycle_accurate_trains = True
    p3.base_activation = lif3
    p3.activation = lif3
    for p in (p1, p2, p3):
        _int_grid_(p.layer)
    repr_ = ModelRepresentation(
        PerceptronMapper(PerceptronMapper(PerceptronMapper(inp, p1), p2), p3))
    mark_encoding_layers(repr_)
    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=32, max_neurons=32,
        hardware_bias=hardware_bias,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 8}],
        per_hop_neural_segments=per_hop,
    )
    return hybrid


def _flow(hybrid, *, thresholding: str, firing: str, packed: bool):
    flow = SpikingHybridCoreFlow(
        (8,), hybrid, T,
        firing_mode=firing, spike_mode="Uniform", thresholding_mode=thresholding,
        spiking_mode="lif", cycle_accurate_lif_forward=True,
        lif_execution_synchronized=False,
    )
    flow.use_packed_cycle_executor = packed
    return flow


def _counts(flow, x):
    captured = []
    flow.stage_count_recorder = (
        lambda stage, counts: captured.append(counts.detach().clone())
    )
    with torch.no_grad():
        out = flow(x)
    flow.stage_count_recorder = None
    return out, captured


def _assert_bit_equal(hybrid, *, thresholding: str, firing: str):
    torch.manual_seed(3)
    x = torch.rand(4, 8) * 0.9
    ref_out, ref_counts = _counts(
        _flow(hybrid, thresholding=thresholding, firing=firing, packed=False), x)
    got_out, got_counts = _counts(
        _flow(hybrid, thresholding=thresholding, firing=firing, packed=True), x)
    assert len(ref_counts) == len(got_counts) and len(ref_counts) >= 1
    for i, (r, g) in enumerate(zip(ref_counts, got_counts)):
        assert torch.equal(r, g), (
            f"stage {i}: packed counts differ from reference "
            f"(max|d|={float((r - g).abs().max())})"
        )
    assert torch.equal(ref_out, got_out)


def test_packed_equals_reference_multi_latency_segment():
    """One segment, three cores at latencies 0/1/2 (no per-hop retiming)."""
    _assert_bit_equal(_chain(per_hop=False), thresholding="<", firing="Default")


def test_packed_equals_reference_per_hop_retimed():
    _assert_bit_equal(_chain(per_hop=True), thresholding="<", firing="Default")


def test_packed_equals_reference_inclusive_threshold():
    _assert_bit_equal(_chain(per_hop=False), thresholding="<=", firing="Default")


def test_packed_equals_reference_novena_firing():
    _assert_bit_equal(_chain(per_hop=False), thresholding="<", firing="Novena")


def test_packed_equals_reference_hardware_bias():
    _assert_bit_equal(
        _chain(hardware_bias=True, per_hop=False),
        thresholding="<", firing="Default",
    )


def test_packed_respects_membrane_init_precharge():
    hybrid = _chain(per_hop=False)
    torch.manual_seed(3)
    x = torch.rand(4, 8) * 0.9
    outs = []
    for packed in (False, True):
        flow = _flow(hybrid, thresholding="<", firing="Default", packed=packed)
        flow.lif_membrane_init = 0.5
        with torch.no_grad():
            outs.append(flow(x))
    assert torch.equal(outs[0], outs[1])


def test_packed_membrane_readout_matches_reference():
    """[C2] the membrane-decode read: packed per-core membrane views feed the
    same stash; forward outputs (which consume the corrections) bit-match."""
    hybrid = _chain(per_hop=False)
    torch.manual_seed(3)
    x = torch.rand(4, 8) * 0.9
    outs = []
    for packed in (False, True):
        flow = SpikingHybridCoreFlow(
            (8,), hybrid, T,
            firing_mode="Default", spike_mode="Uniform", thresholding_mode="<",
            spiking_mode="lif", cycle_accurate_lif_forward=True,
            lif_execution_synchronized=False,
            membrane_readout=True,
        )
        flow.use_packed_cycle_executor = packed
        with torch.no_grad():
            outs.append(flow(x))
    assert torch.equal(outs[0], outs[1])
    # Non-vacuity: the membrane decode must actually engage on this fixture —
    # a readout-off run must differ, else both paths merely skipped the stash.
    flow_off = SpikingHybridCoreFlow(
        (8,), hybrid, T,
        firing_mode="Default", spike_mode="Uniform", thresholding_mode="<",
        spiking_mode="lif", cycle_accurate_lif_forward=True,
        lif_execution_synchronized=False,
        membrane_readout=False,
    )
    with torch.no_grad():
        out_off = flow_off(x)
    assert not torch.equal(outs[0], out_off), (
        "membrane readout did not engage; the equivalence cell is vacuous"
    )


def test_single_spike_and_recording_fall_back_to_reference():
    """Packed eligibility: single-spike TTFS and recording paths keep the
    per-core reference loop (byte-stable backend records)."""
    import inspect

    from mimarsinan.models.spiking.hybrid import lif_step

    src = inspect.getsource(lif_step.HybridLifStepMixin._run_neural_segment_rate)
    assert "run_neural_segment_packed" in src
    guard = src[src.index("run_neural_segment_packed") - 400:
                src.index("run_neural_segment_packed")]
    assert "single_spike" in guard and "recording" in guard
