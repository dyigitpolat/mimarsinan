"""Both torch executors run ONE kernel under the per-event point.

The packed (stage-flat) executor reaches the fold through ``advance_events``
at the grouped per-axon tensor; the per-core reference loop reaches it through
``step`` with no call-site change. They must be bit-equal — the production
metric takes the packed path and the NF↔SCM gate takes the reference path, so
a difference between them is a defect that no other gate can see.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW, SomaLaw
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.spiking.cycle_policy import (
    LIFCyclePolicy,
    cycle_neuron_policy,
)
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.spiking.serial import (
    CycleAtomicRefusalError,
    SerialLIFCyclePolicy,
    SerialResetLawError,
)
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

T = 8

PER_EVENT_LAW = SomaLaw(
    firing_mode="Novena", thresholding_mode="<=",
    firing_granularity="per_event",
    membrane_arithmetic="saturating_unsigned", membrane_bits=8,
)
PER_EVENT_UNBOUNDED = SomaLaw(
    firing_mode="Novena", thresholding_mode="<=",
    firing_granularity="per_event", membrane_arithmetic="unbounded",
    membrane_bits=0,
)
# The law the row-pair lemma denies: only an EXPLICIT declaration reaches it
# (an axes-only per_event document derives 'Novena').
PER_EVENT_SUBTRACTIVE = SomaLaw(
    firing_mode="Default", thresholding_mode="<=",
    firing_granularity="per_event", membrane_arithmetic="unbounded",
    membrane_bits=0,
)


def _int_grid_(module: nn.Linear, scale: int = 3) -> None:
    with torch.no_grad():
        module.weight.copy_((module.weight * scale).round())
        if module.bias is not None:
            module.bias.copy_((module.bias * scale).round())


def _identity_relay(width: int, in_scale: float) -> Perceptron:
    """A theta=1, W=I relay: k arriving events become k spikes, exactly.

    Fold-INVARIANT by construction, so a mapping may insert it freely under
    the per-event law (depth balancing does exactly this).
    """
    relay = Perceptron(width, width, normalization=nn.Identity())
    relay.set_activation_scale(in_scale)
    relay.per_input_scales = torch.full((width,), float(in_scale))
    with torch.no_grad():
        relay.layer.weight.copy_(torch.eye(width))
        if relay.layer.bias is not None:
            relay.layer.bias.zero_()
    return relay


def build_chain(theta: float = 3.0, *, per_hop: bool = False,
                retimed_levels: bool = False, relay: bool = False,
                hardware_bias: bool = False, n_cores: int = 8,
                max_neurons: int = 32):
    """A three-hop integer-grid MLP: the shared fixture for every ODIN gate."""
    torch.manual_seed(11)
    inp = InputMapper((8,))
    p1 = Perceptron(6, 8, normalization=nn.Identity())
    p1.is_encoding_layer = True
    p1.set_activation_scale(1.0)
    p2 = Perceptron(5, 6, normalization=nn.Identity())
    p2.set_activation_scale(theta)
    p2.per_input_scales = torch.full((6,), 1.0)
    p3 = Perceptron(4, 5, normalization=nn.Identity())
    p3.set_activation_scale(theta)
    p3.per_input_scales = torch.full((5,), float(theta))
    trainable = [p1, p2, p3]
    for p in trainable:
        _int_grid_(p.layer)
    hops = [p1, p2, _identity_relay(5, theta), p3] if relay else trainable
    for p in hops:
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    mapper = inp
    for p in hops:
        mapper = PerceptronMapper(mapper, p)
    repr_ = ModelRepresentation(mapper)
    mark_encoding_layers(repr_)
    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=32,
        max_neurons=max_neurons,
        hardware_bias=hardware_bias,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": max_neurons,
                       "count": n_cores}],
        per_hop_neural_segments=per_hop,
        retimed_level_stages=retimed_levels,
    )
    return repr_, hybrid


def build_flow(hybrid, *, law: SomaLaw, packed: bool, membrane_init: float = 0.0):
    flow = SpikingHybridCoreFlow(
        (8,), hybrid, T,
        firing_mode=law.firing_mode, spike_mode="Uniform",
        thresholding_mode=law.thresholding_mode,
        spiking_mode="lif", cycle_accurate_lif_forward=True,
        lif_execution_synchronized=False,
        lif_membrane_init=membrane_init,
        soma_law=law,
    )
    flow.use_packed_cycle_executor = packed
    return flow


def run_counts(flow, x):
    captured: list = []
    flow.stage_count_recorder = (
        lambda stage, counts: captured.append(counts.detach().clone())
    )
    with torch.no_grad():
        out = flow(x)
    flow.stage_count_recorder = None
    return out, captured


@pytest.mark.parametrize(
    "law", [DEFAULT_SOMA_LAW, PER_EVENT_UNBOUNDED, PER_EVENT_LAW],
    ids=["default_point", "per_event_unbounded", "per_event_sat8"],
)
def test_packed_equals_reference_under_the_point(law):
    _, hybrid = build_chain()
    torch.manual_seed(3)
    x = torch.rand(4, 8) * 0.9
    ref_out, ref_counts = run_counts(build_flow(hybrid, law=law, packed=False), x)
    got_out, got_counts = run_counts(build_flow(hybrid, law=law, packed=True), x)
    assert len(ref_counts) == len(got_counts) >= 1
    for i, (r, g) in enumerate(zip(ref_counts, got_counts)):
        assert torch.equal(r, g), f"stage {i}: packed differs from reference"
    assert torch.equal(ref_out, got_out)


def test_the_per_event_point_is_a_different_computation():
    """Non-vacuity: if the fold reduced to today's law the equivalence cell
    above would prove nothing."""
    _, hybrid = build_chain()
    torch.manual_seed(3)
    x = torch.rand(4, 8) * 0.9
    default_out, _ = run_counts(build_flow(hybrid, law=DEFAULT_SOMA_LAW, packed=True), x)
    serial_out, _ = run_counts(build_flow(hybrid, law=PER_EVENT_UNBOUNDED, packed=True), x)
    assert not torch.equal(default_out, serial_out)


def test_the_witness_actually_multi_spikes_in_a_single_cycle():
    """The NON-DEGENERATE witness: at least one neuron emits >= 2 spikes in
    ONE cycle, with distinct weights and no saturation. A fixture that only
    ever emitted 0/1 would let a cycle-atomic kernel pass every gate here."""
    _, hybrid = build_chain()
    torch.manual_seed(3)
    x = torch.rand(4, 8) * 0.9
    flow = build_flow(hybrid, law=PER_EVENT_UNBOUNDED, packed=True)
    raster: list = []
    stage = next(s for s in hybrid.stages if s.kind == "neural")
    with torch.no_grad():
        flow._run_neural_segment_rate(
            stage,
            input_spike_train=torch.zeros(T, 4, 8, dtype=torch.float64).bernoulli_(0.7),
            output_train=raster,
        )
    assert raster, "the carry raster is the per-cycle multiplicity evidence"
    peak = float(raster[0].max())
    assert peak >= 2.0, (
        f"witness is degenerate: peak per-cycle emission {peak} < 2, so the "
        f"per-event law is indistinguishable from the per-cycle law here"
    )
    # Distinct weights, and the membrane never hit a rail (unbounded law).
    seg = flow._get_segment_tensors(stage, torch.device("cpu"))
    weights = seg["core_params"][0]
    assert torch.unique(weights).numel() >= 3


def test_advance_refuses_a_pre_reduced_contribution():
    policy = cycle_neuron_policy("lif", "cascaded", "Novena",
                                 soma_law=PER_EVENT_UNBOUNDED)
    assert isinstance(policy, SerialLIFCyclePolicy)
    state = policy.make_state(2, 3, torch.device("cpu"), torch.float64)
    with pytest.raises(CycleAtomicRefusalError, match="advance"):
        policy.advance(state, torch.zeros(2, 3, dtype=torch.float64),
                       torch.tensor(1.0, dtype=torch.float64),
                       thresholding_mode="<=")


def test_the_subtractive_reset_is_refused_before_a_single_cycle_runs():
    """The batch-dependence witness, end to end: under a per_event point with
    the subtractive reset the packed pass produced counts that changed with who
    else was in the batch. Neither executor may run it — the dispatch itself
    refuses, so no number is ever reported from that law."""
    _, hybrid = build_chain()
    torch.manual_seed(3)
    x = torch.rand(4, 8) * 0.9
    for packed in (False, True):
        with pytest.raises(SerialResetLawError, match="Novena"):
            run_counts(build_flow(hybrid, law=PER_EVENT_SUBTRACTIVE,
                                  packed=packed), x)
    with pytest.raises(SerialResetLawError, match="Novena"):
        cycle_neuron_policy("lif", "cascaded", "Default",
                            soma_law=PER_EVENT_SUBTRACTIVE)


def test_default_point_dispatches_the_identical_policy_object_type():
    policy = cycle_neuron_policy("lif", "cascaded", "Default",
                                 soma_law=DEFAULT_SOMA_LAW)
    assert type(policy) is LIFCyclePolicy


def test_zero_membrane_init_is_admissible_and_both_paths_agree():
    """V0 = 0 is always representable; the equality above already covers the
    dynamics, so this pins that the guard does not refuse the legal case."""
    _, hybrid = build_chain()
    torch.manual_seed(3)
    x = torch.rand(4, 8) * 0.9
    outs = [
        run_counts(build_flow(hybrid, law=PER_EVENT_UNBOUNDED, packed=packed,
                              membrane_init=0.0), x)[0]
        for packed in (False, True)
    ]
    assert torch.equal(outs[0], outs[1])
