"""§7 row 6 in the PRODUCTION gate: window counts AND rasters, atol=0.

``assert_streamed_nf_scm_exact_or_raise`` arms for the per-event point through
``is_streamed_lif`` with no edit; under the point it additionally compares the
per-CYCLE emission rasters, because equal window counts with a different
rhythm are a different computation at the next hop.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mimarsinan.config_schema.defaults import (
    get_default_deployment_parameters,
    get_default_platform_constraints,
)
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.core.nf_scm_parity import (
    NfScmParityError,
    assert_streamed_nf_scm_exact_or_raise,
)
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

_T = 6
_LAW_KEYS = {
    "firing_granularity": "per_event",
    "firing_mode": "Novena",
    "thresholding_mode": "<=",
}


def _perceptron(out_ch, in_features, theta, *, encoding=False):
    p = Perceptron(out_ch, in_features, normalization=nn.Identity())
    p.is_encoding_layer = encoding
    p.set_activation_scale(theta)
    lif = LIFActivation(T=_T, activation_scale=p.activation_scale,
                        firing_mode="Novena", thresholding_mode="<=")
    lif.use_cycle_accurate_trains = True
    p.base_activation = lif
    p.activation = lif
    return p


class _StreamedNFModel(nn.Module):
    def __init__(self, repr_, perceptrons, soma_law):
        super().__init__()
        self.repr_ = repr_
        self.soma_law = soma_law
        self._perceptrons = nn.ModuleList(perceptrons)

    def get_perceptrons(self):
        return list(self._perceptrons)

    def forward(self, x):
        return SegmentForwardDriver(
            self.repr_, _T, LifSegmentPolicy(soma_law=self.soma_law))(x)


def _pipeline(**overrides):
    cfg = get_default_deployment_parameters()
    cfg.update(get_default_platform_constraints())
    cfg.update({
        "spiking_family": "lif", "spiking_variant": "streamed",
        "simulation_steps": _T, "input_shape": (8,), "device": "cpu",
        **overrides,
    })
    return SimpleNamespace(config=cfg)


def _build(soma_law):
    torch.manual_seed(0)
    p0 = _perceptron(8, 8, 1.0, encoding=True)
    p1 = _perceptron(6, 8, 0.5)
    p1.per_input_scales = torch.full((8,), 1.0)
    p2 = _perceptron(4, 6, 1.0)
    p2.per_input_scales = torch.full((6,), 0.5)
    # Amplify onto the integer grid: a per-event law is only distinguishable
    # from a per-cycle one when a single cycle can cross theta more than once.
    with torch.no_grad():
        # A dense, mostly-excitatory encoder: sparse wires cannot multi-spike.
        p0.layer.weight.copy_(p0.layer.weight.abs() * 0.5)
        if p0.layer.bias is not None:
            p0.layer.bias.fill_(0.15)
        for p, gain in ((p1, 3.0), (p2, 6.0)):
            p.layer.weight.copy_((p.layer.weight * gain).round() / 2.0)
            if p.layer.bias is not None:
                p.layer.bias.zero_()
    repr_ = ModelRepresentation(
        PerceptronMapper(PerceptronMapper(PerceptronMapper(
            InputMapper((8,)), p0), p1), p2))
    mark_encoding_layers(repr_)
    repr_.assign_perceptron_indices()
    ir_graph = IRMapping(
        q_max=127.0, firing_mode="Novena", max_axons=32, max_neurons=32,
    ).map(repr_)
    return _StreamedNFModel(repr_, [p0, p1, p2], soma_law).eval(), ir_graph


def _law(pipeline):
    from mimarsinan.chip_simulation.deployment_contract import (
        SpikingDeploymentContract,
    )

    return SpikingDeploymentContract.from_pipeline_config(pipeline.config).soma_law()


def _samples(n: int = 3):
    torch.manual_seed(1)
    return torch.rand(n, 8)


def test_the_gate_holds_at_atol_zero_under_the_per_event_point():
    pipeline = _pipeline(**_LAW_KEYS)
    law = _law(pipeline)
    assert law.is_per_event
    model, ir_graph = _build(law)
    assert_streamed_nf_scm_exact_or_raise(pipeline, model, ir_graph, _samples())


def test_the_witness_is_non_degenerate_under_the_point():
    """At least one neuron must emit >= 2 spikes in ONE cycle, or the gate is
    blind to everything the per-event law adds."""
    from mimarsinan.pipelining.core.nf_scm_parity import (
        _capture_nf_streamed_rasters,
    )

    pipeline = _pipeline(**_LAW_KEYS)
    model, _ = _build(_law(pipeline))
    rasters = _capture_nf_streamed_rasters(model, _samples())
    assert rasters, "no per-cycle trains captured"
    peak = max(float(v.max()) for v in rasters.values())
    assert peak >= 2.0, f"degenerate witness: peak per-cycle emission {peak}"


def test_the_raster_arm_has_teeth_on_a_mutated_theta():
    """Teeth: mutate the deployed threshold and the gate must go RED."""
    pipeline = _pipeline(**_LAW_KEYS)
    model, ir_graph = _build(_law(pipeline))
    for core in ir_graph.get_neural_cores():
        core.threshold = float(core.threshold) * 2.0
    with pytest.raises(NfScmParityError):
        assert_streamed_nf_scm_exact_or_raise(
            pipeline, model, ir_graph, _samples())


def test_the_default_point_gate_is_unchanged_and_skips_the_raster_arm():
    pipeline = _pipeline(firing_mode="Novena", thresholding_mode="<=")
    law = _law(pipeline)
    assert law.is_default_point
    model, ir_graph = _build(law)
    assert_streamed_nf_scm_exact_or_raise(pipeline, model, ir_graph, _samples())
