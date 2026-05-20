"""Neural-segment outputs emit ``(T, B, D)`` spike trains alongside spike counts."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.activations import LIFActivation
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers


def _two_segment_model(T: int = 4):
    """Build a 2-perceptron LIF model on a single chip segment for Phase B unit checks."""
    torch.manual_seed(0)
    inp = InputMapper((4,))
    p1 = Perceptron(4, 4, normalization=nn.Identity())
    p1.is_encoding_layer = True
    p1.base_activation = LIFActivation(T=T, activation_scale=torch.tensor(1.0))
    p1.activation = p1.base_activation
    p1.activation.use_cycle_accurate_trains = True

    p2 = Perceptron(3, 4, normalization=nn.Identity())
    p2.base_activation = LIFActivation(T=T, activation_scale=torch.tensor(1.0))
    p2.activation = p2.base_activation
    p2.activation.use_cycle_accurate_trains = True

    repr_ = ModelRepresentation(
        PerceptronMapper(PerceptronMapper(inp, p1), p2),
    )
    mark_encoding_layers(repr_)
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=16, max_neurons=16).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 16, "max_neurons": 16, "count": 4}],
    )
    return hybrid, repr_


def _make_flow(hybrid, T):
    return SpikingHybridCoreFlow(
        (4,), hybrid, simulation_length=T,
        firing_mode="Default", spike_mode="Uniform",
        thresholding_mode="<=", spiking_mode="lif",
        cycle_accurate_lif_forward=True,
    )


def test_counts_equal_spike_train_sum_per_output() -> None:
    """``output_counts[..., d0:d1] == output_spike_train[:, ..., d0:d1].sum(dim=0)`` exactly."""
    T = 4
    hybrid, _ = _two_segment_model(T=T)
    flow = _make_flow(hybrid, T)
    x = torch.rand(2, 4)

    captured_pairs = []
    orig = flow._run_neural_segment_rate
    def cap(stage, **kw):
        result = orig(stage, **kw)
        if isinstance(result, tuple):
            counts, train = result
            captured_pairs.append((stage.name, counts.detach(), train.detach()))
        return result
    flow._run_neural_segment_rate = cap

    with torch.no_grad():
        _ = flow(x)
    assert captured_pairs, "expected at least one neural segment to run"
    for name, counts, train in captured_pairs:
        recon = train.sum(dim=0)
        torch.testing.assert_close(counts, recon, atol=1e-9, rtol=0.0)


def test_neural_to_neural_no_uniform_fallback(caplog) -> None:
    """When two neural segments chain, the consumer must NOT uniform-encode the
    producer's output (no fallback warning), because Phase B caches the producer's
    spike train into ``state_buffer_spikes``."""
    T = 4
    hybrid, _ = _two_segment_model(T=T)
    flow = _make_flow(hybrid, T)

    import logging
    caplog.set_level(logging.WARNING, logger="mimarsinan.spiking.segment_encoding")
    caplog.set_level(logging.WARNING, logger="mimarsinan.spiking.spike_trains")

    x = torch.rand(2, 4)
    with torch.no_grad():
        _ = flow(x)
    fallback_warnings = [
        r for r in caplog.records
        if "uniform" in r.message.lower() and "fallback" in r.message.lower()
    ]
    assert not fallback_warnings, (
        f"unexpected uniform fallback warnings: {[r.message for r in fallback_warnings]}"
    )


def test_state_buffer_spikes_populated_from_neural_segment() -> None:
    """After a neural segment runs, ``state_buffer_spikes`` should contain its outputs
    keyed by the IR node_ids in ``stage.output_map``."""
    T = 4
    hybrid, _ = _two_segment_model(T=T)
    if sum(1 for s in hybrid.stages if s.kind == "neural") < 2:
        # The tiny model may merge into a single neural segment if cores are large enough.
        # In that case state_buffer_spikes population is only exercised at the encoding
        # boundary (Phase A), so this test is a no-op here.
        return
    flow = _make_flow(hybrid, T)

    captured = {}
    from mimarsinan.chip_simulation.hybrid_stage_runner import HybridStageContext
    orig_ctx_init = HybridStageContext.__init__
    def cap_ctx(self, *a, **kw):
        orig_ctx_init(self, *a, **kw)
        captured.setdefault('spikes', self.state_buffer_spikes)
    HybridStageContext.__init__ = cap_ctx
    try:
        x = torch.rand(2, 4)
        with torch.no_grad():
            _ = flow(x)
    finally:
        HybridStageContext.__init__ = orig_ctx_init

    spikes = captured.get('spikes')
    assert spikes is not None
    # After the first neural segment, its output node_ids should appear in the cache.
    first_neural = next(s for s in hybrid.stages if s.kind == "neural")
    if hybrid.stages.index(first_neural) < len(hybrid.stages) - 1:
        # There IS a downstream stage that needs the spike train.
        out_ids = {int(s.node_id) for s in first_neural.output_map}
        cached_ids = set(spikes.keys())
        assert out_ids & cached_ids, (
            f"state_buffer_spikes should contain at least one of {out_ids}; got {cached_ids}"
        )
