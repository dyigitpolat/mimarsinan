"""Boundary spike-train encoding: classification + cycle-accurate emission."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.hybrid_hardcore_mapping import (
    HybridStage,
    SegmentIOSlice,
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.activations import LIFActivation, run_cycle_accurate
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.spiking.segment_encoding import (
    BoundaryKind,
    BoundaryLifCache,
    SegmentEncodingConfig,
    build_segment_input_spike_train,
    classify_encoding_boundary,
    emit_compute_spike_train,
)
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers


def _config(T: int = 4, *, cycle_accurate: bool = True) -> SegmentEncodingConfig:
    return SegmentEncodingConfig(
        simulation_length=T,
        spiking_mode="lif",
        cycle_accurate=cycle_accurate,
        spike_mode="Uniform",
        thresholding_mode="<=",
        firing_mode="Default",
        compute_dtype=torch.float64,
    )


def _tiny_lif_model(T: int = 4):
    inp = InputMapper((8,))
    p1 = Perceptron(6, 8, normalization=nn.Identity())
    p1.is_encoding_layer = True
    lif1 = LIFActivation(T=T, activation_scale=torch.tensor(1.0))
    lif1.use_cycle_accurate_trains = True
    p1.base_activation = lif1
    p1.activation = lif1
    p2 = Perceptron(3, 6, normalization=nn.Identity())
    lif2 = LIFActivation(T=T, activation_scale=torch.tensor(1.0))
    lif2.use_cycle_accurate_trains = True
    p2.base_activation = lif2
    p2.activation = lif2
    repr_ = ModelRepresentation(PerceptronMapper(PerceptronMapper(inp, p1), p2))
    mark_encoding_layers(repr_)
    ir = IRMapping(
        q_max=127.0,
        firing_mode="Default",
        max_axons=32,
        max_neurons=32,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 4}],
    )
    return hybrid, p1


def test_segment_encoding_config_use_cycle_accurate_trains() -> None:
    assert _config().use_cycle_accurate_trains is True
    assert _config(cycle_accurate=False).use_cycle_accurate_trains is False
    rate_cfg = SegmentEncodingConfig(
        simulation_length=4,
        spiking_mode="rate",
        cycle_accurate=True,
    )
    assert rate_cfg.use_cycle_accurate_trains is False


def test_boundary_lif_cache_reuses_instance() -> None:
    cache = BoundaryLifCache()
    scale = torch.tensor(1.5)
    a = cache.get(T=4, activation_scale=scale, thresholding_mode="<=", firing_mode="Default")
    b = cache.get(T=4, activation_scale=scale, thresholding_mode="<=", firing_mode="Default")
    assert a is b
    other_T = cache.get(T=5, activation_scale=scale, thresholding_mode="<=", firing_mode="Default")
    assert other_T is not a
    other_scale = cache.get(T=4, activation_scale=torch.tensor(2.0), thresholding_mode="<=", firing_mode="Default")
    assert other_scale is not a


def test_classify_encoding_boundary_perceptron_lif_op() -> None:
    hybrid, _ = _tiny_lif_model()
    cfg = _config()
    compute_ops = [s.compute_op for s in hybrid.stages if s.kind == "compute"]
    assert compute_ops, "expected at least one compute op in hybrid mapping"
    # The encoding perceptron is the first compute op.
    kind = classify_encoding_boundary(compute_ops[0], hybrid, cfg)
    assert kind == BoundaryKind.ENCODING_LIF_PERCEPTRON


def test_classify_legacy_rate_when_cycle_accurate_disabled() -> None:
    hybrid, _ = _tiny_lif_model()
    cfg = _config(cycle_accurate=False)
    op = next(s.compute_op for s in hybrid.stages if s.kind == "compute")
    kind = classify_encoding_boundary(op, hybrid, cfg)
    assert kind == BoundaryKind.LEGACY_RATE


def test_emit_compute_spike_train_matches_nf_for_encoding_perceptron() -> None:
    """Boundary emission must replay NF's per-cycle pattern, not feed constant ``linear(x)``."""
    torch.manual_seed(0)
    T = 4
    hybrid, p1 = _tiny_lif_model(T=T)
    cfg = _config(T=T)
    cache = BoundaryLifCache()
    op = next(s.compute_op for s in hybrid.stages if s.kind == "compute")

    x = torch.rand(2, 8).to(cfg.compute_dtype)

    # NF reference: run the wrapped Perceptron cycle-accurately on uniform-encoded raw x.
    nf_train = run_cycle_accurate(p1, x.to(torch.float32), T)
    # run_cycle_accurate returns the *mean*; reconstruct the per-cycle spike pattern by
    # invoking the same internal loop (single-step LIF, uniform-encoded inputs).
    from spikingjelly.activation_based import functional
    from mimarsinan.spiking.spike_trains import uniform_spike_train
    spike_train_in = uniform_spike_train(x.to(torch.float32), T)
    lif = p1.activation
    lif.set_cycle_accurate(True)
    functional.reset_net(lif)
    nf_per_cycle = torch.stack([p1(spike_train_in[t]) for t in range(T)], dim=0)
    lif.set_cycle_accurate(False)

    emitted = emit_compute_spike_train(
        op=op,
        state_buffer={-2: x},
        state_buffer_spikes={},
        config=cfg,
        hybrid_mapping=hybrid,
        lif_cache=cache,
    )
    assert emitted is not None
    assert emitted.shape == nf_per_cycle.shape
    torch.testing.assert_close(
        emitted.to(torch.float32),
        nf_per_cycle.to(torch.float32),
        atol=1e-6, rtol=0.0,
    )


def test_emit_compute_spike_train_returns_none_in_legacy_rate_mode() -> None:
    hybrid, _ = _tiny_lif_model()
    cfg = _config(cycle_accurate=False)
    cache = BoundaryLifCache()
    op = next(s.compute_op for s in hybrid.stages if s.kind == "compute")
    x = torch.rand(2, 8).to(cfg.compute_dtype)
    out = emit_compute_spike_train(
        op=op,
        state_buffer={-2: x},
        state_buffer_spikes={},
        config=cfg,
        hybrid_mapping=hybrid,
        lif_cache=cache,
    )
    assert out is None


def test_build_segment_input_spike_train_consumes_cache_verbatim() -> None:
    hybrid, _ = _tiny_lif_model()
    cfg = _config()
    cache = BoundaryLifCache()
    stage = next(s for s in hybrid.stages if s.kind == "neural")
    in_size = sum(s.size for s in stage.input_map)
    fake_train = torch.rand(cfg.simulation_length, 1, in_size, dtype=cfg.compute_dtype)
    state_buffer_spikes = {}
    offset = 0
    for s in stage.input_map:
        state_buffer_spikes[int(s.node_id)] = (
            fake_train[:, :, offset : offset + s.size].contiguous()
        )
        offset += s.size

    rates = torch.zeros(1, in_size, dtype=cfg.compute_dtype)
    out = build_segment_input_spike_train(
        stage,
        rates,
        state_buffer_spikes,
        config=cfg,
        hybrid_mapping=hybrid,
        lif_cache=cache,
        T=cfg.simulation_length,
        batch_size=1,
        device=rates.device,
    )
    torch.testing.assert_close(out, fake_train, atol=0.0, rtol=0.0)


def test_build_segment_input_raw_input_uniform_encoded() -> None:
    """A stage that consumes only raw input uniform-encodes the rate (no cache available)."""
    hybrid, _ = _tiny_lif_model()
    cfg = _config()
    cache = BoundaryLifCache()
    # Find the first compute-stage whose only input is raw (-2).
    raw_stages = [
        s for s in hybrid.stages
        if s.kind == "neural"
        and len(s.input_map) == 1
        and int(s.input_map[0].node_id) == -2
    ]
    if not raw_stages:
        pytest.skip("no raw-input neural stage in the tiny model (encoding wraps it)")
    stage = raw_stages[0]
    in_size = sum(s.size for s in stage.input_map)
    rates = torch.tensor([[0.5] * in_size], dtype=cfg.compute_dtype)
    out = build_segment_input_spike_train(
        stage,
        rates,
        {},
        config=cfg,
        hybrid_mapping=hybrid,
        lif_cache=cache,
        T=cfg.simulation_length,
        batch_size=1,
        device=rates.device,
    )
    # Uniform encoding of 0.5 over T=4 fires twice per neuron.
    assert out.shape == (cfg.simulation_length, 1, in_size)
    assert int(out.sum().item()) == 2 * in_size


def test_build_segment_input_partial_cache_raises() -> None:
    """When one slice has a cached train but another doesn't (and it's not raw input), error."""
    hybrid, _ = _tiny_lif_model()
    cfg = _config()
    cache = BoundaryLifCache()
    stage = next(s for s in hybrid.stages if s.kind == "neural")
    if len(stage.input_map) < 2:
        # Construct a synthetic stage with multiple slices.
        s1 = SegmentIOSlice(node_id=42, offset=0, size=3)
        s2 = SegmentIOSlice(node_id=43, offset=3, size=2)
        synth = HybridStage(
            kind="neural",
            name="synth",
            hard_core_mapping=stage.hard_core_mapping,
            compute_op=None,
            input_map=[s1, s2],
            output_map=list(stage.output_map),
        )
        stage = synth
    else:
        s1, s2 = stage.input_map[0], stage.input_map[1]

    in_size = sum(s.size for s in stage.input_map)
    rates = torch.zeros(1, in_size, dtype=cfg.compute_dtype)
    state_buffer_spikes = {
        int(s1.node_id): torch.rand(cfg.simulation_length, 1, s1.size, dtype=cfg.compute_dtype),
    }
    # Missing s2 — and s2 is not raw input — should raise.
    if int(s2.node_id) == -2:
        pytest.skip("test requires a non-raw missing slice")
    with pytest.raises(ValueError, match="missing spike train"):
        build_segment_input_spike_train(
            stage,
            rates,
            state_buffer_spikes,
            config=cfg,
            hybrid_mapping=hybrid,
            lif_cache=cache,
            T=cfg.simulation_length,
            batch_size=1,
            device=rates.device,
        )
