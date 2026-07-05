"""W1b wire contract: rate/LIF boundaries emit uniform trains of clamp(value/theta)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    apply_input_shifts_numpy,
    compute_input_state_with_shifts,
)
from mimarsinan.chip_simulation.recording._spike_encoding import uniform_rate_encode
from mimarsinan.mapping.ir import ComputeOp, IRSource
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
    build_hybrid_hard_core_mapping,
)
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.spiking.segment_boundary import (
    BoundaryConfig,
    boundary_normalization_scales,
    encode_compute_boundary,
    normalize_boundary_slices_numpy,
    normalize_boundary_slices_torch,
    normalize_boundary_value,
)
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.spiking.spike_trains import uniform_spike_train
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers


def _config(T: int) -> BoundaryConfig:
    return BoundaryConfig(
        simulation_length=T,
        spiking_mode="lif",
        cycle_accurate=True,
        spike_mode="Uniform",
        thresholding_mode="<=",
        firing_mode="Default",
        compute_dtype=torch.float64,
    )


def _tiny_lif_model(T: int, theta_enc: float, theta_hidden: float = 1.0):
    """input(8) -> encoding Perceptron(6, theta_enc) -> Perceptron(3, theta_hidden)."""
    torch.manual_seed(0)
    inp = InputMapper((8,))
    p1 = Perceptron(6, 8, normalization=nn.Identity())
    p1.is_encoding_layer = True
    p1.set_activation_scale(theta_enc)
    lif1 = LIFActivation(T=T, activation_scale=p1.activation_scale)
    lif1.use_cycle_accurate_trains = True
    p1.base_activation = lif1
    p1.activation = lif1
    p2 = Perceptron(3, 6, normalization=nn.Identity())
    p2.set_activation_scale(theta_hidden)
    # Production scale propagation folds the producer's out-scale into the
    # consumer's effective weights (per_input_scales); the wire contract
    # multiplies the same scale back.
    p2.per_input_scales = torch.full((6,), float(theta_enc))
    lif2 = LIFActivation(T=T, activation_scale=p2.activation_scale)
    lif2.use_cycle_accurate_trains = True
    p2.base_activation = lif2
    p2.activation = lif2
    repr_ = ModelRepresentation(PerceptronMapper(PerceptronMapper(inp, p1), p2))
    mark_encoding_layers(repr_)
    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=32, max_neurons=32,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 4}],
    )
    return repr_, hybrid, p1, p2


def _encoder_wire_rate(p1: Perceptron, x: torch.Tensor) -> torch.Tensor:
    """The encoder's rate-mode value, normalized to the wire domain."""
    from spikingjelly.activation_based import functional

    lif = p1.activation
    lif.set_cycle_accurate(False)
    functional.reset_net(lif.if_node)
    with torch.no_grad():
        value = p1(x)
    theta = lif.activation_scale
    return (value / theta.clamp(min=1e-12)).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# 1. SSOT transcode
# ---------------------------------------------------------------------------

def test_normalize_boundary_value_contract() -> None:
    v = torch.tensor([[0.5, 1.0, 2.0, 6.0]])
    out = normalize_boundary_value(v, 2.0)
    torch.testing.assert_close(
        out, torch.tensor([[0.25, 0.5, 1.0, 1.0]]), atol=0.0, rtol=0.0,
    )
    # Saturation pin: any value above theta encodes as a full-rate wire.
    assert float(normalize_boundary_value(torch.tensor([[7.3]]), 2.0)) == 1.0
    # theta = 1 is the clamp identity.
    torch.testing.assert_close(
        normalize_boundary_value(v, 1.0), v.clamp(0.0, 1.0), atol=0.0, rtol=0.0,
    )
    # Per-channel theta broadcasts channelwise.
    per_ch = normalize_boundary_value(
        torch.tensor([[1.0, 1.0, 2.0, 4.0]]), torch.tensor([1.0, 2.0, 4.0, 4.0]),
    )
    torch.testing.assert_close(
        per_ch, torch.tensor([[1.0, 0.5, 0.5, 1.0]]), atol=0.0, rtol=0.0,
    )


# ---------------------------------------------------------------------------
# 2. Plain-LIF subsumed boundary: the t0_04 / t0_05 regression pin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theta,T", [(5.239892, 32), (2.185, 4)])
def test_plain_lif_subsumed_boundary_record_is_wire_domain(theta, T) -> None:
    """The hybrid record's seg_input carries clamp(v/theta) and its counts equal
    round(rate*T); uniform re-encode of the recorded rates reproduces the counts
    exactly (the parity gate's seg_input invariant)."""
    _, hybrid, p1, _ = _tiny_lif_model(T, theta)
    x = 3.0 * torch.rand(1, 8)

    wire = _encoder_wire_rate(p1, x)
    # Non-vacuous: the value domain genuinely exceeds the [0,1] clamp here.
    with torch.no_grad():
        assert float((wire * theta).max()) > 1.0

    flow = SpikingHybridCoreFlow(
        (8,), hybrid, simulation_length=T,
        spiking_mode="lif", cycle_accurate_lif_forward=True,
    )
    with torch.no_grad():
        _, record = flow.forward_with_recording(x)

    neural_stages = [
        (i, s) for i, s in enumerate(hybrid.stages) if s.kind == "neural"
    ]
    assert neural_stages
    stage_index, stage = neural_stages[0]
    seg = record.segments[stage_index]

    # (a) the record is wire-domain
    expected = wire.to(torch.float32).numpy()
    for s in stage.input_map:
        np.testing.assert_allclose(
            seg.seg_input_rates[:, s.offset : s.offset + s.size],
            expected[:, : s.size],
            atol=1e-6, rtol=0.0,
        )

    # (b) counts obey the uniform wire contract
    np.testing.assert_array_equal(
        seg.seg_input_spike_count,
        np.rint(seg.seg_input_rates[0] * T).astype(np.int64),
    )

    # (c) the gate invariant: re-encoding the recorded rates reproduces the counts
    reencoded = uniform_rate_encode(seg.seg_input_rates, T)
    np.testing.assert_array_equal(
        reencoded[0].sum(axis=1).astype(np.int64), seg.seg_input_spike_count,
    )


# ---------------------------------------------------------------------------
# 3. Wrapper-boundary twin (the t0_01 blind spot)
# ---------------------------------------------------------------------------

class _WrapperShapedModule:
    """Conv2DPerceptronMapper-shaped: exposes a child perceptron with a scale."""

    def __init__(self, theta: float):
        self.perceptron = Perceptron(4, 4, normalization=nn.Identity())
        self.perceptron.set_activation_scale(theta)


def _synthetic_hybrid_with_compute_producer(module, out_size: int = 4):
    sources = np.array(
        [IRSource(node_id=-2, index=i) for i in range(out_size)], dtype=object,
    ).reshape(1, out_size)
    op = ComputeOp(
        id=7, name="patch_embed", op_type="Wrapper", input_sources=sources,
        params={"module": module},
    )
    compute_stage = HybridStage(
        kind="compute", name="patch_embed", compute_op=op,
        input_map=[], output_map=[SegmentIOSlice(node_id=7, offset=0, size=out_size)],
    )
    neural_stage = HybridStage(
        kind="neural", name="seg0", hard_core_mapping=None,
        input_map=[SegmentIOSlice(node_id=7, offset=0, size=out_size)],
        output_map=[],
    )
    return HybridHardCoreMapping(stages=[compute_stage, neural_stage]), op


def test_wrapper_boundary_scale_covers_t0_01_seam() -> None:
    theta = 3.3148
    mapping, op = _synthetic_hybrid_with_compute_producer(_WrapperShapedModule(theta))
    divisors = boundary_normalization_scales(mapping)
    assert set(divisors) == {int(op.id)}
    divisor = float(np.asarray(divisors[int(op.id)]))
    assert divisor == pytest.approx(theta)

    seg_input = np.array([[0.5, 1.0, 2.0, 6.0]], dtype=np.float64)
    input_map = [SegmentIOSlice(node_id=7, offset=0, size=4)]
    out = normalize_boundary_slices_numpy(input_map, seg_input, divisors)
    np.testing.assert_allclose(out, seg_input / divisor, atol=1e-12, rtol=0.0)
    # torch twin agrees bit-for-bit
    out_t = normalize_boundary_slices_torch(
        input_map, torch.tensor(seg_input), divisors,
    )
    np.testing.assert_allclose(out_t.numpy(), out, atol=0.0, rtol=0.0)


def test_neural_and_wire_producers_are_not_rescaled() -> None:
    """Neural producers (counts/T) and ScaleNormalizingWrapper ops are already
    wire-domain: the divisor map must skip them."""
    from mimarsinan.mapping.support.compute_modules import ScaleNormalizingWrapper

    wrapper = ScaleNormalizingWrapper(
        nn.Identity(), [torch.tensor(2.0)], torch.tensor(3.0),
    )
    mapping, op = _synthetic_hybrid_with_compute_producer(wrapper)
    assert boundary_normalization_scales(mapping) == {}

    # Absent divisors: normalization is the identity (same object, no copy).
    seg_input = np.array([[0.25, 0.5]], dtype=np.float64)
    input_map = [SegmentIOSlice(node_id=3, offset=0, size=2)]
    assert normalize_boundary_slices_numpy(input_map, seg_input, {}) is seg_input


def test_value_domain_chain_propagates_through_structural_ops() -> None:
    """encoder(theta) -> structural host op -> neural: the structural op's slice
    still needs the encoder's divisor; a structural op fed by neural outputs
    keeps divisor 1."""
    enc_sources = np.array(
        [IRSource(node_id=-2, index=i) for i in range(4)], dtype=object,
    ).reshape(1, 4)
    enc_module = Perceptron(4, 4, normalization=nn.Identity())
    enc_module.set_activation_scale(2.5)
    enc = ComputeOp(
        id=1, name="enc", op_type="Perceptron", input_sources=enc_sources,
        params={"module": enc_module},
    )
    structural_sources = np.array(
        [IRSource(node_id=1, index=i) for i in range(4)], dtype=object,
    ).reshape(1, 4)
    reshape_op = ComputeOp(
        id=2, name="reshape", op_type="Reshape", input_sources=structural_sources,
        params={"module": nn.Identity()},
    )
    neural_fed_sources = np.array(
        [IRSource(node_id=9, index=i) for i in range(4)], dtype=object,
    ).reshape(1, 4)
    add_op = ComputeOp(
        id=3, name="add", op_type="Add", input_sources=neural_fed_sources,
        params={"module": nn.Identity()},
    )
    stages = [
        HybridStage(kind="compute", name="enc", compute_op=enc),
        HybridStage(kind="compute", name="reshape", compute_op=reshape_op),
        HybridStage(kind="compute", name="add", compute_op=add_op),
    ]
    divisors = boundary_normalization_scales(HybridHardCoreMapping(stages=stages))
    assert set(divisors) == {1, 2}
    assert float(np.asarray(divisors[1])) == pytest.approx(2.5)
    assert float(np.asarray(divisors[2])) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# 4. Emission twin: uniform wire train on both sides of the mirror
# ---------------------------------------------------------------------------

def test_encode_compute_boundary_emits_uniform_wire_train() -> None:
    theta, T = 2.185, 8
    _, hybrid, p1, _ = _tiny_lif_model(T, theta)
    cfg = _config(T)
    op = next(s.compute_op for s in hybrid.stages if s.kind == "compute")
    assert op is not None

    x = (3.0 * torch.rand(2, 8)).to(cfg.compute_dtype)
    from spikingjelly.activation_based import functional

    p1.activation.set_cycle_accurate(False)
    functional.reset_net(p1.activation.if_node)
    with torch.no_grad():
        value = p1(x.to(torch.float32)).to(cfg.compute_dtype)

    emitted = encode_compute_boundary(
        op=op,
        state_buffer={-2: x, int(op.id): value},
        state_buffer_spikes={},
        config=cfg,
        hybrid_mapping=hybrid,
    )
    assert emitted is not None
    expected = uniform_spike_train(
        normalize_boundary_value(value, p1.activation.activation_scale), T,
    ).to(cfg.compute_dtype)
    assert emitted.shape == (T, 2, 6)
    torch.testing.assert_close(emitted, expected, atol=0.0, rtol=0.0)


def test_lif_segment_policy_mirrors_uniform_boundary_emission() -> None:
    """The NF policy's encoding-boundary train must be the same uniform wire
    train (times theta, NF value-domain magnitudes) — pinned through the
    timing-sensitive downstream subtractive-LIF cascade."""
    from spikingjelly.activation_based import functional

    theta, T = 2.185, 8
    repr_, _, p1, p2 = _tiny_lif_model(T, theta, theta_hidden=0.5)
    torch.manual_seed(7)
    x = 3.0 * torch.rand(2, 8)

    driver = SegmentForwardDriver(repr_, T, LifSegmentPolicy())
    with torch.no_grad():
        nf_out = driver(x)

    # Manual mirror: uniform train of the normalized encoder value, then the
    # downstream perceptron per-cycle.
    lif1, lif2 = p1.activation, p2.activation
    lif1.set_cycle_accurate(False)
    functional.reset_net(lif1.if_node)
    with torch.no_grad():
        rate_norm = (p1(x) / lif1.activation_scale.clamp(min=1e-12)).clamp(0.0, 1.0)
        train = uniform_spike_train(rate_norm, T) * lif1.activation_scale
        lif2.set_cycle_accurate(True)
        functional.reset_net(lif2.if_node)
        outs = [p2(train[t]) for t in range(T)]
        lif2.set_cycle_accurate(False)
        expected = torch.stack(outs, dim=0).mean(dim=0)

    torch.testing.assert_close(nf_out, expected, atol=1e-6, rtol=0.0)


def test_nf_driver_equals_hybrid_flow_with_scaled_encoder() -> None:
    """Cross-twin mirror: NF driver and HCM flow agree with a theta != 1 encoder
    (catches one-sided emission changes)."""
    theta, T = 2.185, 8
    repr_, hybrid, _, p2 = _tiny_lif_model(T, theta, theta_hidden=0.5)
    torch.manual_seed(11)
    x = 3.0 * torch.rand(2, 8)

    driver = SegmentForwardDriver(repr_, T, LifSegmentPolicy())
    flow = SpikingHybridCoreFlow(
        (8,), hybrid, simulation_length=T,
        spiking_mode="lif", cycle_accurate_lif_forward=True,
    )
    with torch.no_grad():
        nf = driver(x) / p2.activation.activation_scale.clamp(min=1e-12)
        hc = flow(x) / T
    torch.testing.assert_close(
        nf.to(torch.float32), hc.to(torch.float32), atol=1e-6, rtol=0.0,
    )


# ---------------------------------------------------------------------------
# 5. Shift interplay: normalize, then wire-domain shift, then clamp
# ---------------------------------------------------------------------------

def test_normalization_precedes_wire_domain_shift() -> None:
    input_map = [SegmentIOSlice(node_id=4, offset=0, size=2)]
    divisors = {4: 2.0}
    shifts = {4: np.array([0.25, 0.25])}
    seg_input = np.array([[1.5, 0.5]], dtype=np.float64)

    out_np = apply_input_shifts_numpy(
        input_map,
        normalize_boundary_slices_numpy(input_map, seg_input, divisors),
        shifts,
    ).clip(0.0, 1.0)
    # normalize -> shift -> clamp: clamp(1.5/2 + 0.25) = 1.0, not clamp(1.75/2).
    np.testing.assert_allclose(out_np, [[1.0, 0.5]], atol=1e-12, rtol=0.0)

    out_torch = normalize_boundary_slices_torch(
        input_map, torch.tensor(seg_input), divisors,
    )
    sh = torch.as_tensor(shifts[4], dtype=out_torch.dtype)
    out_torch = (out_torch + sh).clamp(0.0, 1.0)
    np.testing.assert_allclose(out_torch.numpy(), out_np, atol=0.0, rtol=0.0)


def test_compute_input_state_with_shifts_twins() -> None:
    """The host value path gathers a shifted producer LIFTED (the consumer's
    baked bias B' = B - W*s expects it); torch and numpy twins agree."""
    sources = np.array(
        [IRSource(node_id=5, index=0), IRSource(node_id=5, index=1)], dtype=object,
    ).reshape(1, 2)
    op = ComputeOp(id=9, name="enc", op_type="Perceptron", input_sources=sources)
    shifts = {5: np.array([1.0, 0.0])}

    buf_t = {5: torch.tensor([[-0.5, 0.5]], dtype=torch.float64)}
    view_t = compute_input_state_with_shifts(op, buf_t, shifts)
    torch.testing.assert_close(
        view_t[5], torch.tensor([[0.5, 0.5]], dtype=torch.float64),
        atol=0.0, rtol=0.0,
    )
    # The underlying buffer is untouched.
    assert float(buf_t[5][0, 0]) == -0.5

    buf_n = {5: np.array([[-0.5, 0.5]], dtype=np.float64)}
    view_n = compute_input_state_with_shifts(op, buf_n, shifts)
    np.testing.assert_allclose(view_n[5], view_t[5].numpy(), atol=0.0, rtol=0.0)
    assert buf_n[5][0, 0] == -0.5

    # No shifts: identity (same mapping object, zero copies).
    assert compute_input_state_with_shifts(op, buf_n, {}) is buf_n
    assert compute_input_state_with_shifts(op, buf_n, None) is buf_n
    assert compute_input_state_with_shifts(op, buf_n, {77: np.array([1.0])}) is buf_n


# ---------------------------------------------------------------------------
# 6. TTFS regression guard: the W1 (defect-A) contract is untouched
# ---------------------------------------------------------------------------

def test_ttfs_boundary_alias_unchanged() -> None:
    from mimarsinan.spiking.segment_boundary import normalize_ttfs_boundary_value

    v = torch.tensor([[0.5, 3.0]])
    torch.testing.assert_close(
        normalize_ttfs_boundary_value(v, 2.0),
        normalize_boundary_value(v, 2.0),
        atol=0.0, rtol=0.0,
    )
