"""The conversion boundary algebra as executable spec (conversion_boundary_algebra.md).

Micro-fixture: on-chip LIF segment -> plain SIGNED host op (LayerNorm) ->
on-chip LIF segment, under encoding_layer_placement=offload — the ViT seam
topology at unit scale. The identities I1/I3 are violated by today's code
exactly here (V-A host-op currency, V-B the kappa divisor hole, V-D sigma);
those tests are `xfail(strict=True)`: they document the violation now and
fail LOUD the moment the unification lands, forcing promotion to plain
asserts. Subsume/homogeneous seams are GREEN locks and must stay green
through the fix (tier-0 inertness).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.activations.autograd import ChipInputQuantizer
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.spiking.segment_boundary import boundary_normalization_scales
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

T = 32
THETA_1 = 1.7   # producer theta == kappa_fold at the LN seam (the ViT's 1.03-4.14)
THETA_2 = 0.9


def _lif_perceptron(out_ch, in_features, theta):
    p = Perceptron(out_ch, in_features, normalization=nn.Identity())
    p.set_activation_scale(theta)
    lif = LIFActivation(T=T, activation_scale=p.activation_scale)
    lif.use_cycle_accurate_trains = True
    p.base_activation = lif
    p.activation = lif
    return p


def _signed_seam_model():
    """input(8) -> P1(6, THETA_1) -> LayerNorm(6) -> P2(3, THETA_2), offload.

    The LN is a plain (unwrapped) host ComputeOp producing SIGNED values; the
    production fold bakes kappa_fold = THETA_1 into P2's effective weights.
    """
    torch.manual_seed(0)
    inp = InputMapper((8,))
    p1 = _lif_perceptron(6, 8, THETA_1)
    m1 = PerceptronMapper(inp, p1)
    ln = nn.LayerNorm(6)
    with torch.no_grad():
        ln.weight.fill_(1.0)
        # beta=0.5 puts seam mass into (1, kappa]: the raw-domain clamp
        # (V-B/V-C) deletes it while the /kappa encode keeps it.
        ln.bias.fill_(0.5)
    host = ComputeOpMapper(m1, ln, input_shape=(6,), output_shape=(6,))
    p2 = _lif_perceptron(3, 6, THETA_2)
    p2.per_input_scales = torch.full((6,), float(THETA_1))
    p2.set_input_activation_scale(torch.tensor(float(THETA_1)))
    m2 = PerceptronMapper(host, p2)
    repr_ = ModelRepresentation(m2)
    mark_encoding_layers(repr_, placement="offload")
    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=32, max_neurons=32,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 4}],
    )
    return repr_, hybrid, p1, p2, host


# ---------------------------------------------------------------------------
# T1 — V-B: the temporal entry divisor must equal kappa_fold/kappa_buf.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason="V-B: boundary_normalization_scales carries NO entry for a plain "
           "host producer (measured: empty table on this fixture — the "
           "divisor silently defaults to the identity); the consumer fold "
           "and trained entry expect value/kappa_fold "
           "(conversion_boundary_algebra.md sec.4)",
)
def test_t1_plain_host_divisor_equals_kappa_fold():
    _, hybrid, _, _, _ = _signed_seam_model()
    divisors = boundary_normalization_scales(hybrid)
    values = [
        float(torch.as_tensor(v).float().mean())
        for v in divisors.values() if v is not None
    ]
    # The LN seam's divisor must carry the fold currency (kappa_fold), not
    # default-by-absence to 1.
    assert values and max(values) == pytest.approx(THETA_1)


# ---------------------------------------------------------------------------
# T2 — GREEN lock: the temporal family is internally consistent.
# ---------------------------------------------------------------------------

def test_t2_nf_equals_hcm_across_signed_seam():
    """NF and the HCM twin agree bit-for-bit on the signed seam — TODAY, on
    the shared (wrong) convention (measured: 0.000000 here, and 32/32 argmax
    on the offloaded large-backbone cell). P3 must move BOTH sides jointly:
    this lock holding through the fix is the joint-movement proof."""
    repr_, hybrid, _, p2, _ = _signed_seam_model()
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
# T3 — I3: the temporal walk must match the value-domain composition
#          within the grid-noise bound (the host op computes on VALUES).
# ---------------------------------------------------------------------------

def _value_domain_reference(p1, p2, host, x):
    """The analytic composition: values everywhere, seams encode divide-first."""
    from mimarsinan.models.spiking.wire_semantics import lif_count_staircase

    with torch.no_grad():
        z1 = p1.layer(x)
        v1 = lif_count_staircase(
            z1, p1.activation.activation_scale, T, compare_mode="<=",
        ).clamp(min=0.0)
        h = host.module(v1)                       # host computes on VALUE
        kappa = float(p2.input_activation_scale)  # the fold currency
        r = (h / kappa).clamp(0.0, 1.0)           # divide-first seam encode
        v_in = torch.round(r * T) / T * kappa
        z2 = torch.nn.functional.linear(v_in, p2.layer.weight, p2.layer.bias)
        v2 = lif_count_staircase(
            z2, p2.activation.activation_scale, T, compare_mode="<=",
        ).clamp(min=0.0)
    return v2


@pytest.mark.xfail(
    strict=True,
    reason="V-B/V-C: the temporal seam clamps the RAW signed value at [0, 1] "
           "(divisor hole + clamp-then-scale) — a deterministic per-seam "
           "bias that shifts the MEAN far beyond the zero-mean grid noise "
           "(memo sec.3 composition laws)",
)
def test_t3_temporal_matches_value_domain_reference():
    repr_, _, p1, p2, host = _signed_seam_model()
    torch.manual_seed(13)
    x = 3.0 * torch.rand(64, 8)
    driver = SegmentForwardDriver(repr_, T, LifSegmentPolicy())
    with torch.no_grad():
        nf = driver(x)
    ref = _value_domain_reference(p1, p2, host, x)
    # Grid noise is zero-mean (memo sec.3): the batch-mean absolute gap must
    # sit within the per-seam grid scale. A convention mismatch is a
    # deterministic bias and blows through it.
    bound = float(p2.input_activation_scale) / (2 * T)
    assert float((nf - ref).abs().mean()) <= bound


# ---------------------------------------------------------------------------
# kappa_fold has ONE propagation: NF table == IR table, nodewise.
# ---------------------------------------------------------------------------

def test_kappa_fold_nf_and_ir_tables_agree_nodewise():
    """GREEN lock: read_boundary_out_scales (NF twin) and
    compute_node_output_scales (IR SSOT) resolve the same currencies — the
    permanent contract P2's generalization must keep."""
    from mimarsinan.mapping.ir import ComputeOp, NeuralCore
    from mimarsinan.mapping.support.activation_scales import (
        compute_node_output_scales,
    )
    from mimarsinan.spiking.scale_aware_boundaries import read_boundary_out_scales

    torch.manual_seed(0)
    inp = InputMapper((8,))
    p1 = _lif_perceptron(6, 8, THETA_1)
    m1 = PerceptronMapper(inp, p1)
    ln = nn.LayerNorm(6)
    host = ComputeOpMapper(m1, ln, input_shape=(6,), output_shape=(6,))
    p2 = _lif_perceptron(3, 6, THETA_2)
    p2.per_input_scales = torch.full((6,), float(THETA_1))
    m2 = PerceptronMapper(host, p2)
    repr_ = ModelRepresentation(m2)
    mark_encoding_layers(repr_, placement="offload")

    repr_._ensure_exec_graph()
    nf_scales = read_boundary_out_scales(repr_, input_data_scale=1.0)
    nf_by_type: dict[str, list[float]] = {}
    for node in repr_._exec_order:
        nf_by_type.setdefault(type(node).__name__, []).append(
            float(torch.as_tensor(nf_scales[node]).float().mean())
        )

    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=32, max_neurons=32,
    ).map(repr_)
    ir_scales = compute_node_output_scales(ir)
    ir_by_kind: dict[str, list[float]] = {}
    for node in ir.nodes:
        kind = type(node).__name__
        if isinstance(node, (NeuralCore, ComputeOp)):
            ir_by_kind.setdefault(kind, []).append(
                float(torch.as_tensor(ir_scales[node.id]).float().mean())
            )

    # Neural producers carry theta; the plain host op is a pass-through of
    # its producer's currency — in BOTH tables.
    assert sorted(nf_by_type["PerceptronMapper"]) == pytest.approx(
        sorted(ir_by_kind["NeuralCore"])
    )
    assert nf_by_type["ComputeOpMapper"] == pytest.approx(ir_by_kind["ComputeOp"])
    assert nf_by_type["ComputeOpMapper"] == [pytest.approx(THETA_1)]


# ---------------------------------------------------------------------------
# T4 — sigma value preservation at kappa_fold != 1 (the V-D identity pair).
# ---------------------------------------------------------------------------

def _sigma_bake_fixture():
    """Signed values v, real calibration sigma, real bias bake B' = B - W.sigma."""
    from mimarsinan.mapping.support.bias_compensation import apply_negative_shift_bias

    torch.manual_seed(5)
    p = _lif_perceptron(3, 6, THETA_2)
    # Range chosen so (v + sigma)/kappa stays in [0, 1]: the required
    # composition is then clamp-free and must be EXACT.
    v = torch.rand(16, 6) * 1.2 - 0.4
    sigma = (-v.amin(dim=0)).clamp(min=0.0)
    weight = p.layer.weight.detach().clone()
    bias_before = p.layer.bias.detach().clone()
    apply_negative_shift_bias(p, sigma)
    bias_baked = p.layer.bias.detach().clone()
    return v, sigma, weight, bias_before, bias_baked


def test_t4_required_sigma_composition_preserves_value():
    """GREEN lock: E(v) = (v + sigma)/kappa_fold with the baked bias is exactly
    W.v + B — the identity P3 must implement (memo sec.1 bias bake)."""
    v, sigma, weight, bias_before, bias_baked = _sigma_bake_fixture()
    kappa = THETA_1
    wire = ((v + sigma) / kappa).clamp(0.0, 1.0)
    charge = torch.nn.functional.linear(kappa * wire, weight) + bias_baked
    exact = torch.nn.functional.linear(v, weight) + bias_before
    torch.testing.assert_close(charge, exact, atol=1e-5, rtol=0.0)


@pytest.mark.xfail(
    strict=True,
    reason="V-D x V-B: today's temporal entry adds raw sigma at the identity "
           "divisor; with kappa_fold != 1 the consumer charge is off by the "
           "kappa factor and saturates early — value preservation breaks",
)
def test_t4_todays_sigma_composition_preserves_value():
    v, sigma, weight, bias_before, bias_baked = _sigma_bake_fixture()
    kappa = THETA_1
    wire = (v + sigma).clamp(0.0, 1.0)   # divisor 1, sigma raw
    charge = torch.nn.functional.linear(kappa * wire, weight) + bias_baked
    exact = torch.nn.functional.linear(v, weight) + bias_before
    torch.testing.assert_close(charge, exact, atol=1e-5, rtol=0.0)


# ---------------------------------------------------------------------------
# T5 — I1: the trained entry op equals the deployed seam composition.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason="V-B/I1: the trained ChipInputQuantizer divides by kappa_fold; "
           "today's deployed seam clamps at the identity divisor — the two "
           "compositions disagree on any value above 1 or below 0",
)
def test_t5_trained_entry_equals_deployed_seam():
    kappa = THETA_1
    quantizer = ChipInputQuantizer(T=T, activation_scale=torch.tensor(kappa))
    v = torch.linspace(-2.0, 2.5, 19)   # signed, wide-range seam values
    with torch.no_grad():
        trained = quantizer(v)
    # Today's deployed seam: clamp at divisor 1, grid, re-scale by kappa via
    # the consumer weight fold.
    deployed = torch.round(v.clamp(0.0, 1.0) * T) / T * kappa
    torch.testing.assert_close(trained, deployed, atol=1e-6, rtol=0.0)


# ---------------------------------------------------------------------------
# T6 — GREEN lock: gauge classification is grounded in homogeneity.
# ---------------------------------------------------------------------------

class TestT6HomogeneityGrounding:
    """wire_transparent membership requires f(a·x) = a·f(x) for a > 0; LN and
    softmax must fail it (they are value_ops). Pure math — GREEN forever."""

    @pytest.mark.parametrize("op", [
        nn.MaxPool2d(2), nn.AvgPool2d(2), nn.Identity(),
    ])
    def test_homogeneous_ops_commute_with_scale(self, op):
        torch.manual_seed(3)
        x = torch.randn(2, 3, 8, 8)
        for alpha in (0.5, 1.7, 3.0):
            torch.testing.assert_close(
                op(alpha * x), alpha * op(x), atol=1e-6, rtol=1e-6,
            )

    def test_layernorm_is_not_homogeneous_across_channels(self):
        # Per-CHANNEL scaling (the per-channel kappa case) does not commute.
        torch.manual_seed(4)
        ln = nn.LayerNorm(6)
        x = torch.randn(5, 6)
        alpha = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        assert not torch.allclose(ln(alpha * x), alpha * ln(x), atol=1e-3)


# ---------------------------------------------------------------------------
# T7 — GREEN lock: the subsume/homogeneous seam is bit-stable (inertness pin).
# ---------------------------------------------------------------------------

class _HostRelay(nn.Module):
    def forward(self, x):
        return x * 1.0


def test_t7_subsume_homogeneous_seam_nf_equals_hcm():
    """The tier-0-shaped seam (encoding perceptron -> homogeneous relay ->
    perceptron, subsume): NF == HCM today and MUST stay identical through the
    unification (the wire_transparent path is byte-preserved)."""
    torch.manual_seed(0)
    inp = InputMapper((8,))
    p1 = _lif_perceptron(6, 8, THETA_1)
    p1.is_encoding_layer = True
    m1 = PerceptronMapper(inp, p1)
    host = ComputeOpMapper(m1, _HostRelay(), input_shape=(6,), output_shape=(6,))
    p2 = _lif_perceptron(3, 6, THETA_2)
    p2.per_input_scales = torch.full((6,), float(THETA_1))
    m2 = PerceptronMapper(host, p2)
    repr_ = ModelRepresentation(m2)
    mark_encoding_layers(repr_)
    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=32, max_neurons=32,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 4}],
    )
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
