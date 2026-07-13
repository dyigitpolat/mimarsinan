"""[R3] Per-channel theta torch↔deployed-sim parity on mmixcore under LIF.

The lossless wave v3 regression (t0_01_lif_mmixcore_wq_s4, 0.8945 < 0.98):
promoting unequal per-channel theta arms ``per_source_scales`` on the host
ComputeOps downstream of a promoted producer, so IR emission wraps them in
``ScaleNormalizingWrapper`` (the deployed sim decodes ``wire * theta_c`` per
channel) — while the chip-aligned torch twin ran the RAW module on undecoded
wire rates. Both sides must consume the vector: the twin routes wire-domain
host value nodes through the same emitted wrapper composition.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from integration._split_reassembly import hcm_per_perceptron_counts

from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.torch_mlp_mixer_core import TorchMLPMixerCore
from mimarsinan.pipelining.core.nf_scm_parity import (
    assert_torch_vs_deployed_sim_parity_or_raise,
)
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.spiking.per_channel_theta import eligible_per_channel_perceptrons
from mimarsinan.spiking.scale_aware_boundaries import (
    calibrate_scale_aware_boundaries,
    propagate_boundary_input_scales,
)
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

T = 4
INPUT_SHAPE = (1, 28, 28)


class _TwinForward(nn.Module):
    """The pipeline's chip-aligned NF forward (the gate's torch side)."""

    def __init__(self, flow, T):
        super().__init__()
        self.flow = flow
        self.T = T

    def forward(self, x):
        return chip_aligned_segment_forward(self.flow, x, self.T)


def _build(promote: bool):
    torch.manual_seed(0)
    model = TorchMLPMixerCore(
        input_shape=INPUT_SHAPE, num_classes=10,
        patch_n_1=4, patch_m_1=4, patch_c_1=6, fc_w_1=8, fc_w_2=6,
    ).eval()
    flow = convert_torch_model(model, input_shape=INPUT_SHAPE, num_classes=10).eval()
    repr_ = flow.get_mapper_repr()

    perceptrons = list(flow.get_perceptrons())
    for p in perceptrons:
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        p.base_activation = lif
        p.activation = lif
    mark_encoding_layers(repr_)
    repr_.assign_perceptron_indices()

    maxima = {}
    handles = []
    for i, p in enumerate(perceptrons):
        def hook(_m, _i, out, idx=i):
            maxima[idx] = out.detach().abs().max().item()
        handles.append(p.activation.register_forward_hook(hook))
    torch.manual_seed(7)
    with torch.no_grad():
        flow(torch.rand(8, *INPUT_SHAPE))
    for h in handles:
        h.remove()
    scales = [max(maxima[i], 1e-3) for i in range(len(perceptrons))]
    calibrate_scale_aware_boundaries(flow, scales, input_data_scale=1.0)

    promoted = []
    if promote:
        for p in eligible_per_channel_perceptrons(flow).values():
            base = float(p.activation_scale)
            p.set_activation_scale(
                base * torch.linspace(0.6, 1.6, p.output_channels)
            )
            promoted.append(p.name)
        propagate_boundary_input_scales(flow, input_data_scale=1.0)

    compute_per_source_scales(repr_)

    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=512, max_neurons=512,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 512, "max_neurons": 512, "count": 2000}],
        strategy=MappingStrategy.from_permissions(
            allow_neuron_splitting=True, allow_coalescing=False),
    )
    flow_hcm = SpikingHybridCoreFlow(
        INPUT_SHAPE, hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode="Default", spike_mode="Uniform", thresholding_mode="<=",
        spiking_mode="lif", cycle_accurate_lif_forward=True,
    )
    return flow, flow_hcm, hybrid, promoted


def _samples(n=16):
    torch.manual_seed(1)
    return torch.rand(n, *INPUT_SHAPE)


def test_scalar_theta_twin_and_deployed_stay_bit_consistent():
    """Byte-identical scalar contract: no wrap arms, deployed == T * twin exactly."""
    flow, flow_hcm, _, _ = _build(promote=False)
    x = _samples()
    with torch.no_grad():
        twin = chip_aligned_segment_forward(flow, x, T)
        deployed = flow_hcm(x)
    assert float((deployed - twin * T).abs().max()) == 0.0


def test_promoted_unequal_theta_passes_the_deployed_parity_gate():
    """The t0_01 regression pin: unequal promoted theta must hold the exact gate
    that tripped in the field (assert_torch_vs_deployed_sim_parity_or_raise)."""
    flow, flow_hcm, _, promoted = _build(promote=True)
    assert promoted, "fixture must promote at least one per-channel theta"
    thetas = {
        p.name: p.activation_scale for p in flow.get_perceptrons()
        if torch.is_tensor(p.activation_scale) and p.activation_scale.dim() > 0
    }
    assert any(
        not torch.allclose(v, v[0].expand_as(v)) for v in thetas.values()
    ), "promoted thetas must be genuinely unequal"

    agreement = assert_torch_vs_deployed_sim_parity_or_raise(
        _TwinForward(flow, T), flow_hcm, _samples(), min_agreement=0.98,
    )
    assert agreement == 1.0


def test_promoted_unequal_theta_twin_and_deployed_logits_match():
    """The seam pin: the twin must run the emitted ScaleNormalizingWrapper
    composition at wire-domain host value nodes, so deployed == T * twin."""
    flow, flow_hcm, _, _ = _build(promote=True)

    wrapped = [
        n for n in flow.get_mapper_repr()._exec_order
        if isinstance(n, ComputeOpMapper)
        and n.per_source_scales is not None and n.output_scale is not None
    ]
    assert wrapped, (
        "fixture must arm per_source_scales on host ops downstream of a "
        "promoted producer (the mean -> classifier tail)"
    )

    x = _samples()
    with torch.no_grad():
        twin = chip_aligned_segment_forward(flow, x, T)
        deployed = flow_hcm(x)
    torch.testing.assert_close(
        deployed.to(torch.float32), (twin * T).to(torch.float32),
        atol=1e-4, rtol=1e-4,
    )


def test_promoted_unequal_theta_per_neuron_counts_stay_exact():
    """Intra-segment vector folding is exact: every mixer FC's per-neuron spike
    count matches between the twin's LIF nodes and the HCM record."""
    flow, flow_hcm, hybrid, _ = _build(promote=True)
    x = _samples(1)

    nodes = {
        i: p.activation for i, p in enumerate(flow.get_perceptrons())
        if isinstance(p.activation, LIFActivation)
    }
    counts: dict[int, np.ndarray] = {}
    handles = []
    for i, lif in nodes.items():
        def hook(module, _inp, out, idx=i):
            scale = module.activation_scale
            s = scale.detach().to(out.dtype)
            spikes = (out / s.clamp(min=1e-12)).round()
            flat = spikes.detach().reshape(-1).cpu().numpy()
            counts[idx] = counts.get(idx, 0) + flat
        handles.append(lif.register_forward_hook(hook))
    try:
        with torch.no_grad():
            chip_aligned_segment_forward(flow, x, T)
    finally:
        for h in handles:
            h.remove()

    with torch.no_grad():
        _, record = flow_hcm.forward_with_recording(x, sample_index=0)
    hcm_counts = hcm_per_perceptron_counts(record, hybrid)

    mixer = sorted(i for i in hcm_counts if i >= 1)
    assert mixer, "expected on-chip mixer FCs in the HCM record"
    for i in mixer:
        t = np.asarray(counts[i], dtype=np.int64)
        h = hcm_counts[i]
        assert t.shape == h.shape
        assert np.array_equal(t, h), (
            f"perceptron {i}: {int((t != h).sum())}/{t.size} neurons differ"
        )
