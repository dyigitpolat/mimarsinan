"""Round-2b subsume vs offload switch for encoding-layer neuralOps.

`mark_encoding_layers(placement="offload")` leaves segment-start perceptrons
unmarked so the mappers emit them as on-chip NeuralCores (instead of host
ComputeOps), enlarging the hardware-accelerated surface. The segment input is then
encoded directly by the flow (Uniform/TTFS) — the encode arm that already exists.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.models.torch_mlp_mixer_core import TorchMLPMixerCore
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.ir import NeuralCore, ComputeOp


def _map(placement):
    model = TorchMLPMixerCore(
        input_shape=(1, 28, 28), num_classes=10,
        patch_n_1=4, patch_m_1=4, patch_c_1=6, fc_w_1=8, fc_w_2=6,
    )
    model.eval()
    flow = convert_torch_model(model, input_shape=(1, 28, 28), num_classes=10)
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_, placement=placement)
    repr_.assign_perceptron_indices()
    return IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=512, max_neurons=512,
    ).map(repr_)


def test_offload_maps_encoding_layer_on_chip():
    sub = _map("subsume")
    off = _map("offload")

    sub_perc = {getattr(c, "perceptron_index", None)
                for c in sub.nodes if isinstance(c, NeuralCore)}
    off_perc = {getattr(c, "perceptron_index", None)
                for c in off.nodes if isinstance(c, NeuralCore)}

    # patch_embed is perceptron_index 0: a host ComputeOp under subsume (absent from
    # neural cores), an on-chip NeuralCore under offload.
    assert 0 not in sub_perc
    assert 0 in off_perc

    n_sub_cores = sum(isinstance(n, NeuralCore) for n in sub.nodes)
    n_off_cores = sum(isinstance(n, NeuralCore) for n in off.nodes)
    n_sub_compute = sum(isinstance(n, ComputeOp) for n in sub.nodes)
    n_off_compute = sum(isinstance(n, ComputeOp) for n in off.nodes)

    assert n_off_cores > n_sub_cores       # encoding layer now on-chip
    assert n_off_compute < n_sub_compute   # one fewer host ComputeOp


def test_offload_placement_validates():
    model = TorchMLPMixerCore(
        input_shape=(1, 28, 28), num_classes=10,
        patch_n_1=4, patch_m_1=4, patch_c_1=6, fc_w_1=8, fc_w_2=6,
    )
    flow = convert_torch_model(model, input_shape=(1, 28, 28), num_classes=10)
    with pytest.raises(ValueError, match="placement"):
        mark_encoding_layers(flow.get_mapper_repr(), placement="bogus")


def _build_hcm(placement, T=4):
    import torch.nn as nn
    from mimarsinan.models.nn.activations import LIFActivation
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
    from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

    torch.manual_seed(0)
    m = TorchMLPMixerCore(
        input_shape=(1, 28, 28), num_classes=10,
        patch_n_1=4, patch_m_1=4, patch_c_1=6, fc_w_1=8, fc_w_2=6,
    )
    m.eval()
    flow = convert_torch_model(
        m, input_shape=(1, 28, 28), num_classes=10,
        encoding_layer_placement=placement,
    )
    repr_ = flow.get_mapper_repr()
    for p in flow.get_perceptrons():
        lif = LIFActivation(T=T, activation_scale=torch.tensor(1.0), thresholding_mode="<=")
        p.base_activation = lif
        p.activation = lif
    repr_.assign_perceptron_indices()
    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=512, max_neurons=512,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 512, "max_neurons": 512, "count": 400}],
        allow_neuron_splitting=True,
    )
    return SpikingHybridCoreFlow(
        (1, 28, 28), hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode="Default", spike_mode="Uniform", thresholding_mode="<=",
        spiking_mode="lif", cycle_accurate_lif_forward=True,
    )


def test_offload_runs_through_hcm_and_matches_subsume():
    T = 4
    sub = _build_hcm("subsume", T)
    off = _build_hcm("offload", T)
    x = torch.rand(1, 1, 28, 28)
    with torch.no_grad():
        out_sub = sub(x)
        out_off = off(x)
    assert out_sub.shape == out_off.shape == (1, 10)
    # The encoding layer computes identically host-side (subsume) vs on-chip
    # (offload) — both uniform-encode the input then run the same signed-IF LIF.
    torch.testing.assert_close(out_off, out_sub, atol=1e-6, rtol=0.0)
