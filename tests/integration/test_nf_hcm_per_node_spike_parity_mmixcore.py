"""Per-neuron LIF spike-count parity: torch NF nodes == HCM, on mmixcore.

The requirement: the spike count at each SpikingJelly LIF node (per neuron, summed
over T) must equal the corresponding HCM hard-core neuron's recorded count, for
every mixer FC of the mlp_mixer_core ("mmixcore") template under LIF.

This is exact ``k == k`` per-neuron parity, enabled by:
- signed integrate-and-fire (the relu-on-membrane bug was removed) so the node and
  the HCM crossbar share dynamics (see ``test_lif_step_vs_activation_parity``);
- mmixcore being a single neural segment (compute ops — patch-embed encoder, mean,
  classifier — sit only at the ends), so the whole-graph cycle-accurate forward is
  already chip-aligned;
- ``placement.ir_node_id == IR NeuralCore.id`` and the mapper emitting cores in the
  torch output's flattening order, so each node's flat neurons are the concatenation
  (IR-id order) of its cores' neurons.

patch_embed (perceptron 0) is the encoding layer — a Conv wrapper mapped to a host
ComputeOp, with no on-chip core — so per-neuron parity is asserted on the 8 mixer
FCs (perceptron_index 1..8).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.models.torch_mlp_mixer_core import TorchMLPMixerCore
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

from integration._split_reassembly import (
    hcm_per_perceptron_counts,
    torch_lif_node_counts,
)


def _build(
    T, *, patch=4, chan=6, fc1=8, fc2=6, allow_coalescing=False, max_dim=512,
    firing_mode="Default", thresholding_mode="<=",
):
    torch.manual_seed(0)
    m = TorchMLPMixerCore(
        input_shape=(1, 28, 28), num_classes=10,
        patch_n_1=patch, patch_m_1=patch, patch_c_1=chan, fc_w_1=fc1, fc_w_2=fc2,
    )
    m.eval()
    flow = convert_torch_model(m, input_shape=(1, 28, 28), num_classes=10)
    flow.eval()
    repr_ = flow.get_mapper_repr()

    nodes = {}
    for i, p in enumerate(flow.get_perceptrons()):
        lif = LIFActivation(
            T=T, activation_scale=torch.tensor(1.0),
            thresholding_mode=thresholding_mode, firing_mode=firing_mode,
        )
        p.base_activation = lif
        p.activation = lif
        nodes[i] = lif
    mark_encoding_layers(repr_)
    repr_.assign_perceptron_indices()

    ir = IRMapping(
        q_max=127.0, firing_mode=firing_mode,
        max_axons=max_dim, max_neurons=max_dim, allow_coalescing=allow_coalescing,
    ).map(repr_)
    counts = [{"max_axons": max_dim, "max_neurons": max_dim, "count": 2000}]
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=counts,
        strategy=MappingStrategy.from_permissions(
            allow_neuron_splitting=True, allow_coalescing=allow_coalescing),
    )
    flow_hcm = SpikingHybridCoreFlow(
        (1, 28, 28), hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode=firing_mode, spike_mode="Uniform",
        thresholding_mode=thresholding_mode,
        spiking_mode="lif", cycle_accurate_lif_forward=True,
    )
    return flow, nodes, ir, hybrid, flow_hcm


def _run_parity(**build_kwargs):
    T = 4
    flow, nodes, ir, hybrid, flow_hcm = _build(T, **build_kwargs)
    x = torch.rand(1, 1, 28, 28)

    torch_counts = torch_lif_node_counts(
        lambda: chip_aligned_segment_forward(flow, x, T), nodes, x, T)
    with torch.no_grad():
        _, record = flow_hcm.forward_with_recording(x, sample_index=0)
    hcm_counts = hcm_per_perceptron_counts(record, hybrid)

    # Mixer FCs are perceptron_index 1..8; 0 is the patch-embed encoder (host op).
    mixer = sorted(i for i in torch_counts if i >= 1)
    assert mixer == [1, 2, 3, 4, 5, 6, 7, 8], f"unexpected mixer FCs: {mixer}"

    for i in mixer:
        assert i in hcm_counts, f"perceptron {i} missing from HCM record"
        t = torch_counts[i]
        h = hcm_counts[i]
        assert t.shape == h.shape, f"perceptron {i} neuron-count shape {t.shape} vs {h.shape}"
        assert np.array_equal(t, h), (
            f"perceptron {i} per-neuron mismatch: "
            f"{int((t != h).sum())}/{t.size} neurons differ "
            f"(torch_total={int(t.sum())} hcm_total={int(h.sum())})"
        )


def test_per_neuron_spike_parity_mmixcore_lif():
    """Exact per-neuron NF-node == HCM, no coalescing (each FC → grouped cores)."""
    _run_parity(allow_coalescing=False)


def test_per_neuron_spike_parity_mmixcore_lif_neuron_split():
    """Force neuron tiling by shrinking the hard core so wide FCs split across cores;
    per-neuron concatenation must still reconstruct each node exactly."""
    _run_parity(allow_coalescing=False, patch=4, chan=6, fc1=8, fc2=6, max_dim=24)


def test_per_neuron_spike_parity_mmixcore_lif_novena():
    """Novena (zero-reset) cycle-accurate NF == HCM per-neuron: the chip-faithful
    forward the deployment gate requires for Novena. The analytical rate forward
    (cycle_accurate_lif_forward=False) is the one that diverges (~12pp on mmixcore);
    this locks that the cascade path is bit-exact under the zero-reset kernel."""
    _run_parity(allow_coalescing=False, firing_mode="Novena", thresholding_mode="<")
