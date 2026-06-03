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

from collections import defaultdict

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.models.torch_mlp_mixer_core import TorchMLPMixerCore
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward


def _build(T, *, patch=4, chan=6, fc1=8, fc2=6, allow_coalescing=False, max_dim=512):
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
        lif = LIFActivation(T=T, activation_scale=torch.tensor(1.0), thresholding_mode="<=")
        p.base_activation = lif
        p.activation = lif
        nodes[i] = lif
    mark_encoding_layers(repr_)
    repr_.assign_perceptron_indices()

    ir = IRMapping(
        q_max=127.0, firing_mode="Default",
        max_axons=max_dim, max_neurons=max_dim, allow_coalescing=allow_coalescing,
    ).map(repr_)
    counts = [{"max_axons": max_dim, "max_neurons": max_dim, "count": 2000}]
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=counts,
        allow_neuron_splitting=True, allow_coalescing=allow_coalescing,
    )
    flow_hcm = SpikingHybridCoreFlow(
        (1, 28, 28), hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode="Default", spike_mode="Uniform", thresholding_mode="<=",
        spiking_mode="lif", cycle_accurate_lif_forward=True,
    )
    return flow, nodes, ir, hybrid, flow_hcm


def _torch_node_counts(flow, nodes, x, T):
    """Per-LIF-node, per-neuron spike counts (flattened, summed over T cycles)."""
    acc = {i: None for i in nodes}

    def mk(i):
        def hook(mod, inp, out):
            v = out.detach().reshape(out.shape[0], -1)  # scale=1 → binary spikes
            acc[i] = v.clone() if acc[i] is None else acc[i] + v
        return hook

    handles = [nodes[i].register_forward_hook(mk(i)) for i in nodes]
    try:
        with torch.no_grad():
            chip_aligned_segment_forward(flow, x, T)
    finally:
        for h in handles:
            h.remove()
    return {i: acc[i][0].numpy().astype(np.int64) for i in acc if acc[i] is not None}


def _hcm_node_counts(record, hybrid):
    """Per-perceptron, per-neuron HCM counts: concat of the perceptron's IR cores
    (IR-id order, local-neuron order), filtering coalescing/psum partial cores.

    ``placement.ir_node_id == IR NeuralCore.id`` (compaction preserves ids and the
    segment remap rewrites only sources), so cores group + order by id directly.
    """
    # ir_id -> list of (orig_neuron_offset, perceptron_index, counts). Neuron-split
    # fragments of one IR core share its id; reassemble them by original offset.
    frags: dict[int, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
    for si, seg in record.segments.items():
        placements = hybrid.stages[si].hard_core_mapping.soft_core_placements_per_hard_core
        for ci, cc in enumerate(seg.cores):
            for pl in placements[ci]:
                pi = pl.get("perceptron_index")
                if pi is None or pi < 0:
                    continue
                if pl.get("coalescing_role") not in (None, "master", "accum"):
                    continue
                if pl.get("psum_role") not in (None, "accum"):
                    continue
                n0 = pl["neuron_offset"]
                n1 = n0 + pl["neurons"]
                counts = np.asarray(cc.output_spike_count[n0:n1], dtype=np.int64)
                orig_off = (
                    int(pl["neuron_range_in_original"][0])
                    if pl.get("split_group_id") is not None else 0
                )
                frags[pl["ir_node_id"]].append((orig_off, pi, counts))

    # Reassemble each IR core's neurons (fragments in original-offset order).
    per_core: dict[int, tuple[int, np.ndarray]] = {}
    for ir_id, flist in frags.items():
        flist.sort(key=lambda f: f[0])
        per_core[ir_id] = (flist[0][1], np.concatenate([c for _, _, c in flist]))

    by_perc: dict[int, list[int]] = defaultdict(list)
    for ir_id, (pi, _) in per_core.items():
        by_perc[pi].append(ir_id)
    return {
        pi: np.concatenate([per_core[cid][1] for cid in sorted(ids)])
        for pi, ids in by_perc.items()
    }


def _run_parity(**build_kwargs):
    T = 4
    flow, nodes, ir, hybrid, flow_hcm = _build(T, **build_kwargs)
    x = torch.rand(1, 1, 28, 28)

    torch_counts = _torch_node_counts(flow, nodes, x, T)
    with torch.no_grad():
        _, record = flow_hcm.forward_with_recording(x, sample_index=0)
    hcm_counts = _hcm_node_counts(record, hybrid)

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
