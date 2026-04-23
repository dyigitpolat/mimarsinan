"""Verify the HCM simulator's bank-aware weight resolution.

After the bank-aware HCM refactor, softcores produced from shared
``WeightBank``s must flow through a deduplicated bank tensor in the
simulator's segment cache (``bank_tensors[bank_id]``) rather than being
materialised as per-hardcore dense copies.  The per-hw-core resolved
weight ``core_params[ci]`` is expected to be a zero-copy **view** into
the bank tensor when the placement exactly covers the hw core's
occupied rectangle, so N hw cores sharing one bank keep exactly one
resident tensor on device.

These tests assert:

1. When the IRGraph has weight banks, ``HardCoreMapping.weight_banks``
   carries them through.
2. ``SoftCore`` and placement-dict metadata both carry bank provenance
   (bank_id + axon / neuron ranges) end-to-end.
3. The segment cache uploads the bank once and every single-placement
   bank-backed hw core points at a view of that single tensor (no
   per-core copy).
4. HCM forward output is numerically identical between the bank-view
   path and an equivalent dense upload — proving the refactor
   preserves correctness.
"""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.mapping.ir import (
    IRGraph, IRSource, NeuralCore, WeightBank,
    ir_graph_to_soft_core_mapping,
)
from mimarsinan.mapping.softcore_mapping import HardCoreMapping, HardCore
from mimarsinan.mapping.hybrid_hardcore_mapping import (
    HybridHardCoreMapping, HybridStage, SegmentIOSlice,
)
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow


def _build_bank_graph(
    n_positions: int,
    in_feats: int,
    out_feats: int,
    *,
    distinct_threshold_groups: bool = False,
):
    """Build a tiny IRGraph with one shared bank and ``n_positions`` shared cores.

    When ``distinct_threshold_groups`` is True, each position gets a
    unique ``perceptron_index`` so the packer is forced to assign them
    to separate hardware cores (cross-hw-core bank dedup path).  When
    False (default), all positions share the same perceptron and the
    packer naturally output-tiles them into one hw core (single-hw-core
    block-diagonal layout — the common conv pattern).
    """
    rng = np.random.default_rng(0)
    W = rng.standard_normal((out_feats, in_feats)).astype(np.float32)
    scale = 127.0 / max(float(np.max(np.abs(W))), 1e-12)
    W_q = np.clip(np.round(W * scale), -128, 127).astype(np.int8)
    # Bank core_matrix is stored as (in_features, out_features).
    bank_matrix = W_q.T.copy()
    bank = WeightBank(
        id=0,
        core_matrix=bank_matrix,
        activation_scale=torch.tensor(1.0),
        parameter_scale=torch.tensor(1.0),
        input_activation_scale=torch.tensor(1.0),
        hardware_bias=None,
    )

    nodes = []
    for i in range(n_positions):
        src = np.array(
            [IRSource(node_id=-2, index=j) for j in range(in_feats)],
            dtype=object,
        )
        nc = NeuralCore(
            id=i,
            name=f"pos{i}",
            input_sources=src,
            core_matrix=None,
            threshold=float(scale),
            weight_bank_id=0,
            weight_row_slice=(0, out_feats),
            latency=0,
            perceptron_index=(i if distinct_threshold_groups else 0),
        )
        nodes.append(nc)

    out_srcs = np.array(
        [IRSource(node_id=nodes[0].id, index=k) for k in range(out_feats)],
        dtype=object,
    )
    return IRGraph(nodes=nodes, output_sources=out_srcs, weight_banks={0: bank})


def _build_single_stage_hcm(ir_graph: IRGraph, pool_size: int = 8) -> HybridHardCoreMapping:
    """Pack the IRGraph into a single-stage HybridHardCoreMapping."""
    soft = ir_graph_to_soft_core_mapping(ir_graph)
    pool = [HardCore(256, 256, has_bias_capability=True) for _ in range(pool_size)]
    hard = HardCoreMapping(pool)
    hard.map(soft)
    stage = HybridStage(
        kind="neural",
        name="seg0",
        hard_core_mapping=hard,
        input_map=[SegmentIOSlice(node_id=-2, offset=0, size=ir_graph.nodes[0].get_input_count())],
        output_map=[
            SegmentIOSlice(
                node_id=ir_graph.nodes[0].id,
                offset=0,
                size=ir_graph.nodes[0].weight_row_slice[1],
            )
        ],
    )
    return HybridHardCoreMapping(stages=[stage], output_sources=ir_graph.output_sources)


class TestBankPathTaken:
    def test_hardcore_mapping_carries_banks(self):
        ir = _build_bank_graph(n_positions=4, in_feats=8, out_feats=6)
        soft = ir_graph_to_soft_core_mapping(ir)
        assert 0 in soft.weight_banks, "SoftCoreMapping must expose bank matrices"
        hard = HardCoreMapping([HardCore(256, 256) for _ in range(4)])
        hard.map(soft)
        assert 0 in hard.weight_banks, "HardCoreMapping.map must propagate banks"

    def test_softcore_keeps_bank_reference(self):
        ir = _build_bank_graph(n_positions=2, in_feats=8, out_feats=6)
        soft = ir_graph_to_soft_core_mapping(ir)
        for sc in soft.cores:
            assert sc.weight_bank_id == 0
            assert sc.bank_neuron_slice == (0, 6)
            assert sc.bank_axon_slice == (0, 8)

    def test_placement_dict_records_bank_fields(self):
        ir = _build_bank_graph(n_positions=3, in_feats=8, out_feats=6)
        hcm = _build_single_stage_hcm(ir, pool_size=4)
        mapping = hcm.stages[0].hard_core_mapping
        seen_bank_placements = 0
        for placements in mapping.soft_core_placements_per_hard_core:
            for p in placements:
                if "weight_bank_id" in p:
                    assert p["weight_bank_id"] == 0
                    assert p["bank_axon_range"] == (0, 8)
                    assert p["bank_neuron_range"] == (0, 6)
                    seen_bank_placements += 1
        assert seen_bank_placements >= 1

    def test_segment_cache_shares_bank_tensor(self):
        """Cross-hw-core bank dedup: when positions can't share one hw
        core (distinct threshold groups), each lands in its own hw core
        and they all view a single uploaded bank tensor."""
        ir = _build_bank_graph(
            n_positions=3, in_feats=8, out_feats=6,
            distinct_threshold_groups=True,
        )
        hcm = _build_single_stage_hcm(ir, pool_size=4)
        mapping = hcm.stages[0].hard_core_mapping
        assert len(mapping.cores) >= 3, (
            "distinct threshold groups should force one hw core per softcore"
        )
        flow = SpikingHybridCoreFlow(
            input_shape=(8,),
            hybrid_mapping=hcm,
            simulation_length=4,
            firing_mode="TTFS",
            spike_mode="TTFS",
            thresholding_mode="<=",
            spiking_mode="ttfs_quantized",
        )
        seg = flow._get_segment_tensors(hcm.stages[0], torch.device("cpu"))
        bank_t = seg["bank_tensors"][0]
        assert bank_t.shape == (6, 8), "bank uploaded as (out_features, in_features)"
        views_into_bank = sum(
            1 for w in seg["core_params"] if w.data_ptr() == bank_t.data_ptr()
        )
        assert views_into_bank == len(mapping.cores), (
            "every hw core should reference the bank tensor as a view"
        )


class TestBankVsDenseEquivalence:
    """Numerical equivalence: bank-view output == dense-upload output."""

    def test_output_matches_dense_path(self):
        ir = _build_bank_graph(n_positions=3, in_feats=8, out_feats=6)
        hcm = _build_single_stage_hcm(ir, pool_size=4)
        device = torch.device("cpu")
        x = torch.randn(4, 8)

        flow_a = SpikingHybridCoreFlow(
            input_shape=(8,), hybrid_mapping=hcm, simulation_length=4,
            firing_mode="TTFS", spike_mode="TTFS",
            thresholding_mode="<=", spiking_mode="ttfs_quantized",
        )
        out_bank = flow_a(x.clone())

        # Force the dense path: prime the cache, then replace every
        # core_params tensor with a materialised copy (detached view →
        # contiguous tensor).  The next forward runs through dense
        # upload semantics with the identical weights.
        flow_b = SpikingHybridCoreFlow(
            input_shape=(8,), hybrid_mapping=hcm, simulation_length=4,
            firing_mode="TTFS", spike_mode="TTFS",
            thresholding_mode="<=", spiking_mode="ttfs_quantized",
        )
        seg = flow_b._get_segment_tensors(hcm.stages[0], device)
        seg["core_params"] = [w.clone().contiguous() for w in seg["core_params"]]
        # Clear bank_tensors so the dense path can't accidentally view it.
        seg["bank_tensors"] = {}
        out_dense = flow_b(x.clone())

        torch.testing.assert_close(out_bank, out_dense, rtol=0, atol=0)
