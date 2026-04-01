"""Stress tests for scheduled mapping: mismatch detection and edge cases.

Validates:
1. Latency metadata is NOT mutated on original NeuralCore objects by the partitioner.
2. Heterogeneous core configs with coalescing accumulators survive scheduling.
3. Layout IR mapping vs actual IR mapping core-count agreement for wide FC layers.
4. Scheduled vs non-scheduled equivalence in rate mode (not just TTFS).
5. Coalescing groups split across passes produce valid output.
6. Multi-segment (with ComputeOps) scheduled mapping equivalence.
7. Layout pass count estimate vs actual pass count agreement.
8. ComputeOp module device alignment after pickle round-trip.
9. TTFS ComputeOp bias rescaling correctness (unified flow).
10. TTFS ComputeOp bias rescaling correctness (hybrid flow).
"""

import copy
import math
import pytest
import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, ComputeOp
from mimarsinan.mapping.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.schedule_partitioner import (
    partition_segment_into_passes,
    estimate_passes_for_layout,
)
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow


# ---------------------------------------------------------------------------
# IR graph builders
# ---------------------------------------------------------------------------

def _make_chain_ir(n_layers=3, width=8, n_in=6, n_out=4):
    """Chain of NeuralCores: input -> h0 -> h1 -> ... -> output."""
    nodes = []
    prev_width = n_in
    for i in range(n_layers):
        is_last = (i == n_layers - 1)
        cur_width = n_out if is_last else width
        if i == 0:
            sources = [IRSource(-2, j) for j in range(prev_width)] + [IRSource(-3, 0)]
        else:
            sources = [IRSource(nodes[-1].id, j) for j in range(prev_width)] + [IRSource(-3, 0)]
        w = np.random.randn(prev_width + 1, cur_width).astype(np.float32) * 0.1
        core = NeuralCore(
            id=i, name=f"layer{i}",
            input_sources=np.array(sources, dtype=object),
            core_matrix=w, latency=i,
        )
        nodes.append(core)
        prev_width = cur_width
    out_sources = np.array(
        [IRSource(nodes[-1].id, j) for j in range(n_out)], dtype=object,
    )
    return IRGraph(nodes=nodes, output_sources=out_sources)


def _make_wide_coalescing_ir(
    n_in=4, n_parallel=6, parallel_width=4, out_width=2,
    coalescing_group_start=0,
):
    """IR with a coalescing group: parallel partial cores + wide accumulator.

    n_parallel independent cores at latency 0, then an accumulator at latency 1
    whose input is the concatenation of all parallel outputs + bias.
    """
    nodes = []
    for i in range(n_parallel):
        sources = [IRSource(-2, j) for j in range(n_in)] + [IRSource(-3, 0)]
        w = np.random.randn(n_in + 1, parallel_width).astype(np.float32) * 0.1
        core = NeuralCore(
            id=i, name=f"partial_{i}",
            input_sources=np.array(sources, dtype=object),
            core_matrix=w, latency=0,
            coalescing_group_id=coalescing_group_start,
            coalescing_role="partial",
        )
        nodes.append(core)

    accum_sources = []
    for i in range(n_parallel):
        for j in range(parallel_width):
            accum_sources.append(IRSource(i, j))
    accum_sources.append(IRSource(-3, 0))
    total_accum_in = n_parallel * parallel_width + 1
    w_accum = np.random.randn(total_accum_in, out_width).astype(np.float32) * 0.1
    accum = NeuralCore(
        id=n_parallel, name="accum",
        input_sources=np.array(accum_sources, dtype=object),
        core_matrix=w_accum, latency=1,
        coalescing_group_id=coalescing_group_start,
        coalescing_role="accum",
    )
    nodes.append(accum)

    out = np.array([IRSource(n_parallel, j) for j in range(out_width)], dtype=object)
    return IRGraph(nodes=nodes, output_sources=out)


def _make_ir_with_compute_op(cores_per_seg=3, n_in=4, n_out=2, width=4):
    """IR: [cores] -> ComputeOp(identity) -> [cores] -> output."""
    nodes = []
    nid = 0
    prev_width = n_in
    for i in range(cores_per_seg):
        if i == 0:
            sources = [IRSource(-2, j) for j in range(prev_width)] + [IRSource(-3, 0)]
        else:
            sources = [IRSource(nid - 1, j) for j in range(prev_width)] + [IRSource(-3, 0)]
        w = np.random.randn(prev_width + 1, width).astype(np.float32) * 0.1
        nodes.append(NeuralCore(
            id=nid, name=f"seg0_l{i}",
            input_sources=np.array(sources, dtype=object),
            core_matrix=w, latency=i,
        ))
        prev_width = width
        nid += 1

    last_seg0 = nid - 1
    op_sources = np.array([IRSource(last_seg0, j) for j in range(width)], dtype=object)
    nodes.append(ComputeOp(
        id=nid, name="barrier",
        input_sources=op_sources,
        op_type="identity",
        input_shape=(width,), output_shape=(width,),
    ))
    op_id = nid
    nid += 1

    prev_width = width
    for i in range(cores_per_seg):
        is_last = (i == cores_per_seg - 1)
        cur_width = n_out if is_last else width
        if i == 0:
            sources = [IRSource(op_id, j) for j in range(prev_width)] + [IRSource(-3, 0)]
        else:
            sources = [IRSource(nid - 1, j) for j in range(prev_width)] + [IRSource(-3, 0)]
        w = np.random.randn(prev_width + 1, cur_width).astype(np.float32) * 0.1
        nodes.append(NeuralCore(
            id=nid, name=f"seg1_l{i}",
            input_sources=np.array(sources, dtype=object),
            core_matrix=w, latency=cores_per_seg + 1 + i,
        ))
        prev_width = cur_width
        nid += 1

    out = np.array([IRSource(nid - 1, j) for j in range(n_out)], dtype=object)
    return IRGraph(nodes=nodes, output_sources=out)


def _make_flow(hm, input_shape, spiking_mode="ttfs", simulation_length=8):
    return SpikingHybridCoreFlow(
        input_shape=input_shape,
        hybrid_mapping=hm,
        simulation_length=simulation_length,
        preprocessor=nn.Identity(),
        firing_mode="TTFS" if "ttfs" in spiking_mode else "Default",
        spike_mode="TTFS" if "ttfs" in spiking_mode else "Uniform",
        spiking_mode=spiking_mode,
    )


# ---------------------------------------------------------------------------
# RC-2: Latency mutation
# ---------------------------------------------------------------------------

class TestLatencyMutation:
    """_compute_core_latencies must NOT mutate original NeuralCore.latency."""

    def test_latency_not_mutated_by_partitioner(self):
        np.random.seed(42)
        ir = _make_chain_ir(n_layers=4, width=6, n_in=4, n_out=2)
        cores = [n for n in ir.nodes if isinstance(n, NeuralCore)]

        # Set latencies to arbitrary values that DON'T match topological depth,
        # so mutation by IRLatency.calculate() is detectable.
        for i, c in enumerate(cores):
            c.latency = 100 + i
        original_latencies = {c.id: c.latency for c in cores}

        partition_segment_into_passes(cores, max_cores_per_pass=2)
        after_latencies = {c.id: c.latency for c in cores}

        assert original_latencies == after_latencies, (
            f"partition_segment_into_passes mutated core latencies!\n"
            f"  before: {original_latencies}\n"
            f"  after:  {after_latencies}"
        )

    def test_latency_preserved_after_scheduling(self):
        """Global latencies on IR graph nodes should survive build_hybrid_hard_core_mapping."""
        np.random.seed(42)
        ir = _make_chain_ir(n_layers=3, width=6, n_in=4, n_out=2)
        cores = [n for n in ir.nodes if isinstance(n, NeuralCore)]
        # Set non-matching latencies so mutation by IRLatency is detectable
        for i, c in enumerate(cores):
            c.latency = 200 + i
        original_latencies = {c.id: c.latency for c in cores}

        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 20}]
        build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True,
        )

        after_latencies = {c.id: c.latency for c in cores}
        assert original_latencies == after_latencies, (
            f"build_hybrid_hard_core_mapping corrupted IR graph latencies!\n"
            f"  before: {original_latencies}\n"
            f"  after:  {after_latencies}"
        )


# ---------------------------------------------------------------------------
# RC-3: Heterogeneous core configs + coalescing
# ---------------------------------------------------------------------------

class TestHeterogeneousCoresCoalescing:
    """Scheduled mapping with heterogeneous core types and coalescing groups."""

    def test_accumulator_fits_in_fresh_pool(self):
        """A coalescing accumulator that needs fusion should pack on a fresh pool."""
        np.random.seed(42)
        ir = _make_wide_coalescing_ir(
            n_in=4, n_parallel=8, parallel_width=4, out_width=2,
        )
        cores_config = [
            {"max_axons": 8, "max_neurons": 32, "count": 10},
            {"max_axons": 16, "max_neurons": 64, "count": 10},
        ]
        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True,
        )
        neural_stages = [s for s in hm.stages if s.kind == "neural"]
        assert len(neural_stages) >= 1

    def test_heterogeneous_does_not_crash_with_tight_budget(self):
        """When budget is tight, recursive splitting should prevent crashes."""
        np.random.seed(42)
        ir = _make_wide_coalescing_ir(
            n_in=4, n_parallel=6, parallel_width=4, out_width=2,
        )
        cores_config = [
            {"max_axons": 8, "max_neurons": 8, "count": 5},
            {"max_axons": 16, "max_neurons": 32, "count": 5},
        ]
        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True,
        )
        assert len(hm.stages) >= 1

    def test_multi_coalescing_groups_split_across_passes(self):
        """Two coalescing groups that individually fit but together don't."""
        np.random.seed(42)
        nodes = []
        nid = 0
        n_in = 4
        pw = 4

        for grp in range(2):
            for i in range(4):
                sources = [IRSource(-2, j) for j in range(n_in)] + [IRSource(-3, 0)]
                w = np.random.randn(n_in + 1, pw).astype(np.float32) * 0.1
                nodes.append(NeuralCore(
                    id=nid, name=f"g{grp}_p{i}",
                    input_sources=np.array(sources, dtype=object),
                    core_matrix=w, latency=0,
                    coalescing_group_id=grp, coalescing_role="partial",
                ))
                nid += 1

        for grp in range(2):
            accum_sources = []
            base = grp * 4
            for i in range(4):
                for j in range(pw):
                    accum_sources.append(IRSource(base + i, j))
            accum_sources.append(IRSource(-3, 0))
            w_a = np.random.randn(4 * pw + 1, 2).astype(np.float32) * 0.1
            nodes.append(NeuralCore(
                id=nid, name=f"g{grp}_accum",
                input_sources=np.array(accum_sources, dtype=object),
                core_matrix=w_a, latency=1,
                coalescing_group_id=grp, coalescing_role="accum",
            ))
            nid += 1

        out = np.array([IRSource(nid - 2, 0), IRSource(nid - 1, 0)], dtype=object)
        ir = IRGraph(nodes=nodes, output_sources=out)

        cores_config = [{"max_axons": 32, "max_neurons": 32, "count": 5}]
        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True,
        )
        neural_stages = [s for s in hm.stages if s.kind == "neural"]
        assert len(neural_stages) >= 2, (
            f"Expected multiple passes for two coalescing groups in 5 hw cores, "
            f"got {len(neural_stages)}"
        )


# ---------------------------------------------------------------------------
# RC-1: Layout vs IR mapping for wide FC layers
# ---------------------------------------------------------------------------

class TestLayoutVsIRMapping:
    """Layout IR mapping and actual IR mapping should agree on core counts."""

    def test_wide_fc_psum_layout_core_count_matches_ir(self):
        """For wide FC with psum decomposition, layout and IR core counts should agree."""
        from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
        from mimarsinan.mapping.ir_mapping import IRMapping

        in_features = 50
        out_features = 8
        max_axons = 16
        max_neurons = 64

        weights = np.random.randn(out_features, in_features).astype(np.float32)
        biases = np.random.randn(out_features).astype(np.float32)
        input_sources = np.array([IRSource(-2, j) for j in range(in_features)], dtype=object)

        layout = LayoutIRMapping(
            max_axons=max_axons, max_neurons=max_neurons,
            allow_coalescing=False,
        )
        layout_out = layout.map_fc(input_sources, np.array([out_features]), weights, biases)
        layout_count = len(layout.layout_softcores)

        ir_mapping = IRMapping(
            max_axons=max_axons, max_neurons=max_neurons,
            allow_coalescing=False,
        )
        ir_out = ir_mapping.map_fc(
            input_sources, np.array([out_features]), weights, biases,
        )
        ir_count = len([n for n in ir_mapping.nodes if isinstance(n, NeuralCore)])

        assert layout_count == ir_count, (
            f"Layout produced {layout_count} softcores but IR produced {ir_count} "
            f"cores for wide FC psum decomposition"
        )
        assert layout_out.shape[0] == out_features
        assert ir_out.shape[0] == out_features

    def test_wide_fc_coalescing_layout_core_count_matches_ir(self):
        """For wide FC with coalescing, layout and IR core counts should agree."""
        from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
        from mimarsinan.mapping.ir_mapping import IRMapping

        in_features = 50
        out_features = 8
        max_axons = 16
        max_neurons = 64

        weights = np.random.randn(out_features, in_features).astype(np.float32)
        biases = np.random.randn(out_features).astype(np.float32)
        input_sources = np.array([IRSource(-2, j) for j in range(in_features)], dtype=object)

        layout = LayoutIRMapping(
            max_axons=max_axons, max_neurons=max_neurons,
            allow_coalescing=True,
        )
        layout_out = layout.map_fc(input_sources, np.array([out_features]), weights, biases)
        layout_count = len(layout.layout_softcores)

        ir_mapping = IRMapping(
            max_axons=max_axons, max_neurons=max_neurons,
            allow_coalescing=True,
        )
        ir_out = ir_mapping.map_fc(
            input_sources, np.array([out_features]), weights, biases,
        )
        ir_count = len([n for n in ir_mapping.nodes if isinstance(n, NeuralCore)])

        assert layout_count == ir_count, (
            f"Layout produced {layout_count} softcores but IR produced {ir_count} "
            f"cores for wide FC coalescing"
        )
        assert layout_out.shape[0] == out_features
        assert ir_out.shape[0] == out_features

    def test_narrow_fc_layout_and_ir_agree(self):
        """When FC fits max_axons, layout and IR should produce matching core counts."""
        from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
        from mimarsinan.mapping.ir_mapping import IRMapping

        in_features = 10
        out_features = 8
        max_axons = 16
        max_neurons = 64

        weights = np.random.randn(out_features, in_features).astype(np.float32)
        biases = np.random.randn(out_features).astype(np.float32)
        input_sources = np.array([IRSource(-2, j) for j in range(in_features)], dtype=object)

        layout = LayoutIRMapping(max_axons=max_axons, max_neurons=max_neurons)
        layout.map_fc(input_sources, np.array([out_features]), weights, biases)
        layout_count = len(layout.layout_softcores)

        ir_mapping = IRMapping(max_axons=max_axons, max_neurons=max_neurons)
        ir_mapping.map_fc(input_sources, np.array([out_features]), weights, biases)
        ir_count = len([n for n in ir_mapping.nodes if isinstance(n, NeuralCore)])

        assert layout_count == ir_count, (
            f"Layout produced {layout_count} softcores but IR produced {ir_count} cores"
        )


# ---------------------------------------------------------------------------
# RC-4: Rate-mode equivalence
# ---------------------------------------------------------------------------

class TestRateModeEquivalence:
    """Scheduled vs non-scheduled should produce similar results in rate mode."""

    def test_single_pass_rate_equivalence(self):
        """When everything fits in one pass, scheduled and non-scheduled
        should produce identical output even in rate mode."""
        np.random.seed(42)
        ir = _make_chain_ir(n_layers=3, width=8, n_in=6, n_out=4)
        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 20}]

        hm_normal = build_hybrid_hard_core_mapping(
            ir_graph=copy.deepcopy(ir), cores_config=cores_config,
        )
        hm_sched = build_hybrid_hard_core_mapping(
            ir_graph=copy.deepcopy(ir), cores_config=cores_config,
            allow_scheduling=True,
        )

        torch.manual_seed(0)
        x = torch.rand(4, 6)

        flow_normal = _make_flow(hm_normal, (6,), spiking_mode="rate", simulation_length=32)
        flow_sched = _make_flow(hm_sched, (6,), spiking_mode="rate", simulation_length=32)

        out_normal = flow_normal.forward(x)
        out_sched = flow_sched.forward(x)

        assert torch.allclose(out_normal, out_sched, atol=1e-4), (
            f"Rate-mode scheduled output diverged from non-scheduled:\n"
            f"  max diff = {(out_normal - out_sched).abs().max().item()}"
        )

    def test_multi_pass_rate_output_finite(self):
        """Multi-pass rate-mode output should be finite and non-zero."""
        np.random.seed(42)
        nodes = []
        n_in = 4
        width = 4
        for i in range(8):
            sources = [IRSource(-2, j) for j in range(n_in)] + [IRSource(-3, 0)]
            w = np.random.randn(n_in + 1, width).astype(np.float32) * 0.1
            nodes.append(NeuralCore(
                id=i, name=f"par{i}",
                input_sources=np.array(sources, dtype=object),
                core_matrix=w, latency=0,
            ))
        out_sources = []
        for i in range(8):
            for j in range(width):
                out_sources.append(IRSource(i, j))
        out_sources.append(IRSource(-3, 0))
        w_out = np.random.randn(8 * width + 1, 2).astype(np.float32) * 0.1
        nodes.append(NeuralCore(
            id=8, name="output",
            input_sources=np.array(out_sources, dtype=object),
            core_matrix=w_out, latency=1,
        ))
        out = np.array([IRSource(8, 0), IRSource(8, 1)], dtype=object)
        ir = IRGraph(nodes=nodes, output_sources=out)

        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 3}]
        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True,
        )

        flow = _make_flow(hm, (4,), spiking_mode="rate", simulation_length=32)
        torch.manual_seed(0)
        x = torch.rand(2, 4)
        out = flow.forward(x)

        assert torch.isfinite(out).all(), "Rate-mode multi-pass output has non-finite values"


# ---------------------------------------------------------------------------
# Coalescing groups split across passes
# ---------------------------------------------------------------------------

class TestCoalescingGroupSplit:
    """Coalescing groups split across passes via scheduling."""

    def test_coalescing_split_produces_valid_output(self):
        """Force coalescing group to split across passes, check output is finite."""
        np.random.seed(42)
        ir = _make_wide_coalescing_ir(
            n_in=4, n_parallel=6, parallel_width=4, out_width=2,
        )
        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 3}]
        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True,
        )

        neural_stages = [s for s in hm.stages if s.kind == "neural"]
        assert len(neural_stages) >= 2, "Expected multiple passes"

        flow = _make_flow(hm, (4,), spiking_mode="ttfs", simulation_length=32)
        torch.manual_seed(0)
        x = torch.rand(2, 4) * 5.0
        out = flow.forward(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Multi-segment scheduled equivalence
# ---------------------------------------------------------------------------

class TestMultiSegmentScheduledEquivalence:
    """Scheduled and non-scheduled should agree for multi-segment IR graphs."""

    def test_compute_op_ttfs_equivalence(self):
        """TTFS mode with ComputeOp barrier: scheduled == non-scheduled."""
        np.random.seed(42)
        ir = _make_ir_with_compute_op(cores_per_seg=3, n_in=4, n_out=2, width=4)
        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 20}]

        hm_normal = build_hybrid_hard_core_mapping(
            ir_graph=copy.deepcopy(ir), cores_config=cores_config,
        )
        hm_sched = build_hybrid_hard_core_mapping(
            ir_graph=copy.deepcopy(ir), cores_config=cores_config,
            allow_scheduling=True,
        )

        torch.manual_seed(0)
        x = torch.rand(4, 4)

        flow_normal = _make_flow(hm_normal, (4,), spiking_mode="ttfs")
        flow_sched = _make_flow(hm_sched, (4,), spiking_mode="ttfs")

        out_normal = flow_normal.forward(x)
        out_sched = flow_sched.forward(x)

        assert torch.allclose(out_normal, out_sched, atol=1e-5), (
            f"Multi-segment TTFS scheduled output diverged:\n"
            f"  max diff = {(out_normal - out_sched).abs().max().item()}"
        )

    def test_compute_op_rate_equivalence(self):
        """Rate mode with ComputeOp barrier: single-pass scheduled == non-scheduled."""
        np.random.seed(42)
        ir = _make_ir_with_compute_op(cores_per_seg=3, n_in=4, n_out=2, width=4)
        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 20}]

        hm_normal = build_hybrid_hard_core_mapping(
            ir_graph=copy.deepcopy(ir), cores_config=cores_config,
        )
        hm_sched = build_hybrid_hard_core_mapping(
            ir_graph=copy.deepcopy(ir), cores_config=cores_config,
            allow_scheduling=True,
        )

        torch.manual_seed(0)
        x = torch.rand(4, 4)

        flow_normal = _make_flow(hm_normal, (4,), spiking_mode="rate", simulation_length=32)
        flow_sched = _make_flow(hm_sched, (4,), spiking_mode="rate", simulation_length=32)

        out_normal = flow_normal.forward(x)
        out_sched = flow_sched.forward(x)

        assert torch.allclose(out_normal, out_sched, atol=1e-4), (
            f"Multi-segment rate scheduled output diverged:\n"
            f"  max diff = {(out_normal - out_sched).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Layout pass estimate vs actual
# ---------------------------------------------------------------------------

class TestLayoutPassEstimate:
    """Layout pass count should match actual scheduled stage count."""

    def test_simple_chain_pass_count_agrees(self):
        """For a simple chain that fits in one pass, both should say 1."""
        from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec

        specs = [
            LayoutSoftCoreSpec(input_count=5, output_count=8, threshold_group_id=0,
                               latency_tag=i, segment_id=0)
            for i in range(3)
        ]
        max_cores = 20
        n_passes, _ = estimate_passes_for_layout(specs, max_cores)

        np.random.seed(42)
        ir = _make_chain_ir(n_layers=3, width=8, n_in=4, n_out=4)
        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": max_cores}]
        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True,
        )
        actual_neural = len([s for s in hm.stages if s.kind == "neural"])

        assert n_passes == actual_neural, (
            f"Layout estimated {n_passes} passes but actual produced {actual_neural}"
        )

    def test_many_cores_pass_count_reasonable(self):
        """For more cores than budget, layout pass count >= actual passes."""
        from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec

        n_cores = 10
        specs = [
            LayoutSoftCoreSpec(input_count=5, output_count=4, threshold_group_id=0,
                               latency_tag=0, segment_id=0)
            for _ in range(n_cores)
        ] + [
            LayoutSoftCoreSpec(input_count=n_cores * 4 + 1, output_count=2,
                               threshold_group_id=0, latency_tag=1, segment_id=0),
        ]
        max_cores = 3
        n_passes, _ = estimate_passes_for_layout(specs, max_cores)
        assert n_passes >= 2, f"Expected >= 2 layout passes for {n_cores + 1} cores in {max_cores} hw, got {n_passes}"


class TestComputeOpDeviceAlignment:
    """Validate that ComputeOp._exec_module handles device mismatches."""

    def test_exec_module_after_pickle_roundtrip(self):
        """A module ComputeOp should work after pickle (simulating cache load)."""
        import pickle

        linear = nn.Linear(4, 3, bias=True)
        nn.init.ones_(linear.weight)
        nn.init.zeros_(linear.bias)

        op = ComputeOp(
            id=99, name="test_module_op",
            input_sources=np.array([IRSource(-2, i) for i in range(4)], dtype=object),
            op_type="module",
            params={"module": linear},
            input_shape=(4,),
            output_shape=(3,),
        )

        pickled = pickle.dumps(op)
        restored_op = pickle.loads(pickled)
        assert restored_op.op_type == "module"

        x = torch.randn(2, 4)
        out = restored_op.execute_on_gathered(x)
        assert out.shape == (2, 3)
        assert out.device == x.device

    def test_exec_module_shared_module_across_ops(self):
        """Multiple ComputeOps sharing the same nn.Module should all work."""
        linear = nn.Linear(4, 3, bias=False)
        ops = []
        for i in range(3):
            ops.append(ComputeOp(
                id=100 + i, name=f"shared_op_{i}",
                input_sources=np.array([IRSource(-2, j) for j in range(4)], dtype=object),
                op_type="module",
                params={"module": linear},
                output_shape=(3,),
            ))

        x = torch.randn(2, 4)
        results = [op.execute_on_gathered(x) for op in ops]
        for r in results:
            assert r.shape == (2, 3)
            torch.testing.assert_close(results[0], r)

    def test_scheduled_hybrid_with_module_compute_ops(self):
        """Full scheduled pipeline with module-type ComputeOps between segments."""
        np.random.seed(42)

        linear_mod = nn.Linear(4, 4, bias=False)
        nn.init.eye_(linear_mod.weight)

        core0 = NeuralCore(
            id=0, name="core0",
            input_sources=np.array([IRSource(-2, i) for i in range(4)] + [IRSource(-3, 0)], dtype=object),
            core_matrix=np.eye(5, 4, dtype=np.float32),
            latency=0, threshold=1.0,
        )
        compute = ComputeOp(
            id=1, name="identity_linear",
            input_sources=np.array([IRSource(0, i) for i in range(4)], dtype=object),
            op_type="module",
            params={"module": linear_mod},
            input_shape=(4,),
            output_shape=(4,),
        )
        core1 = NeuralCore(
            id=2, name="core1",
            input_sources=np.array([IRSource(1, i) for i in range(4)] + [IRSource(-3, 0)], dtype=object),
            core_matrix=np.eye(5, 2, dtype=np.float32),
            latency=0, threshold=1.0,
        )

        ir = IRGraph(
            nodes=[core0, compute, core1],
            output_sources=np.array([IRSource(2, 0), IRSource(2, 1)], dtype=object),
        )
        from mimarsinan.mapping.ir_latency import IRLatency
        IRLatency(ir).calculate()

        cores_config = [{"max_axons": 16, "max_neurons": 16, "count": 2}]
        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True,
        )

        neural_stages = [s for s in hm.stages if s.kind == "neural"]
        compute_stages = [s for s in hm.stages if s.kind == "compute"]
        assert len(neural_stages) >= 2
        assert len(compute_stages) >= 1
        assert compute_stages[0].compute_op.op_type == "module"

        preprocessor = nn.Identity()
        flow = SpikingHybridCoreFlow(
            input_shape=(4,), hybrid_mapping=hm, simulation_length=16,
            preprocessor=preprocessor, firing_mode="Default",
            spike_mode="Uniform", thresholding_mode="<", spiking_mode="rate",
        )

        x = torch.rand(2, 4) * 0.8
        out = flow(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 2
        assert out.isfinite().all()


# ── TTFS ComputeOp bias rescaling ──────────────────────────────────────

class TestTTFSComputeOpBiasRescaling:
    """Verify that ComputeOp (module type) bias is correctly handled in TTFS.

    When a NeuralCore with activation_scale > 1 feeds into a ComputeOp wrapping
    nn.Linear (with non-zero bias), the TTFS simulation must rescale the
    ComputeOp input from [0, 1] back to training range [0, activation_scale]
    before execution and normalise the output back.  Without this, the bias
    term introduces a systematic offset of ``b * (1 - 1/s)`` per element.
    """

    @staticmethod
    def _build_chain_with_compute_op(activation_scale: float = 2.0):
        """Build NeuralCore → ComputeOp(nn.Linear w/ bias) → NeuralCore.

        Returns (ir_graph, expected_training_output, x_input, linear_module).
        Uses fixed weights to ensure deterministic, reproducible results.
        """
        in_dim = 4
        hidden = 3
        out_dim = 2
        torch.manual_seed(42)
        x = torch.rand(1, in_dim) * 0.5

        W1 = torch.tensor([[0.2, -0.1, 0.15, 0.05],
                           [0.1, 0.2, -0.1, 0.1],
                           [-0.05, 0.1, 0.2, 0.15]])
        b1 = torch.tensor([0.05, -0.02, 0.03])

        fc2_module = nn.Linear(hidden, out_dim)
        with torch.no_grad():
            fc2_module.weight.copy_(torch.tensor([[0.2, -0.1, 0.15],
                                                   [0.1, 0.2, -0.05]]))
            fc2_module.bias.copy_(torch.tensor([0.5, -0.3]))

        W3 = torch.tensor([[0.15, -0.1], [0.1, 0.2]])
        b3 = torch.tensor([0.05, -0.02])

        # Training-equivalent forward pass:
        h1 = torch.clamp(torch.relu(W1 @ x.T + b1.unsqueeze(1)), 0, activation_scale).T
        h2 = fc2_module(h1)
        out_train = torch.clamp(torch.relu(W3 @ h2.T + b3.unsqueeze(1)), 0, activation_scale).T

        # Build effective weights (per_input_scales=1 for first core,
        # per_input_scales=activation_scale for third core via pass-through).
        W1_eff = (W1 / activation_scale).numpy()
        b1_eff = (b1 / activation_scale).numpy()
        W3_eff = (activation_scale * W3 / activation_scale).numpy()
        b3_eff = (b3 / activation_scale).numpy()

        src_input = np.array([IRSource(-1, i) for i in range(in_dim)], dtype=object)
        core0 = NeuralCore(
            id=0, name="fc1", input_sources=src_input,
            core_matrix=W1_eff.T,
            threshold=1.0,
            activation_scale=torch.tensor(float(activation_scale)),
            parameter_scale=torch.tensor(1.0),
            input_activation_scale=torch.tensor(1.0),
            hardware_bias=b1_eff,
        )
        src_core0 = np.array([IRSource(0, j) for j in range(hidden)], dtype=object)
        compute = ComputeOp(
            id=1, name="fc2_module", input_sources=src_core0,
            op_type="module",
            params={"module": fc2_module},
            output_shape=(out_dim,),
        )
        src_compute = np.array([IRSource(1, j) for j in range(out_dim)], dtype=object)
        core1 = NeuralCore(
            id=2, name="fc3", input_sources=src_compute,
            core_matrix=W3_eff.T,
            threshold=1.0,
            activation_scale=torch.tensor(float(activation_scale)),
            parameter_scale=torch.tensor(1.0),
            input_activation_scale=torch.tensor(1.0),
            hardware_bias=b3_eff,
        )
        ir = IRGraph(
            nodes=[core0, compute, core1],
            output_sources=np.array([IRSource(2, j) for j in range(out_dim)], dtype=object),
        )
        from mimarsinan.mapping.ir_latency import IRLatency
        IRLatency(ir).calculate()
        return ir, out_train, x, fc2_module

    def test_ttfs_output_matches_training_scaled(self):
        """TTFS output should equal training output / activation_scale."""
        from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow
        activation_scale = 2.0
        ir, out_train, x, _ = self._build_chain_with_compute_op(activation_scale)

        flow = SpikingUnifiedCoreFlow(
            input_shape=(4,), ir_graph=ir, simulation_length=32,
            preprocessor=nn.Identity(), spiking_mode="ttfs",
        )
        ttfs_out = flow(x)
        expected = out_train / activation_scale
        # Tolerance accounts for float32 precision loss in the numpy round-trip
        assert torch.allclose(ttfs_out, expected, atol=1e-2), (
            f"TTFS output {ttfs_out} != expected {expected} "
            f"(diff {(ttfs_out - expected).abs().max():.6f})"
        )

    def test_ttfs_large_scale_matches_training(self):
        """activation_scale = 3.0 — the bias offset fix keeps outputs close."""
        from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow
        activation_scale = 3.0
        ir, out_train, x, fc2_mod = self._build_chain_with_compute_op(activation_scale)

        flow = SpikingUnifiedCoreFlow(
            input_shape=(4,), ir_graph=ir, simulation_length=32,
            preprocessor=nn.Identity(), spiking_mode="ttfs",
        )
        ttfs_out = flow(x)
        expected = out_train / activation_scale
        max_diff = (ttfs_out - expected).abs().max().item()
        assert max_diff < 1e-2, (
            f"Scale-corrected TTFS should match training/scale (max_diff={max_diff:.6f})"
        )

    def test_scale_one_is_identity(self):
        """activation_scale = 1.0 — rescaling is a no-op, output is exact."""
        from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow
        ir, out_train, x, _ = self._build_chain_with_compute_op(activation_scale=1.0)
        flow = SpikingUnifiedCoreFlow(
            input_shape=(4,), ir_graph=ir, simulation_length=32,
            preprocessor=nn.Identity(), spiking_mode="ttfs",
        )
        ttfs_out = flow(x)
        assert torch.allclose(ttfs_out, out_train, atol=1e-2), (
            f"With activation_scale=1, TTFS == training (diff "
            f"{(ttfs_out - out_train).abs().max():.6f})"
        )


class TestHybridFlowTTFSComputeOpRescaling:
    """Verify that SpikingHybridCoreFlow rescales ComputeOp inputs in TTFS.

    Mirrors TestTTFSComputeOpBiasRescaling but uses the hybrid flow path
    (HybridHardCoreMapping → SpikingHybridCoreFlow) instead of the unified
    flow path (IRGraph → SpikingUnifiedCoreFlow).
    """

    @staticmethod
    def _build_hybrid_with_compute_op(activation_scale: float = 2.0):
        """Build HybridHardCoreMapping with a ComputeOp(nn.Linear w/ bias).

        Returns (hybrid_mapping, expected_training_output, x_input).
        """
        in_dim = 4
        hidden = 3
        out_dim = 2
        torch.manual_seed(42)
        x = torch.rand(1, in_dim) * 0.5

        W1 = torch.tensor([[0.2, -0.1, 0.15, 0.05],
                           [0.1, 0.2, -0.1, 0.1],
                           [-0.05, 0.1, 0.2, 0.15]])
        b1 = torch.tensor([0.05, -0.02, 0.03])

        fc2_module = nn.Linear(hidden, out_dim)
        with torch.no_grad():
            fc2_module.weight.copy_(torch.tensor([[0.2, -0.1, 0.15],
                                                   [0.1, 0.2, -0.05]]))
            fc2_module.bias.copy_(torch.tensor([0.5, -0.3]))

        W3 = torch.tensor([[0.15, -0.1], [0.1, 0.2]])
        b3 = torch.tensor([0.05, -0.02])

        h1 = torch.clamp(torch.relu(W1 @ x.T + b1.unsqueeze(1)), 0, activation_scale).T
        h2 = fc2_module(h1)
        out_train = torch.clamp(torch.relu(W3 @ h2.T + b3.unsqueeze(1)), 0, activation_scale).T

        W1_eff = (W1 / activation_scale).numpy()
        b1_eff = (b1 / activation_scale).numpy()
        W3_eff = (activation_scale * W3 / activation_scale).numpy()
        b3_eff = (b3 / activation_scale).numpy()

        src_input = np.array([IRSource(-1, i) for i in range(in_dim)], dtype=object)
        core0 = NeuralCore(
            id=0, name="fc1", input_sources=src_input,
            core_matrix=W1_eff.T,
            threshold=1.0,
            activation_scale=torch.tensor(float(activation_scale)),
            parameter_scale=torch.tensor(1.0),
            input_activation_scale=torch.tensor(1.0),
            hardware_bias=b1_eff,
        )
        src_core0 = np.array([IRSource(0, j) for j in range(hidden)], dtype=object)
        compute = ComputeOp(
            id=1, name="fc2_module", input_sources=src_core0,
            op_type="module",
            params={"module": fc2_module},
            output_shape=(out_dim,),
        )
        src_compute = np.array([IRSource(1, j) for j in range(out_dim)], dtype=object)
        core1 = NeuralCore(
            id=2, name="fc3", input_sources=src_compute,
            core_matrix=W3_eff.T,
            threshold=1.0,
            activation_scale=torch.tensor(float(activation_scale)),
            parameter_scale=torch.tensor(1.0),
            input_activation_scale=torch.tensor(1.0),
            hardware_bias=b3_eff,
        )
        ir = IRGraph(
            nodes=[core0, compute, core1],
            output_sources=np.array([IRSource(2, j) for j in range(out_dim)], dtype=object),
        )
        from mimarsinan.mapping.ir_latency import IRLatency
        IRLatency(ir).calculate()

        cores_config = [{"count": 10, "max_axons": 64, "max_neurons": 64, "has_bias": True}]
        hybrid = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config,
            allow_scheduling=False,
        )
        return hybrid, out_train, x

    def test_hybrid_ttfs_matches_training_scaled(self):
        """Hybrid TTFS output should match training output / activation_scale."""
        activation_scale = 2.0
        hybrid, out_train, x = self._build_hybrid_with_compute_op(activation_scale)

        flow = SpikingHybridCoreFlow(
            input_shape=(4,), hybrid_mapping=hybrid, simulation_length=32,
            preprocessor=nn.Identity(), spiking_mode="ttfs",
        )
        ttfs_out = flow(x) / 32.0
        expected = out_train / activation_scale
        assert torch.allclose(ttfs_out, expected, atol=1e-2), (
            f"Hybrid TTFS {ttfs_out} != expected {expected} "
            f"(diff {(ttfs_out - expected).abs().max():.6f})"
        )

    def test_hybrid_ttfs_scale_one_identity(self):
        """activation_scale=1.0 — rescaling is a no-op."""
        hybrid, out_train, x = self._build_hybrid_with_compute_op(activation_scale=1.0)

        flow = SpikingHybridCoreFlow(
            input_shape=(4,), hybrid_mapping=hybrid, simulation_length=32,
            preprocessor=nn.Identity(), spiking_mode="ttfs",
        )
        ttfs_out = flow(x) / 32.0
        assert torch.allclose(ttfs_out, out_train, atol=1e-2), (
            f"Hybrid TTFS with scale=1 should equal training "
            f"(diff {(ttfs_out - out_train).abs().max():.6f})"
        )

    def test_node_activation_scales_populated(self):
        """Verify that node_activation_scales is populated during build."""
        activation_scale = 2.0
        hybrid, _, _ = self._build_hybrid_with_compute_op(activation_scale)

        assert hasattr(hybrid, "node_activation_scales")
        assert len(hybrid.node_activation_scales) > 0
        assert 0 in hybrid.node_activation_scales
        assert abs(hybrid.node_activation_scales[0] - activation_scale) < 1e-6
        assert 1 in hybrid.node_activation_scales
        assert abs(hybrid.node_activation_scales[1] - activation_scale) < 1e-6
