"""Tests for scheduled mapping: correctness, equivalence, and edge cases.

Verifies:
1. Scheduled mapping produces identical results to non-scheduled when all cores fit.
2. Multi-pass mapping succeeds when cores exceed hardware capacity.
3. Reindex maps are applied exactly once (no double-application corruption).
4. Pruned IR graphs don't lose connections through scheduling.
5. ComputeOp segments are preserved across scheduling.
6. Recursive split on packing failure produces valid stages.
7. SpikingHybridCoreFlow produces correct output with scheduled stages.
"""

import copy
import pytest
import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, ComputeOp
from mimarsinan.mapping.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
    HybridHardCoreMapping,
)
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow


# ---------------------------------------------------------------------------
# IR graph builders
# ---------------------------------------------------------------------------

def _make_chain_ir(n_layers=2, width=4, n_in=4, n_out=2, pruning=False):
    """Build a chain of NeuralCores: input -> h0 -> h1 -> ... -> output.

    When pruning=True, sets pruned_row_mask/pruned_col_mask on each core
    to prune ~25% of neurons (simulating post-pruning IR).
    """
    nodes = []
    prev_width = n_in

    for i in range(n_layers):
        is_last = (i == n_layers - 1)
        cur_width = n_out if is_last else width

        # input_sources: reference previous layer (or network input for layer 0)
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

        if pruning and cur_width > 2 and not is_last:
            # Prune every 4th column and every 4th row (non-bias).
            # Skip the last (output) layer to avoid legitimate off-sources
            # in the network's output_sources.
            col_mask = [((j % 4) == 3) for j in range(cur_width)]
            row_mask = [((j % 4) == 3) for j in range(prev_width)] + [False]  # never prune bias row
            core.pruned_row_mask = row_mask
            core.pruned_col_mask = col_mask

        nodes.append(core)
        prev_width = cur_width

    out_sources = np.array(
        [IRSource(nodes[-1].id, j) for j in range(n_out)], dtype=object
    )
    return IRGraph(nodes=nodes, output_sources=out_sources)


def _make_many_core_ir(n_cores=10, n_in=4, n_out=2, width=4):
    """Build a wide parallel IR graph: many independent cores at latency 0,
    then a final core that consumes all of them.

    Total cores = n_cores + 1 (parallel layer + output layer).
    """
    nodes = []
    # Parallel layer: n_cores independent cores, all reading from input
    for i in range(n_cores):
        sources = [IRSource(-2, j) for j in range(n_in)] + [IRSource(-3, 0)]
        w = np.random.randn(n_in + 1, width).astype(np.float32) * 0.1
        core = NeuralCore(
            id=i, name=f"par{i}",
            input_sources=np.array(sources, dtype=object),
            core_matrix=w, latency=0,
        )
        nodes.append(core)

    # Output core: reads from all parallel cores
    out_sources_list = []
    for i in range(n_cores):
        for j in range(width):
            out_sources_list.append(IRSource(i, j))
    out_sources_list.append(IRSource(-3, 0))  # bias

    total_in = n_cores * width + 1
    w_out = np.random.randn(total_in, n_out).astype(np.float32) * 0.1
    out_core = NeuralCore(
        id=n_cores, name="output",
        input_sources=np.array(out_sources_list, dtype=object),
        core_matrix=w_out, latency=1,
    )
    nodes.append(out_core)

    out = np.array([IRSource(n_cores, j) for j in range(n_out)], dtype=object)
    return IRGraph(nodes=nodes, output_sources=out)


def _make_ir_with_compute_op_many_cores(cores_per_segment=5, n_in=4, n_out=2, width=4):
    """IR graph: [cores_per_segment cores] -> ComputeOp -> [cores_per_segment cores] -> output."""
    nodes = []
    node_id = 0

    # Segment 0: chain of cores
    prev_width = n_in
    for i in range(cores_per_segment):
        if i == 0:
            sources = [IRSource(-2, j) for j in range(prev_width)] + [IRSource(-3, 0)]
        else:
            sources = [IRSource(node_id - 1, j) for j in range(prev_width)] + [IRSource(-3, 0)]
        w = np.random.randn(prev_width + 1, width).astype(np.float32) * 0.1
        core = NeuralCore(
            id=node_id, name=f"seg0_l{i}",
            input_sources=np.array(sources, dtype=object),
            core_matrix=w, latency=i,
        )
        nodes.append(core)
        prev_width = width
        node_id += 1

    # ComputeOp (identity)
    last_seg0_id = node_id - 1
    op_sources = np.array(
        [IRSource(last_seg0_id, j) for j in range(width)], dtype=object
    )
    op = ComputeOp(
        id=node_id, name="barrier",
        input_sources=op_sources,
        op_type="identity",
        input_shape=(width,), output_shape=(width,),
    )
    nodes.append(op)
    op_id = node_id
    node_id += 1

    # Segment 1: chain of cores reading from ComputeOp output
    prev_width = width
    for i in range(cores_per_segment):
        is_last = (i == cores_per_segment - 1)
        cur_width = n_out if is_last else width
        if i == 0:
            sources = [IRSource(op_id, j) for j in range(prev_width)] + [IRSource(-3, 0)]
        else:
            sources = [IRSource(node_id - 1, j) for j in range(prev_width)] + [IRSource(-3, 0)]
        w = np.random.randn(prev_width + 1, cur_width).astype(np.float32) * 0.1
        core = NeuralCore(
            id=node_id, name=f"seg1_l{i}",
            input_sources=np.array(sources, dtype=object),
            core_matrix=w, latency=cores_per_segment + 1 + i,
        )
        nodes.append(core)
        prev_width = cur_width
        node_id += 1

    out = np.array([IRSource(node_id - 1, j) for j in range(n_out)], dtype=object)
    return IRGraph(nodes=nodes, output_sources=out)


def _make_flow(hm, input_shape, spiking_mode="ttfs", simulation_length=8):
    """Create a SpikingHybridCoreFlow from a HybridHardCoreMapping."""
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
# Test: Equivalence — scheduled vs non-scheduled
# ---------------------------------------------------------------------------

class TestScheduledEquivalence:
    """Single-pass scheduled mapping must be identical to non-scheduled."""

    def test_equivalence_no_pruning(self):
        """Without pruning, scheduled and non-scheduled should produce identical output."""
        np.random.seed(42)
        ir = _make_chain_ir(n_layers=3, width=8, n_in=6, n_out=4, pruning=False)
        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 20}]

        hm_normal = build_hybrid_hard_core_mapping(
            ir_graph=copy.deepcopy(ir), cores_config=cores_config)
        hm_sched = build_hybrid_hard_core_mapping(
            ir_graph=copy.deepcopy(ir), cores_config=cores_config, allow_scheduling=True)

        torch.manual_seed(0)
        x = torch.randn(4, 6)

        flow_normal = _make_flow(hm_normal, (6,))
        flow_sched = _make_flow(hm_sched, (6,))

        out_normal = flow_normal.forward(x)
        out_sched = flow_sched.forward(x)

        assert torch.allclose(out_normal, out_sched, atol=1e-5), (
            f"Scheduled output diverged from non-scheduled:\n"
            f"  max diff = {(out_normal - out_sched).abs().max().item()}"
        )

    def test_equivalence_with_pruning(self):
        """With pruning, scheduled and non-scheduled should produce identical output.

        This is the exact scenario that triggers the double-reindex bug.
        """
        np.random.seed(42)
        ir = _make_chain_ir(n_layers=3, width=8, n_in=6, n_out=4, pruning=True)
        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 20}]

        hm_normal = build_hybrid_hard_core_mapping(
            ir_graph=copy.deepcopy(ir), cores_config=cores_config)
        hm_sched = build_hybrid_hard_core_mapping(
            ir_graph=copy.deepcopy(ir), cores_config=cores_config, allow_scheduling=True)

        torch.manual_seed(0)
        x = torch.randn(4, 6)

        flow_normal = _make_flow(hm_normal, (6,))
        flow_sched = _make_flow(hm_sched, (6,))

        out_normal = flow_normal.forward(x)
        out_sched = flow_sched.forward(x)

        assert torch.allclose(out_normal, out_sched, atol=1e-5), (
            f"Pruned scheduled output diverged from non-scheduled:\n"
            f"  max diff = {(out_normal - out_sched).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Test: Multi-pass mapping
# ---------------------------------------------------------------------------

class TestMultiPassMapping:
    """Scheduling must succeed when cores exceed hardware capacity."""

    def test_multi_pass_builds(self):
        """IR with more cores than hardware should produce multiple neural stages."""
        np.random.seed(42)
        ir = _make_many_core_ir(n_cores=10, n_in=4, n_out=2, width=4)
        # Only 3 hardware cores — far fewer than the 11 needed
        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 3}]

        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True)

        neural_stages = [s for s in hm.stages if s.kind == "neural"]
        assert len(neural_stages) > 1, (
            f"Expected multiple neural stages for 11 cores on 3 hw cores, got {len(neural_stages)}"
        )
        for s in neural_stages:
            assert s.schedule_segment_index is not None
            assert s.schedule_pass_index is not None

    def test_multi_pass_output_non_zero(self):
        """Multi-pass output must be finite and non-zero."""
        np.random.seed(42)
        ir = _make_many_core_ir(n_cores=8, n_in=4, n_out=2, width=4)
        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 3}]

        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True)

        flow = _make_flow(hm, (4,))
        torch.manual_seed(0)
        x = torch.rand(2, 4)  # positive inputs for TTFS
        out = flow.forward(x)

        assert torch.isfinite(out).all(), "Output contains non-finite values"
        assert out.abs().sum() > 0, "Output is all zeros"

    def test_without_scheduling_raises(self):
        """Without scheduling, insufficient cores should raise RuntimeError."""
        np.random.seed(42)
        # Each core is 5×4 (4 inputs + bias), needs its own hardware core.
        # 11 softcores into 2 hardware cores of 8×8 won't fit (can pack ~4 each).
        ir = _make_many_core_ir(n_cores=10, n_in=4, n_out=2, width=4)
        cores_config = [{"max_axons": 8, "max_neurons": 8, "count": 2}]

        with pytest.raises(RuntimeError):
            build_hybrid_hard_core_mapping(
                ir_graph=ir, cores_config=cores_config, allow_scheduling=False)


# ---------------------------------------------------------------------------
# Test: Reindex correctness (no off-source corruption)
# ---------------------------------------------------------------------------

class TestReindexCorrectness:
    """Compaction reindex maps must be applied exactly once."""

    def test_no_spurious_off_sources_without_pruning(self):
        """Output sources should not become off-sources when no pruning."""
        np.random.seed(42)
        ir = _make_chain_ir(n_layers=3, width=6, n_in=4, n_out=3, pruning=False)
        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 20}]

        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True)

        for src in hm.output_sources.flatten():
            if isinstance(src, IRSource):
                assert not src.is_off(), f"Output source became off-source: {src}"

    def test_no_spurious_off_sources_with_pruning(self):
        """Output sources should not become off-sources when pruning is active.

        This directly tests the double-reindex bug: pruned reindex maps
        would corrupt valid indices on second application.
        """
        np.random.seed(42)
        ir = _make_chain_ir(n_layers=3, width=8, n_in=6, n_out=4, pruning=True)
        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 20}]

        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True)

        off_count = sum(
            1 for src in hm.output_sources.flatten()
            if isinstance(src, IRSource) and src.is_off()
        )
        assert off_count == 0, (
            f"Found {off_count} off-sources in output (expected 0 — reindex corruption)"
        )

    def test_multi_pass_no_off_sources_with_pruning(self):
        """Multi-pass + pruning: output sources must not be corrupted."""
        np.random.seed(42)
        ir = _make_many_core_ir(n_cores=6, n_in=4, n_out=2, width=4)
        # Add pruning masks to all cores
        for node in ir.nodes:
            if isinstance(node, NeuralCore):
                n_ax, n_neu = node.core_matrix.shape
                if n_neu > 2:
                    node.pruned_col_mask = [((j % 3) == 2) for j in range(n_neu)]
                    node.pruned_row_mask = [((j % 3) == 2) for j in range(n_ax - 1)] + [False]

        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 2}]

        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True)

        off_count = sum(
            1 for src in hm.output_sources.flatten()
            if isinstance(src, IRSource) and src.is_off()
        )
        assert off_count == 0, (
            f"Multi-pass + pruning: {off_count} off-sources in output"
        )


# ---------------------------------------------------------------------------
# Test: With ComputeOp segments
# ---------------------------------------------------------------------------

class TestScheduledWithComputeOps:
    """Scheduling with ComputeOp barriers between segments."""

    def test_compute_ops_preserved(self):
        """ComputeOp stages must be preserved in scheduled mapping."""
        np.random.seed(42)
        ir = _make_ir_with_compute_op_many_cores(cores_per_segment=5, n_in=4, n_out=2)
        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 3}]

        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True)

        kinds = [s.kind for s in hm.stages]
        assert "compute" in kinds, "ComputeOp stage missing from scheduled mapping"
        neural_count = sum(1 for k in kinds if k == "neural")
        assert neural_count >= 2, (
            f"Expected at least 2 neural stages (2 segments), got {neural_count}"
        )

    def test_compute_op_output_valid(self):
        """Output from scheduled mapping with ComputeOps must be finite."""
        np.random.seed(42)
        ir = _make_ir_with_compute_op_many_cores(cores_per_segment=3, n_in=4, n_out=2)
        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 10}]

        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True)

        flow = _make_flow(hm, (4,))
        torch.manual_seed(0)
        x = torch.rand(2, 4)
        out = flow.forward(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test: Recursive split
# ---------------------------------------------------------------------------

class TestRecursiveSplit:
    """When initial partition is too large for packing, recursive split handles it."""

    def test_wide_cores_split_successfully(self):
        """Cores wider than hardware should still map via scheduling + splitting."""
        np.random.seed(42)
        nodes = []
        n_in = 4
        width = 4
        # 6 independent cores at latency 0
        for i in range(6):
            sources = [IRSource(-2, j) for j in range(n_in)] + [IRSource(-3, 0)]
            w = np.random.randn(n_in + 1, width).astype(np.float32) * 0.1
            core = NeuralCore(
                id=i, name=f"c{i}",
                input_sources=np.array(sources, dtype=object),
                core_matrix=w, latency=0,
            )
            nodes.append(core)

        # Output layer reading all 6 cores
        out_sources_list = []
        for i in range(6):
            for j in range(width):
                out_sources_list.append(IRSource(i, j))
        out_sources_list.append(IRSource(-3, 0))
        w_out = np.random.randn(6 * width + 1, 2).astype(np.float32) * 0.1
        out_core = NeuralCore(
            id=6, name="out",
            input_sources=np.array(out_sources_list, dtype=object),
            core_matrix=w_out, latency=1,
        )
        nodes.append(out_core)

        out = np.array([IRSource(6, 0), IRSource(6, 1)], dtype=object)
        ir = IRGraph(nodes=nodes, output_sources=out)

        # Very constrained hardware: only 2 cores
        cores_config = [{"max_axons": 32, "max_neurons": 32, "count": 2}]

        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True)

        neural_stages = [s for s in hm.stages if s.kind == "neural"]
        assert len(neural_stages) >= 2, f"Expected multiple passes, got {len(neural_stages)}"


# ---------------------------------------------------------------------------
# Test: Stage metadata
# ---------------------------------------------------------------------------

class TestStageMetadata:
    """Schedule metadata on HybridStage must be set correctly."""

    def test_single_segment_indices(self):
        """Single-segment scheduled mapping: all stages should have segment_index=0."""
        np.random.seed(42)
        ir = _make_many_core_ir(n_cores=6, n_in=4, n_out=2, width=4)
        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 2}]

        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True)

        neural_stages = [s for s in hm.stages if s.kind == "neural"]
        for s in neural_stages:
            assert s.schedule_segment_index == 0
        # Pass indices should be sequential starting from 0
        pass_indices = [s.schedule_pass_index for s in neural_stages]
        assert pass_indices == list(range(len(neural_stages)))

    def test_non_scheduled_has_no_metadata(self):
        """Non-scheduled mapping should NOT set schedule metadata."""
        np.random.seed(42)
        ir = _make_chain_ir(n_layers=2, width=4, n_in=4, n_out=2)
        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 20}]

        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=False)

        for s in hm.stages:
            if s.kind == "neural":
                assert s.schedule_segment_index is None
                assert s.schedule_pass_index is None


# ---------------------------------------------------------------------------
# Test: IO maps
# ---------------------------------------------------------------------------

class TestIOMapIntegrity:
    """Input/output maps must be valid for each scheduled stage."""

    def test_all_stages_have_io_maps(self):
        """Every neural stage must have non-empty input_map and output_map."""
        np.random.seed(42)
        ir = _make_many_core_ir(n_cores=6, n_in=4, n_out=2, width=4)
        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 2}]

        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=cores_config, allow_scheduling=True)

        for i, s in enumerate(hm.stages):
            if s.kind == "neural":
                assert len(s.input_map) > 0, f"Stage {i} has empty input_map"
                assert len(s.output_map) > 0, f"Stage {i} has empty output_map"
