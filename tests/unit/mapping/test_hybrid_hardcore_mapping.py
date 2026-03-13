"""Tests for HybridHardCoreMapping: building from IR graph and stage layout."""

import pytest
import numpy as np
import torch

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, ComputeOp
from mimarsinan.mapping.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
    HybridHardCoreMapping,
)
from mimarsinan.mapping.softcore_mapping import compact_soft_core_mapping


def _make_two_core_ir():
    """Neural-only IR graph: 4 inputs -> core0 (hidden=4) -> core1 (out=2)."""
    w1 = np.ones((5, 4), dtype=np.float32) * 0.1
    s1 = np.array([IRSource(-2, i) for i in range(4)] + [IRSource(-3, 0)], dtype=object)
    c1 = NeuralCore(id=0, name="h", input_sources=s1, core_matrix=w1, latency=0)

    w2 = np.ones((5, 2), dtype=np.float32) * 0.1
    s2 = np.array([IRSource(0, i) for i in range(4)] + [IRSource(-3, 0)], dtype=object)
    c2 = NeuralCore(id=1, name="o", input_sources=s2, core_matrix=w2, latency=1)

    out = np.array([IRSource(1, 0), IRSource(1, 1)], dtype=object)
    return IRGraph(nodes=[c1, c2], output_sources=out)


def _make_ir_with_compute_op():
    """IR graph: core0 -> flatten ComputeOp -> core1."""
    w1 = np.ones((3, 2), dtype=np.float32)
    s1 = np.array([IRSource(-2, 0), IRSource(-2, 1), IRSource(-3, 0)], dtype=object)
    c1 = NeuralCore(id=0, name="c1", input_sources=s1, core_matrix=w1, latency=0)

    op_src = np.array([IRSource(0, 0), IRSource(0, 1)], dtype=object)
    op = ComputeOp(id=1, name="flat", input_sources=op_src,
                   op_type="identity", input_shape=(2,), output_shape=(2,))

    w2 = np.ones((3, 2), dtype=np.float32)
    s2 = np.array([IRSource(1, 0), IRSource(1, 1), IRSource(-3, 0)], dtype=object)
    c2 = NeuralCore(id=2, name="c2", input_sources=s2, core_matrix=w2, latency=2)

    out = np.array([IRSource(2, 0), IRSource(2, 1)], dtype=object)
    return IRGraph(nodes=[c1, op, c2], output_sources=out)


class TestBuildHybridHardCoreMapping:
    def test_neural_only(self):
        ir = _make_two_core_ir()
        cores_config = [{"max_axons": 32, "max_neurons": 32, "count": 10}]
        hm = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores_config)

        assert isinstance(hm, HybridHardCoreMapping)
        assert len(hm.stages) >= 1
        neural_segs = hm.get_neural_segments()
        assert len(neural_segs) >= 1

    def test_with_compute_op(self):
        ir = _make_ir_with_compute_op()
        cores_config = [{"max_axons": 32, "max_neurons": 32, "count": 10}]
        hm = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores_config)

        neural_segs = hm.get_neural_segments()
        compute_ops = hm.get_compute_ops()
        assert len(neural_segs) == 2
        assert len(compute_ops) == 1
        assert compute_ops[0].op_type == "identity"

    def test_stage_order_preserved(self):
        ir = _make_ir_with_compute_op()
        cores_config = [{"max_axons": 32, "max_neurons": 32, "count": 10}]
        hm = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores_config)

        kinds = [s.kind for s in hm.stages]
        assert kinds == ["neural", "compute", "neural"]

    def test_insufficient_hardware_cores_raises(self):
        ir = _make_two_core_ir()
        cores_config = [{"max_axons": 2, "max_neurons": 1, "count": 1}]
        with pytest.raises(RuntimeError):
            build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores_config)

    def test_empty_ir_graph_raises(self):
        ir = IRGraph(nodes=[], output_sources=np.array([], dtype=object))
        cores_config = [{"max_axons": 32, "max_neurons": 32, "count": 10}]
        with pytest.raises(ValueError, match="no stages"):
            build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores_config)

    def test_heterogeneous_core_types(self):
        ir = _make_two_core_ir()
        cores_config = [
            {"max_axons": 8, "max_neurons": 8, "count": 5},
            {"max_axons": 32, "max_neurons": 32, "count": 2},
        ]
        hm = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores_config)
        assert len(hm.get_neural_segments()) >= 1

    def test_has_bias_produces_bias_capable_hard_cores(self):
        """When cores_config has has_bias=true, allocated HardCores have has_bias_capability=True."""
        ir = _make_two_core_ir()
        cores_config = [{"max_axons": 32, "max_neurons": 32, "count": 10, "has_bias": True}]
        hm = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores_config)
        neural_segs = hm.get_neural_segments()
        assert len(neural_segs) >= 1
        for seg in neural_segs:
            for hc in seg.cores:
                assert hc.has_bias_capability is True

    def test_has_bias_omitted_defaults_true(self):
        """When has_bias is omitted from cores_config, HardCores have has_bias_capability=True."""
        ir = _make_two_core_ir()
        cores_config = [{"max_axons": 32, "max_neurons": 32, "count": 10}]
        hm = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores_config)
        neural_segs = hm.get_neural_segments()
        assert len(neural_segs) >= 1
        for seg in neural_segs:
            for hc in seg.cores:
                assert getattr(hc, "has_bias_capability", True) is True

    def test_fused_hard_core_has_fused_component_axons(self):
        """When packer fuses multiple physical cores, the fused HardCore has fused_component_axons set."""
        # One soft core with 100 axons (needs 2×64 physical cores fused)
        w = np.ones((101, 32), dtype=np.float32) * 0.1  # 100 axons + bias, 32 neurons
        s = np.array([IRSource(-2, i) for i in range(100)] + [IRSource(-3, 0)], dtype=object)
        c = NeuralCore(id=0, name="large", input_sources=s, core_matrix=w, latency=0)
        out = np.array([IRSource(0, i) for i in range(32)], dtype=object)
        ir = IRGraph(nodes=[c], output_sources=out)
        cores_config = [{"max_axons": 64, "max_neurons": 32, "count": 4}]
        hm = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores_config)
        neural_segs = hm.get_neural_segments()
        assert len(neural_segs) == 1
        assert len(neural_segs[0].cores) == 1
        fused_hc = neural_segs[0].cores[0]
        assert getattr(fused_hc, "fused_component_axons", None) is not None
        assert fused_hc.fused_component_axons == [64, 64]


class TestCompactSoftCoreMapping:
    """Compaction uses pruning maps (pruned_row_mask, pruned_col_mask), not parameter values."""

    def test_compacts_using_pruning_maps(self):
        # Pruning maps: row 1 and col 2 pruned (True = pruned)
        pruned_row_mask = [False, True, False, False]
        pruned_col_mask = [False, False, True, False]
        mat = np.ones((4, 4), dtype=np.float64)
        from mimarsinan.code_generation.cpp_chip_model import SpikeSource
        from mimarsinan.mapping.softcore_mapping import SoftCore

        sources = [
            SpikeSource(-2, 0), SpikeSource(-2, 1), SpikeSource(-2, 2), SpikeSource(-2, 3),
        ]
        core = SoftCore(core_matrix=mat.copy(), axon_sources=sources, id=0)
        core.pruned_row_mask = pruned_row_mask
        core.pruned_col_mask = pruned_col_mask
        cores = [core]
        output_sources = [
            SpikeSource(0, 0), SpikeSource(0, 1), SpikeSource(0, 2), SpikeSource(0, 3),
        ]
        compact_soft_core_mapping(cores, output_sources)

        assert cores[0].core_matrix.shape == (3, 3)
        assert len(cores[0].axon_sources) == 3
        assert len(output_sources) == 3

    def test_no_compaction_when_masks_absent(self):
        mat = np.ones((2, 2), dtype=np.float64) * 0.5
        from mimarsinan.code_generation.cpp_chip_model import SpikeSource
        from mimarsinan.mapping.softcore_mapping import SoftCore

        sources = [SpikeSource(-2, 0), SpikeSource(-2, 1)]
        core = SoftCore(core_matrix=mat.copy(), axon_sources=sources, id=0)
        cores = [core]
        output_sources = [SpikeSource(0, 0), SpikeSource(0, 1)]
        compact_soft_core_mapping(cores, output_sources)

        assert cores[0].core_matrix.shape == (2, 2)
        assert len(output_sources) == 2
