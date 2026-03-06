"""Tests for hard–soft core two-way traceability (soft_core_placements_per_hard_core)."""

import pytest
import numpy as np

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, ir_graph_to_soft_core_mapping
from mimarsinan.mapping.softcore_mapping import HardCore, HardCoreMapping


def _make_two_core_ir():
    """Neural-only IR: 4 inputs -> core0 (4 neurons) -> core1 (2 neurons)."""
    w1 = np.ones((5, 4), dtype=np.float32) * 0.1
    s1 = np.array([IRSource(-2, i) for i in range(4)] + [IRSource(-3, 0)], dtype=object)
    c1 = NeuralCore(id=0, name="h", input_sources=s1, core_matrix=w1, latency=0)

    w2 = np.ones((5, 2), dtype=np.float32) * 0.1
    s2 = np.array([IRSource(0, i) for i in range(4)] + [IRSource(-3, 0)], dtype=object)
    c2 = NeuralCore(id=1, name="o", input_sources=s2, core_matrix=w2, latency=0)

    out = np.array([IRSource(1, 0), IRSource(1, 1)], dtype=object)
    return IRGraph(nodes=[c1, c2], output_sources=out)


class TestSoftCorePlacementsPerHardCore:
    """HardCoreMapping.soft_core_placements_per_hard_core after map()."""

    def test_placements_populated_after_map(self):
        ir = _make_two_core_ir()
        soft = ir_graph_to_soft_core_mapping(ir)
        pool = [HardCore(32, 32), HardCore(32, 32)]
        hard = HardCoreMapping(pool)
        hard.map(soft)

        assert hasattr(hard, "soft_core_placements_per_hard_core")
        assert len(hard.soft_core_placements_per_hard_core) == len(hard.cores)

        all_ids = set()
        for placements in hard.soft_core_placements_per_hard_core:
            for pl in placements:
                assert "ir_node_id" in pl
                assert "axon_offset" in pl
                assert "neuron_offset" in pl
                assert "axons" in pl
                assert "neurons" in pl
                all_ids.add(pl["ir_node_id"])

        assert all_ids == {0, 1}, "All soft core ids (IR node ids) should appear in placements"

    def test_placement_geometry_matches_soft_dims(self):
        ir = _make_two_core_ir()  # both cores latency=0 so they can pack together
        soft = ir_graph_to_soft_core_mapping(ir)
        pool = [HardCore(32, 32)]
        hard = HardCoreMapping(pool)
        hard.map(soft)

        # Both soft cores packed into one hard core (same latency)
        assert len(hard.cores) == 1
        placements = hard.soft_core_placements_per_hard_core[0]
        assert len(placements) == 2

        by_id = {p["ir_node_id"]: p for p in placements}
        # core0: 5 axons, 4 neurons; core1: 5 axons, 2 neurons
        assert by_id[0]["axons"] == 5 and by_id[0]["neurons"] == 4
        assert by_id[1]["axons"] == 5 and by_id[1]["neurons"] == 2


class TestSnapshotMappedPlacements:
    """snapshot_hard_core_mapping includes mapped_placements per core."""

    def test_snapshot_includes_mapped_placements(self):
        from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
        from mimarsinan.gui.snapshot import snapshot_hard_core_mapping

        ir = _make_two_core_ir()
        cores_config = [{"max_axons": 32, "max_neurons": 32, "count": 5}]
        hm = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores_config)
        snap = snapshot_hard_core_mapping(hm)

        neural_stages = [s for s in snap["stages"] if s.get("kind") == "neural" and s.get("cores")]
        assert len(neural_stages) >= 1
        stage = neural_stages[0]
        for core in stage["cores"]:
            assert "mapped_placements" in core
            for pl in core["mapped_placements"]:
                assert "ir_node_id" in pl
                assert "axon_offset" in pl
                assert "neuron_offset" in pl
                assert "axons" in pl
                assert "neurons" in pl

    def test_snapshot_placements_match_mapping(self):
        from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
        from mimarsinan.gui.snapshot import snapshot_hard_core_mapping

        ir = _make_two_core_ir()
        cores_config = [{"max_axons": 32, "max_neurons": 32, "count": 5}]
        hm = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores_config)
        snap = snapshot_hard_core_mapping(hm)

        neural_stages = [s for s in snap["stages"] if s.get("kind") == "neural" and s.get("cores")]
        assert len(neural_stages) >= 1
        stage = neural_stages[0]
        hcm = hm.get_neural_segments()[0]
        for ci, core in enumerate(stage["cores"]):
            expected = hcm.soft_core_placements_per_hard_core[ci]
            assert core["mapped_placements"] == expected

    def test_snapshot_raises_when_traceability_missing(self):
        """Snapshot hard-fails with clear error when soft_core_placements_per_hard_core is missing."""
        from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
        from mimarsinan.gui.snapshot import snapshot_hard_core_mapping

        ir = _make_two_core_ir()
        cores_config = [{"max_axons": 32, "max_neurons": 32, "count": 5}]
        hm = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores_config)
        hcm = hm.get_neural_segments()[0]
        del hcm.soft_core_placements_per_hard_core

        with pytest.raises(ValueError, match="missing soft-core traceability"):
            snapshot_hard_core_mapping(hm)
