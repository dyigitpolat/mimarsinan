"""Tests for ``snapshot_ir_graph``'s resource-sharing contract.

The Hard Core Mapping step embeds an IR-graph summary so its inspector can
show soft-core detail. Re-rendering ~1000+ heatmaps a second time is both
wasteful and was empirically responsible for missing PNGs on disk: the
snapshot executor's ``wait_idle`` budget at process exit (~30 s) was being
exhausted before the duplicated renders flushed.

The fix: ``snapshot_ir_graph(source_step_name=...)`` returns an empty
descriptor list and tags every resource ref with ``"step": source_step_name``,
so the frontend points at the SCM step's already-persisted resources.
"""

from __future__ import annotations

import numpy as np

from mimarsinan.gui.snapshot.builders import (
    RESOURCE_KIND_IR_BANK_HEATMAP,
    RESOURCE_KIND_IR_CORE_HEATMAP,
    RESOURCE_KIND_IR_CORE_PRE_PRUNING,
    snapshot_ir_graph,
)
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.ir_pruning import prune_ir_graph


def _make_source_array(specs):
    return np.array(
        [IRSource(node_id=nid, index=idx) for nid, idx in specs], dtype=object
    )


def _build_graph_with_pre_pruning() -> IRGraph:
    """Two-node graph: an owned core and a bank-backed core, both with
    pre-pruning metadata stored so both `ir_core_heatmap` and
    `ir_core_pre_pruning` resources are advertised for the owned node, and
    `ir_bank_heatmap` is advertised for the bank.
    """
    w = np.array(
        [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 0.0, 4.0]], dtype=np.float32
    )
    src = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
    owned = NeuralCore(
        id=1, name="owned", input_sources=src, core_matrix=w,
        threshold=1.0, latency=0,
    )

    bank = WeightBank(id=0, core_matrix=np.ones((4, 4), dtype=np.float64))
    bank_src = _make_source_array([(-2, 0), (-2, 1), (-2, 2), (-3, 0)])
    banked = NeuralCore(
        id=2, name="banked", input_sources=bank_src, core_matrix=None,
        threshold=1.0, latency=0,
        weight_bank_id=0, weight_row_slice=(0, 2),
    )

    out_src = _make_source_array([(1, 0), (1, 2), (2, 0)])
    graph = IRGraph(
        nodes=[owned, banked], output_sources=out_src, weight_banks={0: bank},
    )
    return prune_ir_graph(graph, store_heatmap=True)


class TestSnapshotIRGraphSourceStep:
    def test_default_registers_descriptors_and_omits_step_field(self):
        """Default behavior (no source_step_name): each resource ref has
        no ``step`` field and the corresponding descriptor is registered."""
        graph = _build_graph_with_pre_pruning()
        snap, descriptors = snapshot_ir_graph(graph)

        kinds = {d.kind for d in descriptors}
        assert RESOURCE_KIND_IR_CORE_HEATMAP in kinds
        assert RESOURCE_KIND_IR_CORE_PRE_PRUNING in kinds
        assert RESOURCE_KIND_IR_BANK_HEATMAP in kinds

        for n in snap["nodes"]:
            for key in ("heatmap_resource", "pre_pruning_resource"):
                ref = n.get(key)
                if ref is not None:
                    assert "step" not in ref, (
                        f"default snapshot must not tag {key} with a step name"
                    )
        for bank in snap.get("weight_banks", {}).values():
            ref = bank.get("heatmap_resource")
            assert ref is not None
            assert "step" not in ref

    def test_source_step_tags_resources_and_skips_descriptors(self):
        """When ``source_step_name`` is provided, every advertised resource
        ref must carry that step name and **no** descriptors are returned —
        the source step has already registered them."""
        graph = _build_graph_with_pre_pruning()
        snap, descriptors = snapshot_ir_graph(
            graph, source_step_name="Soft Core Mapping",
        )

        # Same summary structure as default — only the ref tagging changes.
        assert len(snap["nodes"]) == 2
        owned = next(n for n in snap["nodes"] if n["id"] == 1)
        banked = next(n for n in snap["nodes"] if n["id"] == 2)

        assert owned.get("has_heatmap") is True
        assert owned["heatmap_resource"]["step"] == "Soft Core Mapping"
        assert owned["heatmap_resource"]["kind"] == RESOURCE_KIND_IR_CORE_HEATMAP
        assert owned["heatmap_resource"]["rid"] == "core/1"

        assert owned.get("has_pre_pruning") is True
        assert owned["pre_pruning_resource"]["step"] == "Soft Core Mapping"
        assert owned["pre_pruning_resource"]["kind"] == RESOURCE_KIND_IR_CORE_PRE_PRUNING
        assert owned["pre_pruning_resource"]["rid"] == "core/1"

        assert banked.get("has_heatmap") is True
        assert banked["heatmap_resource"]["step"] == "Soft Core Mapping"

        for bank in snap.get("weight_banks", {}).values():
            ref = bank["heatmap_resource"]
            assert ref["step"] == "Soft Core Mapping"
            assert ref["kind"] == RESOURCE_KIND_IR_BANK_HEATMAP

        # Critical: do not duplicate the SCM-step renders.
        assert descriptors == []
