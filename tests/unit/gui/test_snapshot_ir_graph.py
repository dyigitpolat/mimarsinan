"""Tests for ``snapshot_ir_graph``'s resource-sharing and liveness contract.

The Hard Core Mapping step embeds an IR-graph summary so its inspector can
show soft-core detail. Re-rendering ~1000+ heatmaps a second time is both
wasteful and was empirically responsible for missing PNGs on disk: the
snapshot executor's ``wait_idle`` budget at process exit (~30 s) was being
exhausted before the duplicated renders flushed.

The fix: ``snapshot_ir_graph(source_step_name=...)`` returns an empty
descriptor list and tags every resource ref with ``"step": source_step_name``,
so the frontend points at the SCM step's already-persisted resources.

After the dead-core-elimination refactor, ``snapshot_ir_graph`` also emits
per-NeuralCore ``liveness`` ("live" / "bias_only" / "dead_legacy") plus
per-group aggregates (``all_dead``, ``all_dead_or_bias_only``,
``live_core_count``, ``bias_only_count``, ``dead_count``) so the monitor UI
can tint dead/bias-only groups.
"""

from __future__ import annotations

import numpy as np

from mimarsinan.gui.snapshot.builders import (
    LIVENESS_BIAS_ONLY,
    LIVENESS_DEAD_LEGACY,
    LIVENESS_LIVE,
    RESOURCE_KIND_IR_BANK_HEATMAP,
    RESOURCE_KIND_IR_CORE_BIAS,
    RESOURCE_KIND_IR_CORE_HEATMAP,
    RESOURCE_KIND_IR_CORE_PRE_PRUNING,
    snapshot_ir_graph,
)
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.pruning.ir_pruning import prune_ir_graph


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


def _live_neural_core(*, nid: int, name: str = "live") -> NeuralCore:
    src = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
    return NeuralCore(
        id=nid, name=name, input_sources=src,
        core_matrix=np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64
        ),
        threshold=1.0, latency=0,
    )


def _bias_only_neural_core(*, nid: int, name: str = "bias_only") -> NeuralCore:
    """Build a post-compaction BIAS_ONLY core: (1, N) zero matrix with a
    single OFF-source axon and a non-zero ``hardware_bias``."""
    core = NeuralCore(
        id=nid, name=name,
        input_sources=_make_source_array([(-1, 0)]),
        core_matrix=np.zeros((1, 2), dtype=np.float64),
        hardware_bias=np.array([2.0, 1.5], dtype=np.float64),
        threshold=1.0, latency=0,
    )
    # Pre-pruning masks reflecting the BIAS_ONLY history: every original
    # axon was pruned, all columns alive.
    core.pre_pruning_row_mask = [True, True, True]
    core.pre_pruning_col_mask = [False, False]
    core.pruned_row_mask = [True]
    core.pruned_col_mask = [False, False]
    return core


def _dead_legacy_neural_core(*, nid: int, name: str = "legacy") -> NeuralCore:
    """Build the old (1,1) DEAD placeholder pattern that older pickles
    still contain. Detection must surface ``dead_legacy``."""
    core = NeuralCore(
        id=nid, name=name,
        input_sources=_make_source_array([(-1, 0)]),
        core_matrix=np.zeros((1, 1), dtype=np.float64),
        threshold=1.0, latency=0,
    )
    core.pre_pruning_row_mask = [True, True]
    core.pre_pruning_col_mask = [True, True]
    core.pruned_row_mask = [True]
    core.pruned_col_mask = [True]
    return core


class TestSnapshotIRGraphLivenessFields:
    def test_neural_core_emits_liveness_field(self):
        live = _live_neural_core(nid=1)
        graph = IRGraph(
            nodes=[live],
            output_sources=_make_source_array([(1, 0), (1, 1)]),
        )
        snap, _ = snapshot_ir_graph(graph)
        info = snap["nodes"][0]
        assert info["id"] == 1
        assert info["liveness"] == LIVENESS_LIVE

    def test_bias_only_core_emits_bias_resource(self):
        bias_only = _bias_only_neural_core(nid=2)
        # Add a tiny live consumer so the graph has a sane output target.
        live = NeuralCore(
            id=3, name="consumer",
            input_sources=_make_source_array([(2, 0), (2, 1)]),
            core_matrix=np.array([[1.0], [1.0]], dtype=np.float64),
            threshold=1.0, latency=1,
        )
        graph = IRGraph(
            nodes=[bias_only, live],
            output_sources=_make_source_array([(3, 0)]),
        )
        snap, descriptors = snapshot_ir_graph(graph)

        bias_info = next(n for n in snap["nodes"] if n["id"] == 2)
        assert bias_info["liveness"] == LIVENESS_BIAS_ONLY
        assert bias_info.get("has_bias_resource") is True
        assert (
            bias_info["bias_resource"]["kind"] == RESOURCE_KIND_IR_CORE_BIAS
        )
        assert bias_info["bias_resource"]["rid"] == "core/2"
        assert any(
            d.kind == RESOURCE_KIND_IR_CORE_BIAS and d.rid == "core/2"
            for d in descriptors
        )
        # Live consumer has no bias resource.
        live_info = next(n for n in snap["nodes"] if n["id"] == 3)
        assert live_info["liveness"] == LIVENESS_LIVE
        assert live_info.get("has_bias_resource") is not True

    def test_legacy_placeholder_pickle_still_classified_as_dead_legacy(self):
        legacy = _dead_legacy_neural_core(nid=4)
        live = NeuralCore(
            id=5, name="live",
            input_sources=_make_source_array([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        graph = IRGraph(
            nodes=[legacy, live],
            output_sources=_make_source_array([(5, 0)]),
        )
        snap, _ = snapshot_ir_graph(graph)
        info = next(n for n in snap["nodes"] if n["id"] == 4)
        assert info["liveness"] == LIVENESS_DEAD_LEGACY

    def test_group_all_dead_flag_set_when_every_core_dead_legacy(self):
        # Two legacy cores in the same FC-tile-style layer group.
        legacy_a = _dead_legacy_neural_core(nid=10, name="layer_X_tile_0_0")
        legacy_b = _dead_legacy_neural_core(nid=11, name="layer_X_tile_0_1")
        # Live core elsewhere so the graph stays well-formed.
        live = NeuralCore(
            id=12, name="layer_Y_tile_0_0",
            input_sources=_make_source_array([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        graph = IRGraph(
            nodes=[legacy_a, legacy_b, live],
            output_sources=_make_source_array([(12, 0)]),
        )
        snap, _ = snapshot_ir_graph(graph)
        groups = {g["key"]: g for g in snap["groups"]}
        x_group = next(g for k, g in groups.items() if "layer_X" in k)
        assert x_group["all_dead"] is True
        assert x_group["all_dead_or_bias_only"] is True
        assert x_group["dead_count"] == 2
        assert x_group["live_core_count"] == 0

        y_group = next(g for k, g in groups.items() if "layer_Y" in k)
        assert y_group["all_dead"] is False
        assert y_group["all_dead_or_bias_only"] is False
        assert y_group["live_core_count"] == 1

    def test_group_all_dead_or_bias_only_flag_set_for_mixed(self):
        bias_a = _bias_only_neural_core(nid=20, name="layer_M_tile_0_0")
        legacy_b = _dead_legacy_neural_core(nid=21, name="layer_M_tile_0_1")
        live = NeuralCore(
            id=22, name="layer_N_tile_0_0",
            input_sources=_make_source_array([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        graph = IRGraph(
            nodes=[bias_a, legacy_b, live],
            output_sources=_make_source_array([(22, 0)]),
        )
        snap, _ = snapshot_ir_graph(graph)
        groups = {g["key"]: g for g in snap["groups"]}
        m_group = next(g for k, g in groups.items() if "layer_M" in k)
        assert m_group["all_dead"] is False
        assert m_group["all_dead_or_bias_only"] is True
        assert m_group["dead_count"] == 1
        assert m_group["bias_only_count"] == 1
        assert m_group["live_core_count"] == 0
