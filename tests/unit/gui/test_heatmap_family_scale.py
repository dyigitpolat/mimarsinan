"""One symmetric color scale per heatmap FAMILY per snapshot.

Per-core p98 color limits made cross-core comparison meaningless: the same
weight rendered as a different color on every tile, with no legend anywhere.
The contract pinned here:

* every descriptor-producing snapshot loop computes ONE family scale — the max
  of the per-matrix p98 symmetric scales — and stamps it on every family
  ``HeatmapSource`` (bias strips keep their own units and stay out);
* the summary carries ``heatmap_scale`` = numeric ``vmin``/``vmax`` + a
  colorbar resource ref, and ONE colorbar descriptor is emitted per family;
* the embedded (``source_step_name``) IR summary reports the SAME scale while
  registering no descriptors;
* ``HeatmapSource.scale`` survives the .mimsrc round trip, so a deferred run
  reproduces the eager pixels.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.gui.heatmap_renderer import (
    FULL_TARGET_LONG_SIDE,
    render_colorbar_png_bytes,
    render_heatmap_png_bytes,
    symmetric_scale,
)
from mimarsinan.gui.resources import HeatmapSource
from mimarsinan.gui.resources.sources import ColorbarSource, resource_source_from_state
from mimarsinan.gui.runtime.persistence import (
    load_resource_source,
    save_resource_source,
)
from mimarsinan.gui.snapshot import (
    snapshot_hard_core_mapping,
    snapshot_ir_graph,
    snapshot_pruning_layers,
)
from mimarsinan.gui.snapshot.util.constants import (
    RESOURCE_KIND_HARD_CORE_HEATMAP,
    RESOURCE_KIND_HEATMAP_COLORBAR,
    RESOURCE_KIND_IR_CORE_BIAS,
)
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph


def _sources(specs):
    return np.array([IRSource(node_id=n, index=i) for n, i in specs], dtype=object)


def _two_core_graph() -> IRGraph:
    small = NeuralCore(
        id=0, name="small", input_sources=_sources([(-2, 0), (-2, 1), (-3, 0)]),
        core_matrix=np.array([[0.1, -0.2], [0.05, 0.0], [0.15, -0.1]], dtype=np.float64),
        threshold=1.0, latency=0,
    )
    big = NeuralCore(
        id=1, name="big", input_sources=_sources([(0, 0), (0, 1), (-3, 0)]),
        core_matrix=np.array([[5.0, -3.0], [2.0, 0.0], [1.0, -4.0]], dtype=np.float64),
        threshold=1.0, latency=0,
    )
    out = _sources([(1, 0), (1, 1)])
    graph = IRGraph(nodes=[small, big], output_sources=out)
    return prune_ir_graph(graph, store_heatmap=True)


def _family_vmax(descriptors, kinds) -> float:
    scales = [
        d.source.scale
        for d in descriptors
        if d.kind in kinds and isinstance(d.source, HeatmapSource)
    ]
    assert scales, "expected family heatmap descriptors"
    assert len({s for s in scales}) == 1, "family sources must share ONE scale"
    return scales[0]


class TestHeatmapSourceScale:
    def test_render_uses_the_stamped_scale(self):
        m = np.array([[1.0, -2.0], [0.5, 4.0]])
        src = HeatmapSource(m, scale=8.0)
        assert src.render() == render_heatmap_png_bytes(m, scale=8.0)
        assert src.render() != render_heatmap_png_bytes(m)

    def test_scale_survives_the_mimsrc_round_trip(self, tmp_path):
        m = np.arange(12, dtype=np.float64).reshape(3, 4)
        src = HeatmapSource(m, scale=7.5)
        save_resource_source(str(tmp_path), "S", "ir_core_heatmap", "core/0", src)
        loaded = load_resource_source(str(tmp_path), "S", "ir_core_heatmap", "core/0")
        assert loaded is not None
        assert loaded.scale == 7.5
        assert loaded.render() == src.render()

    def test_legacy_state_without_scale_defaults_to_none(self):
        src = HeatmapSource.from_state({"matrix": np.ones((2, 2))})
        assert src.scale is None

    def test_render_full_is_near_native_resolution(self):
        from PIL import Image
        import io

        m = np.zeros((800, 600))
        src = HeatmapSource(m)
        ui = Image.open(io.BytesIO(src.render()))
        full = Image.open(io.BytesIO(src.render_full()))
        assert ui.size == (300, 400)   # k = 2 at the 400px UI target
        assert full.size == (600, 800)  # native: 800 <= FULL_TARGET_LONG_SIDE
        assert FULL_TARGET_LONG_SIDE >= 800


class TestColorbarSource:
    def test_renders_the_shared_colorbar_asset(self):
        assert ColorbarSource().render() == render_colorbar_png_bytes()

    def test_round_trips_through_the_source_registry(self, tmp_path):
        save_resource_source(
            str(tmp_path), "S", "heatmap_colorbar", "family", ColorbarSource(),
        )
        loaded = load_resource_source(str(tmp_path), "S", "heatmap_colorbar", "family")
        assert loaded is not None
        assert loaded.render() == render_colorbar_png_bytes()
        assert resource_source_from_state("colorbar", {}).render() == ColorbarSource().render()


class TestIrGraphFamilyScale:
    def test_summary_carries_scale_and_colorbar_descriptor(self):
        graph = _two_core_graph()
        snap, descriptors = snapshot_ir_graph(graph)

        scale = snap["heatmap_scale"]
        assert scale["vmax"] > 0 and scale["vmin"] == -scale["vmax"]
        assert scale["colorbar_resource"]["kind"] == RESOURCE_KIND_HEATMAP_COLORBAR
        colorbars = [d for d in descriptors if d.kind == RESOURCE_KIND_HEATMAP_COLORBAR]
        assert len(colorbars) == 1
        assert colorbars[0].rid == scale["colorbar_resource"]["rid"]

    def test_family_scale_is_max_of_per_matrix_p98(self):
        graph = _two_core_graph()
        snap, descriptors = snapshot_ir_graph(graph)
        family_kinds = {"ir_core_heatmap", "ir_core_pre_pruning", "ir_bank_heatmap"}
        vmax = _family_vmax(descriptors, family_kinds)
        expected = max(
            symmetric_scale(d.source.matrix)
            for d in descriptors
            if d.kind in family_kinds and isinstance(d.source, HeatmapSource)
        )
        assert vmax == pytest.approx(expected)
        assert snap["heatmap_scale"]["vmax"] == pytest.approx(expected)

    def test_bias_strips_keep_their_own_units(self):
        graph = _two_core_graph()
        _, descriptors = snapshot_ir_graph(graph)
        for d in descriptors:
            if d.kind == RESOURCE_KIND_IR_CORE_BIAS:
                assert d.source.scale is None

    def test_embedded_summary_reports_the_same_scale_without_descriptors(self):
        graph = _two_core_graph()
        default_snap, _ = snapshot_ir_graph(graph)
        embed_snap, embed_descriptors = snapshot_ir_graph(
            graph, source_step_name="Soft Core Mapping",
        )
        assert embed_descriptors == []
        assert embed_snap["heatmap_scale"]["vmax"] == pytest.approx(
            default_snap["heatmap_scale"]["vmax"]
        )
        assert embed_snap["heatmap_scale"]["colorbar_resource"]["step"] == "Soft Core Mapping"


class TestHardCoreFamilyScale:
    def _mapping(self):
        w = np.ones((3, 2), dtype="float32")
        src = _sources([(-2, 0), (-2, 1), (-3, 0)])
        core = NeuralCore(id=0, name="c", input_sources=src, core_matrix=w, latency=0)
        out = np.array([IRSource(0, 0)], dtype=object)
        ir = IRGraph(nodes=[core], output_sources=out)
        return build_hybrid_hard_core_mapping(
            ir_graph=ir,
            cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 2}],
        )

    def test_summary_scale_and_shared_source_stamp(self):
        summary, descriptors = snapshot_hard_core_mapping(self._mapping())
        scale = summary["heatmap_scale"]
        assert scale["vmax"] > 0 and scale["vmin"] == -scale["vmax"]
        vmax = _family_vmax(descriptors, {RESOURCE_KIND_HARD_CORE_HEATMAP})
        assert scale["vmax"] == pytest.approx(vmax)
        colorbars = [d for d in descriptors if d.kind == RESOURCE_KIND_HEATMAP_COLORBAR]
        assert len(colorbars) == 1
        assert colorbars[0].media_type == "image/png"


class TestPruningLayersFamilyScale:
    def test_summary_scale_and_shared_source_stamp(self):
        from types import SimpleNamespace

        layer = torch.nn.Linear(4, 3)
        layer.weight.data = torch.arange(12, dtype=torch.float32).reshape(3, 4) * 0.1
        layer.register_buffer("prune_row_mask", torch.tensor([True, False, False]))
        layer.register_buffer("prune_col_mask", torch.tensor([False, True, False, False]))
        model = SimpleNamespace(
            get_perceptrons=lambda: [SimpleNamespace(layer=layer, name="fc0")]
        )

        summary, descriptors = snapshot_pruning_layers(model)
        scale = summary["heatmap_scale"]
        assert scale["vmax"] > 0 and scale["vmin"] == -scale["vmax"]
        vmax = _family_vmax(descriptors, {"pruning_layer_heatmap"})
        assert scale["vmax"] == pytest.approx(vmax)
        assert any(d.kind == RESOURCE_KIND_HEATMAP_COLORBAR for d in descriptors)
