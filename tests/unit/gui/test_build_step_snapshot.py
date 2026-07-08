"""Regression tests for ``build_step_snapshot`` hot paths after refactors."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mimarsinan.gui.snapshot.builders import build_step_snapshot
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping


def _minimal_ir() -> IRGraph:
    w = __import__("numpy").ones((3, 2), dtype="float32")
    src = __import__("numpy").array(
        [IRSource(-2, 0), IRSource(-2, 1), IRSource(-3, 0)], dtype=object
    )
    core = NeuralCore(id=0, name="c", input_sources=src, core_matrix=w, latency=0)
    out = __import__("numpy").array([IRSource(0, 0)], dtype=object)
    return IRGraph(nodes=[core], output_sources=out)


class _Cache(dict):
    """Pipeline cache with dict-like ``.keys()`` / ``.get()``."""

    def keys(self):  # noqa: D102
        return super().keys()


class _SoftCoreMappingStep:
    promises = ("ir_graph",)
    updates = ()


class _HardCoreMappingStep:
    promises = ("hard_core_mapping",)
    updates = ()


def test_hard_core_mapping_snapshot_embeds_ir_graph_without_name_error():
    """Exercises the branch that calls ``_find_ir_graph_promiser`` (regression for NameError)."""
    ir = _minimal_ir()
    hm = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 2}],
    )
    cache = _Cache(
        {
            "hard_core_mapping": hm,
            "Soft Core Mapping.ir_graph": ir,
        }
    )
    pipeline = SimpleNamespace(
        cache=cache,
        steps=(
            ("Soft Core Mapping", _SoftCoreMappingStep()),
            ("Hard Core Mapping", _HardCoreMappingStep()),
        ),
        config={},
    )
    snap, _kinds, _descs = build_step_snapshot(
        pipeline, "Hard Core Mapping", step=_HardCoreMappingStep()
    )
    assert "hard_core_mapping" in snap
    assert "ir_graph" in snap
    assert "snapshot_error" not in snap


class _ActivationAnalysisStep:
    promises = ("activation_scales", "activation_scale_stats")
    updates = ()


def test_activation_scale_stats_pass_through_to_the_snapshot():
    """The Activation Analysis distribution stats are cached AND snapshotted —
    the step page renders per-layer scale distributions from them."""
    stats = {
        "num_batches": 4,
        "quantile": 0.99,
        "summary": {"min_scale": 1.0, "median_scale": 2.0, "max_scale": 3.0},
        "layers": [
            {"index": 0, "name": "features_0", "scale": 1.0,
             "sample_count": 128, "active_sample_count": 60,
             "sample_min": 0.0, "sample_median": 0.4, "sample_max": 2.5},
        ],
    }
    cache = _Cache({
        "activation_scales": [1.0],
        "activation_scale_stats": stats,
    })
    pipeline = SimpleNamespace(cache=cache, steps=(), config={})
    snap, kinds, _descs = build_step_snapshot(
        pipeline, "Activation Analysis", step=_ActivationAnalysisStep()
    )
    assert snap["activation_scale_stats"]["layers"][0]["name"] == "features_0"
    assert kinds["activation_scale_stats"] == "new"


def test_builders_imports_find_ir_graph_promiser():
    """``builders`` must bind helpers used at runtime, not only via star-import side effects."""
    import mimarsinan.gui.snapshot.builders as builders_mod

    from mimarsinan.gui.snapshot.sanafe_snapshot import _find_ir_graph_promiser

    assert builders_mod._find_ir_graph_promiser is _find_ir_graph_promiser
