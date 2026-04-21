"""Regression tests: SoftCoreMappingStep retains pre-pruning heatmaps by default.

The GUI monitor's IR Graph and Hardware tabs show two heatmaps per soft core
(pre-pruning vs. post-compaction). Those views rely on
``NeuralCore.pre_pruning_heatmap`` being populated by ``prune_ir_graph`` while
the Soft Core Mapping step runs.

A previous refactor (commit ``0503648``) gated that retention on the
``generate_visualizations`` config — which is actually meant to control
graphviz DOT/SVG dumps on disk, a different artefact — so the monitor views
silently disappeared unless the user happened to enable Graphviz output.

These tests pin down the correct contract:

* With pruning enabled and no explicit visualisation config, the Soft Core
  Mapping step **must** store ``pre_pruning_heatmap`` and the pre-compaction
  masks on every neural core whose weights were compacted.
* A user who really needs to save memory can disable the heatmaps by setting
  ``store_pre_pruning_heatmap`` to ``False`` in the pipeline config.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from conftest import (
    MockPipeline,
    make_tiny_supermodel,
    default_config,
    platform_constraints,
)
from mimarsinan.pipelining.pipeline_steps.soft_core_mapping_step import SoftCoreMappingStep
from mimarsinan.mapping.ir import NeuralCore


def _fused_model_with_identity_norm():
    """Return a tiny 'fused' model (Identity norm) usable as fused_model seed."""
    model = make_tiny_supermodel()
    for p in model.get_perceptrons():
        p.normalization = nn.Identity()
    return model


def _register_prune_masks_zeroing_half_columns(model):
    """Mark half of each perceptron's input columns as pruned.

    This is enough to drive ``prune_ir_graph`` down the "columns removed →
    pre_pruning_heatmap captured" branch while keeping the asserted contract
    shape-independent.
    """
    for p in model.get_perceptrons():
        layer = p.layer
        if not hasattr(layer, "weight") or layer.weight is None:
            continue
        out_f = layer.weight.shape[0]
        in_f = layer.weight.shape[1]
        row_mask = torch.zeros(out_f, dtype=torch.bool)
        col_mask = torch.zeros(in_f, dtype=torch.bool)
        col_mask[: max(1, in_f // 2)] = True
        # Zero-out the pruned columns in the weight so value-based pruning
        # (the pipeline uses both mask- and zero-based) stays consistent.
        with torch.no_grad():
            layer.weight[:, col_mask] = 0.0
        layer.register_buffer("prune_row_mask", row_mask)
        layer.register_buffer("prune_col_mask", col_mask)


def _run_soft_core_mapping(mock_pipeline, fused_model, platform_constraints_dict):
    mock_pipeline.config.update(default_config())
    mock_pipeline.config["pruning"] = True
    mock_pipeline.config["weight_quantization"] = False
    mock_pipeline.config["weight_bits"] = 8

    mock_pipeline.seed("fused_model", fused_model, step_name="Normalization Fusion")
    mock_pipeline.seed(
        "platform_constraints_resolved",
        platform_constraints_dict,
        step_name="Model Configuration",
    )

    scm = SoftCoreMappingStep(mock_pipeline)
    scm.name = "Soft Core Mapping"
    mock_pipeline.prepare_step(scm)
    scm.run()
    return mock_pipeline.cache["Soft Core Mapping.ir_graph"]


class TestPrePruningHeatmapDefaultRetention:
    """The Soft Core Mapping step should retain pre-pruning heatmaps by default.

    This keeps the GUI monitor's "before → after pruning" comparison
    working without requiring users to enable ``generate_visualizations``
    (which only controls graphviz output, a separate concern).
    """

    def test_default_config_retains_pre_pruning_heatmap(
        self, mock_pipeline, platform_constraints
    ):
        model = _fused_model_with_identity_norm()
        _register_prune_masks_zeroing_half_columns(model)

        ir_graph = _run_soft_core_mapping(mock_pipeline, model, platform_constraints)

        cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore) and n.core_matrix is not None]
        assert cores, "Expected at least one NeuralCore after Soft Core Mapping"

        cores_with_pre = [c for c in cores if getattr(c, "pre_pruning_heatmap", None) is not None]
        assert cores_with_pre, (
            "Default config must store pre_pruning_heatmap on at least one "
            "NeuralCore so the monitor GUI can render pre/post pruning views."
        )
        for c in cores_with_pre:
            arr = np.asarray(c.pre_pruning_heatmap)
            assert arr.ndim == 2
            assert arr.dtype == np.float32

    def test_explicit_opt_out_skips_pre_pruning_heatmap(
        self, mock_pipeline, platform_constraints
    ):
        """``store_pre_pruning_heatmap=False`` lets memory-constrained runs skip the extra copy."""
        model = _fused_model_with_identity_norm()
        _register_prune_masks_zeroing_half_columns(model)

        mock_pipeline.config["store_pre_pruning_heatmap"] = False
        ir_graph = _run_soft_core_mapping(mock_pipeline, model, platform_constraints)

        cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore) and n.core_matrix is not None]
        assert cores, "Expected at least one NeuralCore after Soft Core Mapping"
        for c in cores:
            assert getattr(c, "pre_pruning_heatmap", None) is None, (
                "With store_pre_pruning_heatmap=False the step must not hold on "
                "to the full-matrix pre-pruning copy."
            )

    def test_generate_visualizations_flag_does_not_gate_heatmap(
        self, mock_pipeline, platform_constraints
    ):
        """``generate_visualizations`` only controls graphviz output, not monitor heatmaps."""
        model = _fused_model_with_identity_norm()
        _register_prune_masks_zeroing_half_columns(model)

        # generate_visualizations=False used to silently disable monitor views.
        mock_pipeline.config["generate_visualizations"] = False
        ir_graph = _run_soft_core_mapping(mock_pipeline, model, platform_constraints)

        cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore) and n.core_matrix is not None]
        cores_with_pre = [c for c in cores if getattr(c, "pre_pruning_heatmap", None) is not None]
        assert cores_with_pre, (
            "generate_visualizations=False must not suppress pre_pruning_heatmap; "
            "that flag is for graphviz DOT/SVG output, not the monitor's heatmap views."
        )
