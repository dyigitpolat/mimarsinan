#!/usr/bin/env python3
"""Rewrite imports after Phase 3 wave 4 splits (no shims)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN = [ROOT / "src", ROOT / "tests"]

REPLACEMENTS: list[tuple[str, str]] = [
    ("mimarsinan.mapping.packing.softcore.softcore_mapping", "mimarsinan.mapping.packing.softcore"),
    ("mimarsinan.mapping.verification.layout_verification_stats import LayoutVerificationStats", "mimarsinan.mapping.verification.layout_verification_types import LayoutVerificationStats"),
    ("mimarsinan.mapping.verification.layout_verification_stats import build_layout_verification_stats", "mimarsinan.mapping.verification.layout_verification_packing import build_layout_verification_stats"),
    ("mimarsinan.mapping.verification.layout_verification_stats import build_stats_from_packing_result", "mimarsinan.mapping.verification.layout_verification_packing import build_stats_from_packing_result"),
    ("mimarsinan.mapping.verification.layout_verification_stats import compute_schedule_sync_count", "mimarsinan.mapping.verification.layout_verification_scheduling import compute_schedule_sync_count"),
    ("mimarsinan.mapping.verification.layout_verification_stats import compute_mapping_stats", "mimarsinan.mapping.verification.layout_verification_scheduling import compute_mapping_stats"),
    ("mimarsinan.mapping.verification.layout_verification_stats import stats_dict_from_hybrid_mapping", "mimarsinan.mapping.verification.layout_verification_hybrid import stats_dict_from_hybrid_mapping"),
    ("mimarsinan.mapping.verification.mapping_verifier import verify_soft_core_mapping", "mimarsinan.mapping.verification.verifier import verify_soft_core_mapping"),
    ("mimarsinan.mapping.verification.mapping_verifier import verify_hardware_config", "mimarsinan.mapping.verification.verifier import verify_hardware_config"),
    ("mimarsinan.mapping.verification.mapping_verifier import MappingVerificationResult", "mimarsinan.mapping.verification.verifier import MappingVerificationResult"),
    ("mimarsinan.mapping.verification.hw_config_suggester import suggest_hardware_config", "mimarsinan.mapping.verification.suggester import suggest_hardware_config"),
    ("mimarsinan.mapping.verification.hw_config_suggester import HardwareSuggestion", "mimarsinan.mapping.verification.suggester.hw_suggestion_types import HardwareSuggestion"),
    ("mimarsinan.mapping.pruning.ir_pruning import prune_ir_graph", "mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph"),
    ("mimarsinan.mapping.pruning.ir_pruning import get_initial_pruning_masks_from_model", "mimarsinan.mapping.pruning.ir_pruning_masks import get_initial_pruning_masks_from_model"),
    ("mimarsinan.mapping.pruning.pruning_graph_propagation import GlobalPruningResult", "mimarsinan.mapping.pruning.graph import GlobalPruningResult"),
    ("mimarsinan.mapping.pruning.pruning_graph_propagation import compute_global_pruned_sets", "mimarsinan.mapping.pruning.graph import compute_global_pruned_sets"),
    ("mimarsinan.mapping.pruning.pruning_propagation import compute_propagated_pruned_rows_cols", "mimarsinan.mapping.pruning.graph import compute_propagated_pruned_rows_cols"),
    ("mimarsinan.mapping.ir_mapping import IRMapping", "mimarsinan.mapping.ir_mapping_class import IRMapping"),
    ("mimarsinan.mapping.ir_mapping import map_model_to_ir", "mimarsinan.mapping.map_model_to_ir import map_model_to_ir"),
    ("mimarsinan.mapping.mappers.conv import Conv2DPerceptronMapper", "mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper"),
    ("mimarsinan.mapping.mappers.conv import Conv1DPerceptronMapper", "mimarsinan.mapping.mappers.conv1d_mapper import Conv1DPerceptronMapper"),
    ("mimarsinan.mapping.mappers.perceptron import PerceptronMapper", "mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper"),
    ("mimarsinan.mapping.mappers.perceptron import ComputeOpMapper", "mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper"),
    ("mimarsinan.mapping.mappers.perceptron import ModuleMapper", "mimarsinan.mapping.mappers.module_mapper import ModuleMapper"),
    ("mimarsinan.mapping.packing.hybrid_build import build_hybrid_hard_core_mapping", "mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping"),
    ("mimarsinan.visualization.graphviz.ir import write_ir_graph_dot", "mimarsinan.visualization.graphviz.ir_dot import write_ir_graph_dot"),
    ("mimarsinan.visualization.graphviz.ir import write_ir_graph_summary_dot", "mimarsinan.visualization.graphviz.ir_summary import write_ir_graph_summary_dot"),
    ("mimarsinan.visualization.graphviz.hybrid import HybridVizArtifacts", "mimarsinan.visualization.graphviz.hybrid_types import HybridVizArtifacts"),
    ("mimarsinan.visualization.graphviz.hybrid import write_hybrid_hardcore_mapping_dots", "mimarsinan.visualization.graphviz.hybrid_segment_dots import write_hybrid_hardcore_mapping_dots"),
    ("mimarsinan.visualization.graphviz.hybrid import write_hybrid_hardcore_mapping_combined_dot", "mimarsinan.visualization.graphviz.hybrid_combined_dot import write_hybrid_hardcore_mapping_combined_dot"),
    ("mimarsinan.visualization.softcore_flowchart import write_softcore_flowchart_dot", "mimarsinan.visualization.softcore_flowchart_dot import write_softcore_flowchart_dot"),
    ("mimarsinan.gui.snapshot.ir_graph_snapshot import snapshot_ir_graph", "mimarsinan.gui.snapshot.ir_graph import snapshot_ir_graph"),
    ("mimarsinan.mapping.layout.layout_source_view import stack_source_views", "mimarsinan.mapping.layout.layout_source_view_ops import stack_source_views"),
    ("mimarsinan.mapping.layout.layout_source_view import concat_source_views", "mimarsinan.mapping.layout.layout_source_view_ops import concat_source_views"),
]


def main() -> int:
    total = 0
    for base in SCAN:
        for path in sorted(base.rglob("*.py")):
            if "mimarsinan-baseline-test" in str(path) or path.name == "migrate_phase3_wave4_imports.py":
                continue
            text = path.read_text(encoding="utf-8")
            new = text
            for old, new_s in REPLACEMENTS:
                new = new.replace(old, new_s)
            if new != text:
                path.write_text(new, encoding="utf-8")
                total += 1
    print(f"Updated {total} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
