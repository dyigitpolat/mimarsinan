#!/usr/bin/env python3
"""Rewrite legacy shim import paths to canonical subpackage paths."""

from __future__ import annotations

import re
from pathlib import Path

# Longer paths first to avoid partial replacements.
REPLACEMENTS: list[tuple[str, str]] = [
    ("mimarsinan.mapping.hybrid_hardcore_mapping", "mimarsinan.mapping.packing.hybrid_hardcore_mapping"),
    ("mimarsinan.mapping.neural_segment_packing", "mimarsinan.mapping.packing.neural_segment_packing"),
    ("mimarsinan.mapping.pruning_graph_propagation", "mimarsinan.mapping.pruning.graph"),
    ("mimarsinan.mapping.layout_verification_stats", "mimarsinan.mapping.verification.layout_verification_scheduling"),
    ("mimarsinan.mapping.layout_mapping_service", "mimarsinan.mapping.verification.layout_mapping_service"),
    ("mimarsinan.mapping.ir_pruning_analysis", "mimarsinan.mapping.pruning.ir_pruning_analysis"),
    ("mimarsinan.mapping.core_quantization_verification_step", "mimarsinan.pipelining.pipeline_steps.mapping.core_quantization_verification_step"),
    ("mimarsinan.pipelining.pipeline_steps.quantization_verification_step", "mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step"),
    ("mimarsinan.pipelining.pipeline_steps.core_quantization_verification_step", "mimarsinan.pipelining.pipeline_steps.mapping.core_quantization_verification_step"),
    ("mimarsinan.pipelining.pipeline_steps.activation_quantization_step", "mimarsinan.pipelining.pipeline_steps.quantization.activation_quantization_step"),
    ("mimarsinan.pipelining.pipeline_steps.architecture_search_step", "mimarsinan.pipelining.pipeline_steps.config.architecture_search_step"),
    ("mimarsinan.pipelining.pipeline_steps.model_configuration_step", "mimarsinan.pipelining.pipeline_steps.config.model_configuration_step"),
    ("mimarsinan.pipelining.pipeline_steps.activation_adaptation_step", "mimarsinan.pipelining.pipeline_steps.adaptation.activation_adaptation_step"),
    ("mimarsinan.pipelining.pipeline_steps.activation_analysis_step", "mimarsinan.pipelining.pipeline_steps.adaptation.activation_analysis_step"),
    ("mimarsinan.pipelining.pipeline_steps.pruning_adaptation_step", "mimarsinan.pipelining.pipeline_steps.adaptation.pruning_adaptation_step"),
    ("mimarsinan.pipelining.pipeline_steps.normalization_fusion_step", "mimarsinan.pipelining.pipeline_steps.quantization.normalization_fusion_step"),
    ("mimarsinan.pipelining.pipeline_steps.hard_core_mapping_step", "mimarsinan.pipelining.pipeline_steps.mapping.hard_core_mapping_step"),
    ("mimarsinan.pipelining.pipeline_steps.soft_core_mapping_step", "mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step"),
    ("mimarsinan.pipelining.pipeline_steps.sanafe_simulation_step", "mimarsinan.pipelining.pipeline_steps.verification.sanafe_simulation_step"),
    ("mimarsinan.pipelining.pipeline_steps.weight_preloading_step", "mimarsinan.pipelining.pipeline_steps.config.weight_preloading_step"),
    ("mimarsinan.pipelining.pipeline_steps.weight_quantization_step", "mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step"),
    ("mimarsinan.pipelining.pipeline_steps.loihi_simulation_step", "mimarsinan.pipelining.pipeline_steps.verification.loihi_simulation_step"),
    ("mimarsinan.pipelining.pipeline_steps.clamp_adaptation_step", "mimarsinan.pipelining.pipeline_steps.adaptation.clamp_adaptation_step"),
    ("mimarsinan.pipelining.pipeline_steps.activation_shift_step", "mimarsinan.pipelining.pipeline_steps.adaptation.activation_shift_step"),
    ("mimarsinan.pipelining.pipeline_steps.model_building_step", "mimarsinan.pipelining.pipeline_steps.config.model_building_step"),
    ("mimarsinan.pipelining.pipeline_steps.torch_mapping_step", "mimarsinan.pipelining.pipeline_steps.config.torch_mapping_step"),
    ("mimarsinan.pipelining.pipeline_steps.lif_adaptation_step", "mimarsinan.pipelining.pipeline_steps.adaptation.lif_adaptation_step"),
    ("mimarsinan.pipelining.pipeline_steps.noise_adaptation_step", "mimarsinan.pipelining.pipeline_steps.adaptation.noise_adaptation_step"),
    ("mimarsinan.pipelining.pipeline_steps.simulation_step", "mimarsinan.pipelining.pipeline_steps.verification.simulation_step"),
    ("mimarsinan.pipelining.pipeline_steps.pretraining_step", "mimarsinan.pipelining.pipeline_steps.training.pretraining_step"),
    ("mimarsinan.visualization.search_visualization", "mimarsinan.visualization.search_viz"),
    ("mimarsinan.visualization.mapping_graphviz", "mimarsinan.visualization.graphviz"),
    ("mimarsinan.mapping.wizard_layout_verify", "mimarsinan.mapping.verification.wizard_layout_verify"),
    ("mimarsinan.mapping.pruning_propagation", "mimarsinan.mapping.pruning.graph.pruning_propagation"),
    ("mimarsinan.mapping.platform_constraints", "mimarsinan.mapping.platform.platform_constraints"),
    ("mimarsinan.mapping.mapping_structure", "mimarsinan.mapping.platform.mapping_structure"),
    ("mimarsinan.mapping.layout_verification_stats", "mimarsinan.mapping.verification.layout_verification_scheduling"),
    ("mimarsinan.mapping.layout_mapping_service", "mimarsinan.mapping.verification.layout_mapping_service"),
    ("mimarsinan.mapping.hw_config_suggester", "mimarsinan.mapping.verification.suggester.hw_config_suggester"),
    ("mimarsinan.mapping.mapping_verifier", "mimarsinan.mapping.verification.verifier"),
    ("mimarsinan.mapping.liveness_semantics", "mimarsinan.mapping.pruning.liveness_semantics"),
    ("mimarsinan.mapping.ir_segmentation", "mimarsinan.mapping.pruning.ir_segmentation"),
    ("mimarsinan.mapping.ir_pruning_analysis", "mimarsinan.mapping.pruning.ir_pruning_analysis"),
    ("mimarsinan.mapping.soft_core_mapper", "mimarsinan.mapping.packing.softcore.soft_core_mapper"),
    ("mimarsinan.mapping.softcore_mapping", "mimarsinan.mapping.packing.softcore"),
    ("mimarsinan.mapping.core_packing", "mimarsinan.mapping.packing.core_packing"),
    ("mimarsinan.mapping.layout_request", "mimarsinan.mapping.verification.layout_request"),
    ("mimarsinan.mapping.chip_quantize", "mimarsinan.mapping.export.chip_quantize"),
    ("mimarsinan.mapping.ir_liveness", "mimarsinan.mapping.pruning.ir_liveness"),
    ("mimarsinan.mapping.chip_export", "mimarsinan.mapping.export.chip_export"),
    ("mimarsinan.mapping.chip_latency", "mimarsinan.mapping.latency.chip"),
    ("mimarsinan.mapping.ir_latency", "mimarsinan.mapping.latency.ir"),
    ("mimarsinan.mapping.ir_pruning", "mimarsinan.mapping.pruning.ir_pruning"),
    ("mimarsinan.mapping.coalescing", "mimarsinan.mapping.platform.coalescing"),
    ("mimarsinan.mapping.pruning_apply", "mimarsinan.mapping.pruning.pruning_apply"),
]

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [ROOT / "src", ROOT / "tests", ROOT / "ARCHITECTURE.md", ROOT / "docs"]


def migrate_text(text: str) -> tuple[str, int]:
    n = 0
    for old, new in REPLACEMENTS:
        count = text.count(old)
        if count:
            text = text.replace(old, new)
            n += count
    return text, n


def main() -> int:
    total = 0
    for base in SCAN_DIRS:
        paths = [base] if base.is_file() else list(base.rglob("*"))
        for path in paths:
            if not path.is_file():
                continue
            if path.suffix not in {".py", ".md"}:
                continue
            if "mimarsinan-baseline-test" in str(path):
                continue
            if path.name == "migrate_phase2_imports.py":
                continue
            text = path.read_text(encoding="utf-8")
            new_text, n = migrate_text(text)
            if n:
                path.write_text(new_text, encoding="utf-8")
                total += n
                print(f"{path}: {n}")
    print(f"Total replacements: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
