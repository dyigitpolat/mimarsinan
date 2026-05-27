#!/usr/bin/env python3
"""Rewrite imports after Phase 3 package restructures (no shims)."""

from __future__ import annotations

import re
from pathlib import Path

# Longer paths first.
REPLACEMENTS: list[tuple[str, str]] = [
    # chip_simulation subpackages
    ("mimarsinan.chip_simulation.ttfs.ttfs_segment", "mimarsinan.chip_simulation.ttfs.ttfs_segment"),
    ("mimarsinan.chip_simulation.ttfs.ttfs_executor", "mimarsinan.chip_simulation.ttfs.ttfs_executor"),
    ("mimarsinan.chip_simulation.ttfs.ttfs_recorder", "mimarsinan.chip_simulation.ttfs.ttfs_recorder"),
    ("mimarsinan.chip_simulation.ttfs.ttfs_encoding", "mimarsinan.chip_simulation.ttfs.ttfs_encoding"),
    ("mimarsinan.models.spiking.ttfs_kernels", "mimarsinan.models.spiking.ttfs_kernels"),
    ("mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner", "mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner"),
    ("mimarsinan.chip_simulation.hybrid_run.hybrid_execution", "mimarsinan.chip_simulation.hybrid_run.hybrid_execution"),
    ("mimarsinan.chip_simulation.hybrid_run.hybrid_semantics", "mimarsinan.chip_simulation.hybrid_run.hybrid_semantics"),
    ("mimarsinan.chip_simulation.nevresim.compile_nevresim", "mimarsinan.chip_simulation.nevresim.compile_nevresim"),
    ("mimarsinan.chip_simulation.nevresim.execute_nevresim", "mimarsinan.chip_simulation.nevresim.execute_nevresim"),
    ("mimarsinan.chip_simulation.nevresim.nevresim_driver", "mimarsinan.chip_simulation.nevresim.nevresim_driver"),
    ("mimarsinan.chip_simulation.recording.spike_recorder", "mimarsinan.chip_simulation.recording.spike_recorder"),
    ("mimarsinan.chip_simulation.recording.spike_modes", "mimarsinan.chip_simulation.recording.spike_modes"),
    ("mimarsinan.chip_simulation.recording._spike_encoding", "mimarsinan.chip_simulation.recording._spike_encoding"),
    # mapping support
    ("mimarsinan.mapping.support.activation_scales", "mimarsinan.mapping.support.activation_scales"),
    ("mimarsinan.mapping.support.core_geometry", "mimarsinan.mapping.support.core_geometry"),
    ("mimarsinan.mapping.support.shape_probe", "mimarsinan.mapping.support.shape_probe"),
    ("mimarsinan.mapping.support.spike_source_spans", "mimarsinan.mapping.support.spike_source_spans"),
    ("mimarsinan.mapping.support.ir_source_spans", "mimarsinan.mapping.support.ir_source_spans"),
    ("mimarsinan.mapping.support.scale_broadcast", "mimarsinan.mapping.support.scale_broadcast"),
    ("mimarsinan.mapping.support.schedule.schedule_partitioner", "mimarsinan.mapping.support.schedule.schedule_partitioner"),
    ("mimarsinan.mapping.support.ttfs_bias", "mimarsinan.mapping.support.ttfs_bias"),
    ("mimarsinan.mapping.support.compute_modules", "mimarsinan.mapping.support.compute_modules"),
    ("mimarsinan.mapping.support.per_source_scales", "mimarsinan.mapping.support.per_source_scales"),
    # pipelining core
    ("mimarsinan.pipelining.core.engine.pipeline_helpers", "mimarsinan.pipelining.core.engine.pipeline_helpers"),
    ("mimarsinan.pipelining.core.steps.pipeline_step", "mimarsinan.pipelining.core.steps.pipeline_step"),
    ("mimarsinan.pipelining.core.registry.trainer_factory", "mimarsinan.pipelining.core.registry.trainer_factory"),
    ("mimarsinan.pipelining.core.steps.tuner_pipeline_step", "mimarsinan.pipelining.core.steps.tuner_pipeline_step"),
    ("mimarsinan.pipelining.core.steps.trainer_pipeline_step", "mimarsinan.pipelining.core.steps.trainer_pipeline_step"),
    ("mimarsinan.pipelining.core.registry.model_registry", "mimarsinan.pipelining.core.registry.model_registry"),
    ("mimarsinan.pipelining.core.accuracy_budget", "mimarsinan.pipelining.core.accuracy_budget"),
    ("mimarsinan.pipelining.core.search_mode", "mimarsinan.pipelining.core.search_mode"),
    ("mimarsinan.pipelining.core.model_config_emit", "mimarsinan.pipelining.core.model_config_emit"),
    ("mimarsinan.pipelining.core.hybrid_mapping_consumer", "mimarsinan.pipelining.core.hybrid_mapping_consumer"),
    ("mimarsinan.pipelining.core.simulation_factory", "mimarsinan.pipelining.core.simulation_factory"),
    ("mimarsinan.pipelining.core.platform_constraints_resolver", "mimarsinan.pipelining.core.platform_constraints_resolver"),
    ("mimarsinan.pipelining.core.engine.pipeline", "mimarsinan.pipelining.core.engine.pipeline"),
    # models spiking / nn
    ("mimarsinan.models.spiking.unified.flow", "mimarsinan.models.spiking.unified.flow"),
    ("mimarsinan.models.spiking.hybrid.flow", "mimarsinan.models.spiking.hybrid.flow"),
    ("mimarsinan.models.spiking.lif_core_step", "mimarsinan.models.spiking.lif_core_step"),
    ("mimarsinan.models.spiking.signal_spans", "mimarsinan.models.spiking.signal_spans"),
    ("mimarsinan.models.spiking.spiking_config", "mimarsinan.models.spiking.spiking_config"),
    ("mimarsinan.models.spiking.ttfs_activation", "mimarsinan.models.spiking.ttfs_activation"),
    ("mimarsinan.models.spiking.ttfs_kernels", "mimarsinan.models.spiking.ttfs_kernels"),
    ("mimarsinan.models.nn.layers", "mimarsinan.models.nn.layers"),
    ("mimarsinan.models.nn.decorators", "mimarsinan.models.nn.decorators"),
    ("mimarsinan.models.nn.activations", "mimarsinan.models.nn.activations"),
    ("mimarsinan.models.nn.lif_kernels", "mimarsinan.models.nn.lif_kernels"),
    # gui runtime
    ("mimarsinan.gui.runtime.collector", "mimarsinan.gui.runtime.collector"),
    ("mimarsinan.gui.runtime.persistence", "mimarsinan.gui.runtime.persistence"),
    ("mimarsinan.gui.runtime.process_manager", "mimarsinan.gui.runtime.process_manager"),
    ("mimarsinan.gui.runtime.active_run_hub", "mimarsinan.gui.runtime.active_run_hub"),
    ("mimarsinan.gui.runtime.run_cache_seed", "mimarsinan.gui.runtime.run_cache_seed"),
    ("mimarsinan.gui.runtime.composite_reporter", "mimarsinan.gui.runtime.composite_reporter"),
    ("mimarsinan.gui.runtime.snapshot_executor", "mimarsinan.gui.runtime.snapshot_executor"),
    # tuning orchestration
    ("mimarsinan.tuning.orchestration.unified_tuner", "mimarsinan.tuning.orchestration.unified_tuner"),
    ("mimarsinan.tuning.orchestration.adaptation_manager_factory", "mimarsinan.tuning.orchestration.adaptation_manager_factory"),
    ("mimarsinan.tuning.orchestration.adaptation_manager", "mimarsinan.tuning.orchestration.adaptation_manager"),
    ("mimarsinan.tuning.orchestration.tuning_budget", "mimarsinan.tuning.orchestration.tuning_budget"),
    # transformations perceptron
    ("mimarsinan.transformations.perceptron.perceptron_transformer", "mimarsinan.transformations.perceptron.perceptron_transformer"),
    # builders torch
    ("mimarsinan.models.builders.torch.torch_vgg16_builder", "mimarsinan.models.builders.torch.torch_vgg16_builder"),
    ("mimarsinan.models.builders.torch.torch_squeezenet11_builder", "mimarsinan.models.builders.torch.torch_squeezenet11_builder"),
    ("mimarsinan.models.builders.torch.torch_vit_builder", "mimarsinan.models.builders.torch.torch_vit_builder"),
    ("mimarsinan.models.builders.torch.torchvision_builder_utils", "mimarsinan.models.builders.torch.torchvision_builder_utils"),
    # packing softcore split
    ("mimarsinan.mapping.packing.softcore", "mimarsinan.mapping.packing.softcore"),
    ("mimarsinan.mapping.packing.softcore.soft_core_mapper", "mimarsinan.mapping.packing.softcore.soft_core_mapper"),
    # agent evolve (if not already migrated)
    ("mimarsinan.search.optimizers.agent_evolve", "mimarsinan.search.optimizers.agent_evolve"),
]

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [ROOT / "src", ROOT / "tests", ROOT / "scripts", ROOT / "ARCHITECTURE.md", ROOT / "docs"]


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
        paths = [base] if base.is_file() else sorted(base.rglob("*.py"))
        for path in paths:
            if not path.is_file() or "mimarsinan-baseline-test" in str(path):
                continue
            text = path.read_text(encoding="utf-8")
            new_text, n = migrate_text(text)
            if n:
                path.write_text(new_text, encoding="utf-8")
                total += n
                print(f"{n:4d}  {path.relative_to(ROOT)}")
    print(f"\nTotal replacements: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
