"""Function-level imports are a tracked, shrinking allowlist (cycle-breakers and optional deps only)."""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"

_INDENTED_IMPORT = re.compile(r"^\s+(from|import)\s+[A-Za-z_.]")

# Files still carrying function-level imports (import cycles to untangle or
# optional heavy backends). Shrink this list; never grow it.
ALLOWLIST = {
    "chip_simulation/backend.py",
    "chip_simulation/cost_extraction.py",
    "chip_simulation/deployment_contract.py",
    "chip_simulation/deployment_faithfulness.py",
    "chip_simulation/firing_strategy.py",
    "chip_simulation/lava_loihi/core_lava.py",
    "chip_simulation/sanafe/arch_synth/spec.py",
    "chip_simulation/sanafe/net_synth/build.py",
    "chip_simulation/sanafe/net_synth/spike_trains.py",
    "chip_simulation/sanafe/neuron_model.py",
    "chip_simulation/sanafe/records/run.py",
    "chip_simulation/sanafe/runner/core.py",
    "chip_simulation/sanafe/runner/neural_stage.py",
    "chip_simulation/sanafe/runner/neural_stage_record.py",
    "chip_simulation/sanafe/runner/segment_io.py",
    "chip_simulation/spiking_mode_policy.py",
    "chip_simulation/test_cross_sim_parity.py",
    "chip_simulation/ttfs/ttfs_segment.py",
    "code_generation/cpp_chip_model_types.py",
    "common/diagnostics.py",
    "common/file_utils.py",
    "config_schema/defaults.py",
    "config_schema/display_view.py",
    "config_schema/display_view_build.py",
    "data_handling/data_loader_factory.py",
    "data_handling/data_provider.py",
    "data_handling/data_provider_factory.py",
    "data_handling/data_providers/imagenet_data_provider.py",
    "data_handling/ffcv/loader_factory.py",
    "data_handling/ffcv/writer.py",
    "gui/heatmap_renderer.py",
    "gui/reporter.py",
    "gui/runs.py",
    "gui/runtime/collector/collector.py",
    "gui/runtime/collector/mixins/read_api.py",
    "gui/runtime/collector/mixins/steps.py",
    "gui/runtime/process_monitor.py",
    "gui/server/app.py",
    "gui/server/routes_layout.py",
    "gui/server/routes_pipeline.py",
    "gui/server/routes_resources.py",
    "gui/server/routes_wizard.py",
    "gui/snapshot/heatmap.py",
    "gui/snapshot/ir_graph/ir_graph_resources.py",
    "gui/snapshot/ir_graph/ir_graph_topology.py",
    "gui/snapshot/mapping_snapshot.py",
    "gui/snapshot/util/helpers.py",
    "mapping/ir/legacy_convert.py",
    "mapping/ir/types.py",
    "mapping/layout/layout_ir_mapping.py",
    "mapping/packing/hybrid_build_pool.py",
    "mapping/packing/hybrid_build_scheduled.py",
    "mapping/packing/hybrid_segment.py",
    "mapping/packing/neural_segment_packing.py",
    "mapping/packing/softcore/compaction.py",
    "mapping/packing/softcore/hard_core.py",
    "mapping/packing/softcore/hard_core_mapping.py",
    "mapping/packing/softcore/soft_core.py",
    "mapping/packing/softcore/soft_core_mapper.py",
    "mapping/pruning/boundary_policy.py",
    "mapping/pruning/ir_liveness.py",
    "mapping/pruning/liveness_semantics.py",
    "mapping/support/neg_shift_bias.py",
    "mapping/support/residual_merge.py",
    "mapping/support/schedule/schedule_partitioner.py",
    "mapping/support/schedule/schedule_split.py",
    "mapping/verification/onchip_fraction.py",
    "mapping/verification/verifier/mapping_verifier_hw.py",
    "mapping/verification/verifier/mapping_verifier_soft.py",
    "mapping/verification/verifier/mapping_verifier_types.py",
    "models/nn/activations/lif.py",
    "models/nn/activations/ttfs_spiking.py",
    "models/perceptron_mixer/perceptron.py",
    "models/pretrained_bridge.py",
    "models/spiking/training/blended_genuine_forward.py",
    "models/test_pretrained_bridge.py",
    "models/test_squeezenet.py",
    "pipelining/core/deployment_plan.py",
    "pipelining/core/nf_scm_parity.py",
    "pipelining/core/platform_constraints_resolver.py",
    "pipelining/core/registry/model_registry.py",
    "pipelining/pipeline_steps/config/architecture_search_helpers.py",
    "pipelining/pipeline_steps/mapping/hard_core_mapping_step.py",
    "pipelining/pipeline_steps/mapping/soft_core_mapping_step.py",
    "pipelining/pipeline_steps/mapping/soft_core_mapping_viz.py",
    "search/optimizers/__init__.py",
    "search/optimizers/compilagent/backend/backend_tools.py",
    "search/optimizers/compilagent/guidance_blocks.py",
    "search/optimizers/compilagent/guided_toolset.py",
    "search/optimizers/compilagent/sink/sink.py",
    "search/optimizers/compilagent/tools.py",
    "search/optimizers/llm/trace.py",
    "search/search_space_description.py",
    "spiking/compute_boundary.py",
    "spiking/distribution_matching.py",
    "spiking/lif_utils.py",
    "spiking/segment_policies.py",
    "spiking/spike_trains.py",
    "torch_mapping/converter.py",
    "torch_mapping/converter_handlers/conv_mixin.py",
    "torch_mapping/converter_handlers/linear_mixin.py",
    "torch_mapping/converter_handlers/structural_mixin.py",
    "training/imagenet_fast_train.py",
    "transformations/perceptron/perceptron_transformer.py",
    "transformations/pruning/masks.py",
    "tuning/orchestration/adaptation_manager.py",
    "tuning/orchestration/calibration_pipeline.py",
    "tuning/orchestration/conversion_policy.py",
    "tuning/orchestration/fast_ladder.py",
    "tuning/orchestration/kd_blend_adaptation_tuner.py",
    "tuning/orchestration/ramp_strategy.py",
    "tuning/orchestration/ttfs_adaptation_plan.py",
    "tuning/tuners/activation_adaptation_tuner.py",
    "tuning/tuners/activation_shift_tuner.py",
    "tuning/tuners/lif_adaptation_tuner.py",
    "tuning/tuners/ttfs_cycle_adaptation_tuner.py",
}


def _files_with_indented_imports():
    found = set()
    for path in SRC.rglob("*.py"):
        for line in path.read_text().splitlines():
            if _INDENTED_IMPORT.match(line) and not line.lstrip().startswith("#"):
                found.add(str(path.relative_to(SRC)))
                break
    return found


def test_no_new_function_level_imports():
    offenders = _files_with_indented_imports() - ALLOWLIST
    assert not offenders, (
        "new function-level imports (move to module top, or justify a cycle-break "
        f"and extend the allowlist): {sorted(offenders)}"
    )


def test_allowlist_has_no_dead_entries():
    dead = ALLOWLIST - _files_with_indented_imports()
    assert not dead, f"allowlist entries no longer needed (remove them): {sorted(dead)}"
