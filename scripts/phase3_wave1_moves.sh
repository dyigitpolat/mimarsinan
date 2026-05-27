#!/usr/bin/env bash
# Phase 3 wave 1: move modules into subpackages (run from mimarsinan repo root).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src/mimarsinan"
cd "$ROOT"

mkdir -p "$SRC/chip_simulation/ttfs" "$SRC/chip_simulation/hybrid_run" \
  "$SRC/chip_simulation/nevresim" "$SRC/chip_simulation/recording" \
  "$SRC/mapping/support" "$SRC/pipelining/core" \
  "$SRC/models/spiking" "$SRC/models/nn" \
  "$SRC/gui/runtime" "$SRC/tuning/orchestration" \
  "$SRC/transformations/perceptron" "$SRC/models/builders/torch" \
  "$SRC/mapping/packing/softcore"

git_mv() { git mv "$1" "$2" 2>/dev/null || mv "$1" "$2"; }

# chip_simulation
for f in ttfs_segment ttfs_executor ttfs_recorder ttfs_encoding; do
  git_mv "$SRC/chip_simulation/${f}.py" "$SRC/chip_simulation/ttfs/${f}.py"
done
for f in hybrid_stage_runner hybrid_execution hybrid_semantics; do
  git_mv "$SRC/chip_simulation/${f}.py" "$SRC/chip_simulation/hybrid_run/${f}.py"
done
for f in compile_nevresim execute_nevresim nevresim_driver; do
  git_mv "$SRC/chip_simulation/${f}.py" "$SRC/chip_simulation/nevresim/${f}.py"
done
for f in spike_recorder spike_modes _spike_encoding; do
  git_mv "$SRC/chip_simulation/${f}.py" "$SRC/chip_simulation/recording/${f}.py"
done
rm -f "$SRC/chip_simulation/ttfs_kernels.py"

# mapping support
for f in activation_scales core_geometry shape_probe spike_source_spans ir_source_spans \
  scale_broadcast schedule_partitioner ttfs_bias compute_modules per_source_scales; do
  git_mv "$SRC/mapping/${f}.py" "$SRC/mapping/support/${f}.py"
done

# pipelining core
for f in pipeline pipeline_step pipeline_helpers trainer_factory tuner_pipeline_step \
  trainer_pipeline_step model_registry accuracy_budget search_mode model_config_emit \
  hybrid_mapping_consumer simulation_factory platform_constraints_resolver; do
  git_mv "$SRC/pipelining/${f}.py" "$SRC/pipelining/core/${f}.py"
done

# models
for f in unified_core_flow hybrid_core_flow lif_core_step signal_spans spiking_config ttfs_activation ttfs_kernels; do
  git_mv "$SRC/models/${f}.py" "$SRC/models/spiking/${f}.py"
done
for f in layers decorators activations lif_kernels; do
  git_mv "$SRC/models/${f}.py" "$SRC/models/nn/${f}.py"
done

# gui runtime
for f in data_collector persistence process_manager active_run_stream run_cache_seed \
  composite_reporter snapshot_executor; do
  git_mv "$SRC/gui/${f}.py" "$SRC/gui/runtime/${f}.py"
done

# tuning orchestration
for f in unified_tuner adaptation_manager_factory adaptation_manager tuning_budget; do
  git_mv "$SRC/tuning/${f}.py" "$SRC/tuning/orchestration/${f}.py"
done

# transformations
git_mv "$SRC/transformations/perceptron_transformer.py" "$SRC/transformations/perceptron/perceptron_transformer.py"

# builders torch (4 largest)
for f in torch_vgg16_builder torch_squeezenet11_builder torch_vit_builder torchvision_builder_utils; do
  git_mv "$SRC/models/builders/${f}.py" "$SRC/models/builders/torch/${f}.py"
done

# packing softcore (move only; split in later step)
git_mv "$SRC/mapping/packing/softcore_mapping.py" "$SRC/mapping/packing/softcore/softcore_mapping.py"
git_mv "$SRC/mapping/packing/soft_core_mapper.py" "$SRC/mapping/packing/softcore/soft_core_mapper.py"

echo "Wave 1 file moves done."
