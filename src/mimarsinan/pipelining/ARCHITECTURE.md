# pipelining/ -- Pipeline Engine

Orchestrates the end-to-end deployment pipeline: step sequencing, cache
management, data contract verification, and performance tolerance enforcement.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `pipeline.py` | `Pipeline` | Orchestration engine: step registration, verification, execution (uses `step.pipeline_metric()` for target), hook management |
| `pipeline_step.py` | `PipelineStep` | Abstract base class with data contracts (`requires`, `promises`, `updates`, `clears`); `process()`, `validate()`, `pipeline_metric()` (auto-discovers trainer for full-test-set evaluation), and optional `cleanup()` for resource release |
| `trainer_pipeline_step.py` | `TrainerPipelineStep` | Base for steps that own a `BasicTrainer`; default `validate()` delegates to trainer. |
| `tuner_pipeline_step.py` | `TunerPipelineStep` | Tuner-backed steps: `validate()`, `_commit_tuner_entries`, `run_tuner(TunerCls, …)`. |
| `trainer_factory.py` | `make_basic_trainer` | Shared `BasicTrainer` construction (`report_function`, optional `recipe`). Used by pretrain, fusion, preload, torch map, activation analysis, SCM, quant verify. |
| `pipeline_helpers.py` | `require_lif_spiking_mode`, `run_optional_viz`, `safe_warmup_forward` | Loihi/SANA-FE guards, non-fatal mapping viz, model warmup. |
| `platform_constraints_resolver.py` | `build_platform_constraints_resolved` | Single builder for `platform_constraints_resolved` (model config + NAS fixed path). |
| `model_config_emit.py` | `emit_model_config_entries` | Shared `model_builder` / `model_config` cache emission. |
| `simulation_factory.py` | `build_hybrid_mapping_for_pipeline`, … | Hybrid mapping, SCM/HCM metric, cached `hybrid_mapping`, Loihi/SANA-FE parity. Segment flush respects `use_legacy_softcore_flush` in deployment config (default: `neural_segment_packing`). |
| `model_registry.py` | `ModelRegistry`, `get_model_types`, `get_model_config_schema` | Registry populated by builders via `@ModelRegistry.register`; builders expose `get_config_schema()` for GUI form generation |

### Subdirectories

| Directory | Purpose |
|-----------|---------|
| `cache/` | Pipeline cache with pluggable serialization strategies |
| `pipelines/` | `DeploymentPipeline` — **`get_pipeline_step_specs(config)`** is the single source of truth for step order (LIF vs TTFS activation chains, optional Loihi/SANA-FE after Simulation). See `pipelines/deployment_pipeline.py`. |
| `pipeline_steps/` | Individual step implementations |

## Dependencies

- **Internal**: `common.file_utils` (`prepare_containing_directory`), `pipelining.cache`.
- **External**: `os`, `json`.

## Dependents

- Entry point (`main.py`) creates and runs `DeploymentPipeline`.
- `gui` registers hooks on `Pipeline` for monitoring.

## Exported API (\_\_init\_\_.py)

`Pipeline`, `PipelineStep`.
