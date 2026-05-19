# pipelining/ -- Pipeline Engine

Orchestrates the end-to-end deployment pipeline: step sequencing, cache
management, data contract verification, and performance tolerance enforcement.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `pipeline.py` | `Pipeline` | Orchestration engine: step registration, verification, execution (uses `step.pipeline_metric()` for target), hook management |
| `pipeline_step.py` | `PipelineStep` | Abstract base class with data contracts (`requires`, `promises`, `updates`, `clears`); `process()`, `validate()`, `pipeline_metric()` (auto-discovers trainer for full-test-set evaluation), and optional `cleanup()` for resource release |
| `tuner_pipeline_step.py` | `TunerPipelineStep` | Shared validate / commit pattern for tuner-backed steps. |
| `simulation_factory.py` | `build_hybrid_mapping_for_pipeline`, `build_spiking_hybrid_flow`, `run_hcm_spiking_test` | Shared hybrid mapping + HCM flow construction for SCM, HCM, Loihi, and SANA-FE steps. |
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
