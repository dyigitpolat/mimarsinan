# pipelining/ -- Pipeline Engine

Orchestrates the end-to-end deployment pipeline: step sequencing, cache
management, data contract verification, and performance tolerance enforcement.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `pipeline.py` | `Pipeline` | Orchestration engine: step registration, verification, execution (uses `step.pipeline_metric()` for target), hook management.  Enforces a **two-layer accuracy floor** per step: `max(previous_metric * tolerance, baseline_test_metric * (1 - degradation_tolerance))`.  Auto-seeds `baseline_test_metric` from the first non-zero `pipeline_metric` (monotonic non-decreasing), so pre-training steps that legitimately report `0.0` don't trip the global floor.  Honors `step.skip_from_floor_check` (Phase B3): skipped steps neither fire the floor assertion nor overwrite `previous_metric`, so a pass-through / setup step reporting `0.0` can no longer silently reset the per-step ratchet. |
| `pipeline_step.py` | `PipelineStep` | Abstract base class with data contracts (`requires`, `promises`, `updates`, `clears`); `process()`, `validate()`, `pipeline_metric()`, and optional `cleanup()` for resource release.  `pipeline_metric()` is the **only** place in the framework allowed to hit the test set: it runs once per step, after any attached tuner has finalised, so test labels never leak into tuner decision logic.  Subclasses that don't produce a meaningful metric (pure configuration / model-building steps) can opt out of the floor assertion by setting the class attribute `skip_from_floor_check = True` (Phase B3). |
| `model_registry.py` | `ModelRegistry`, `get_model_types`, `get_model_config_schema` | Registry populated by builders via `@ModelRegistry.register`; builders expose `get_config_schema()` for GUI form generation |

### Subdirectories

| Directory | Purpose |
|-----------|---------|
| `cache/` | Pipeline cache with pluggable serialization strategies |
| `pipelines/` | Concrete pipeline assemblies (e.g., `DeploymentPipeline`) |
| `pipeline_steps/` | Individual step implementations |

## Dependencies

- **Internal**: `common.file_utils` (`prepare_containing_directory`), `pipelining.cache`.
- **External**: `os`, `json`.

## Dependents

- Entry point (`main.py`) creates and runs `DeploymentPipeline`.
- `gui` registers hooks on `Pipeline` for monitoring.

## Exported API (\_\_init\_\_.py)

`Pipeline`, `PipelineStep`.
