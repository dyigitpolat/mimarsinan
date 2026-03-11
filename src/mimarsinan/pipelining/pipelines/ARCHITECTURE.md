# pipelining/pipelines/ -- Concrete Pipeline Assemblies

Contains fully assembled pipelines that compose pipeline steps into
end-to-end workflows.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `deployment_pipeline.py` | `DeploymentPipeline` | Unified configurable pipeline; assembles steps dynamically from `pipeline_mode` and `spiking_mode` |
| `deployment_pipeline.py` | `get_pipeline_step_specs` | Pure function: given a config dict, returns ordered list of `(step_name, step_class)`. Single source of truth for step order and presence; used by `_assemble_steps()` and by the wizard API for pipeline preview. |

## Pipeline step preview

The wizard calls **POST `/api/pipeline_steps`** (see `gui/server.py`) with the current deployment config; the handler merges config, calls `get_pipeline_step_specs(config)`, and returns `{"steps": [name, ...]}` so the UI can show the pipeline that would run without executing it.

## Dependencies

- **Internal**: `pipelining.pipeline`, `pipelining.pipeline_steps` (all step classes), `data_handling.data_provider_factory`.
- **External**: None.

## Dependents

- Entry point (`main.py`) instantiates and runs `DeploymentPipeline`.

## Exported API (\_\_init\_\_.py)

`DeploymentPipeline`.
