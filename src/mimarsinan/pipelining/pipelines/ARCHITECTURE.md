# pipelining/pipelines/ -- Concrete Pipeline Assemblies

Contains fully assembled pipelines that compose pipeline steps into
end-to-end workflows.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `deployment_pipeline.py` | `DeploymentPipeline` | Unified configurable pipeline; assembles steps dynamically from `pipeline_mode` and `spiking_mode` |

## Dependencies

- **Internal**: `pipelining.pipeline`, `pipelining.pipeline_steps` (all step classes), `data_handling.data_provider_factory`.
- **External**: None.

## Dependents

- Entry point (`main.py`) instantiates and runs `DeploymentPipeline`.

## Exported API (\_\_init\_\_.py)

`DeploymentPipeline`.
