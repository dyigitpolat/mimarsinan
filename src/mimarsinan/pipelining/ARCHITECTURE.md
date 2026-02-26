# pipelining/ -- Pipeline Engine

Orchestrates the end-to-end deployment pipeline: step sequencing, cache
management, data contract verification, and performance tolerance enforcement.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `pipeline.py` | `Pipeline` | Orchestration engine: step registration, verification, execution, hook management |
| `pipeline_step.py` | `PipelineStep` | Abstract base class with data contracts (`requires`, `promises`, `updates`, `clears`) |

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
