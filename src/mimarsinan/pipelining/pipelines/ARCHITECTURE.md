# pipelining/pipelines/ -- Concrete Pipeline Assemblies

Contains fully assembled pipelines that compose pipeline steps into
end-to-end workflows.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `deployment_pipeline.py` | `DeploymentPipeline` | Unified configurable pipeline; assembles steps dynamically from `pipeline_mode` and `spiking_mode` |
| `deployment_pipeline.py` | `get_pipeline_step_specs` | Pure function: given a config dict, returns ordered list of `(step_name, step_class)`. Single source of truth for step order and presence; used by `_assemble_steps()` and by the wizard API for pipeline preview. |
| `deployment_pipeline.py` | `get_pipeline_semantic_group_by_step_name` | Pure function: given a config dict, returns `{step_name: semantic_group_id}` for every step. Group ids are stable lowercase snake_case keys (e.g. `"activation"`, `"weight_quantization"`, `"hardware"`) used by the GUI to colour pipeline step bars. |
| `deployment_pipeline.py` | `_SEMANTIC_GROUP_BY_STEP_CLASS` | Internal dict mapping every step class to its semantic group id. |

## Pipeline step preview

The wizard calls **POST `/api/pipeline_steps`** (see `gui/server.py`) with the current deployment config; the handler merges config via `config_schema.deployment_derivation.derive_deployment_parameters` (when building flat config server-side), calls `get_pipeline_step_specs(config)`, and returns `{"steps": [name, ...], "semantic_groups": [group_id, ...]}` so the UI can show the pipeline that would run without executing it, with colour-coded semantic groups.

## Step ordering (`get_pipeline_step_specs`)

Single source of truth in `deployment_pipeline.py`. High-level branches:

| Condition | Effect |
|-----------|--------|
| `search_mode != "fixed"` | `ArchitectureSearchStep` instead of `ModelConfigurationStep` |
| `spiking_mode == "lif"` | `LIFAdaptationStep` after activation analysis |
| `spiking_mode != "lif"` and (`activation_quantization` or TTFS) | `ClampAdaptationStep`; optional activation-quant chain when `activation_quantization` |
| `weight_quantization` | weight-quant steps before mapping; `CoreQuantizationVerificationStep` after soft-core mapping |
| `enable_loihi_simulation` | `LoihiSimulationStep` after `SimulationStep` (LIF only at runtime) |
| `enable_sanafe_simulation` | `SanafeSimulationStep` after simulation (optional HCM parity gate) |

Always: model build → (pretrain or preload) → optional torch map → optional pruning → activation analysis/adaptation → normalization fusion → soft core mapping → hard core mapping → nevresim simulation.

## Dependencies

- **Internal**: `pipelining.pipeline`, `pipelining.pipeline_steps` (all step classes), `data_handling.data_provider_factory`.
- **External**: None.

## Dependents

- Entry point (`main.py`) instantiates and runs `DeploymentPipeline`.

## Exported API (\_\_init\_\_.py)

`DeploymentPipeline`, `get_pipeline_step_specs`, `get_pipeline_semantic_group_by_step_name`.
