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

The wizard calls **POST `/api/pipeline_steps`** (see `gui/server/routes_wizard.py`) with the current deployment config; the handler merges config via `config_schema.deployment_derivation.derive_deployment_parameters` (when building flat config server-side), calls `get_pipeline_step_specs(config)`, and returns `{"steps": [name, ...], "semantic_groups": [group_id, ...]}` so the UI can show the pipeline that would run without executing it, with colour-coded semantic groups.

## Step ordering (`get_pipeline_step_specs`)

Single source of truth in `deployment_pipeline.py`. High-level branches:

| Condition | Effect |
|-----------|--------|
| `search_mode != "fixed"` | `ArchitectureSearchStep` instead of `ModelConfigurationStep` |
| cycle-accurate tuning (`lif` / `ttfs_cycle_based`) | Activation Adaptation → Clamp → Shift → Activation Quantization before `LIFAdaptationStep` / `TTFSCycleAdaptationStep` |
| analytical/rate modes | `ClampAdaptationStep` when activation quantization or TTFS firing requires it; Shift/AQ only when `activation_quantization` |
| `weight_quantization` | weight-quant steps before mapping; `CoreQuantizationVerificationStep` after soft-core mapping |
| backend steps (nevresim / loihi / sanafe) | Selected + validated by **`chip_simulation.backend.BACKEND_REGISTRY`** (Vector V3): `selected_step_specs(plan)` reads each backend's `enable_*` predicate AND consults the `_BACKEND_CAPS` capability matrix UP-FRONT — an enabled backend×unsupported-mode (e.g. Loihi + TTFS) raises an actionable `ValueError` at assembly, before any step is appended. nevresim is skipped for the synchronized TTFS-cycle schedule (no synchronized-window backend yet). |

Always: model build → (pretrain or preload) → optional torch map → optional pruning → activation analysis/preconditioning → optional LIF/TTFS-cycle tuning → normalization fusion → soft core mapping → hard core mapping → backend simulation steps (registry-selected).

## Dependencies

- **Internal**: `pipelining.pipeline`, `pipelining.pipeline_steps` (all step classes), `chip_simulation.backend` (`BACKEND_REGISTRY` for backend step selection/validation), `data_handling.data_provider_factory`.
- **External**: None.

## Dependents

- Entry point (`main.py`) instantiates and runs `DeploymentPipeline`.

## Exported API (\_\_init\_\_.py)

`DeploymentPipeline`, `get_pipeline_step_specs`, `get_pipeline_semantic_group_by_step_name`.
