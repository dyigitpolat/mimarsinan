# pipelining/ — Pipeline engine: turns a deployment config into an ordered, cache-backed, contract-verified step run

This module owns the end-to-end deployment pipeline: parsing a deployment-config
JSON, resolving it into a declarative `DeploymentPlan`, assembling the ordered
step sequence via a contract-driven `StepPlan`, and executing it with the
`Pipeline` engine (persisted `PipelineCache`, requires/promises data contracts,
per-step metric-retention tolerance, pre/post-step hooks). `PipelineSession` in
`session.py` is the composition root the `run.py` entry point drives; the
concrete `PipelineStep` implementations live under `pipeline_steps/`.

## Key files
| File | Purpose |
|---|---|
| `session.py` | Composition root: `parse_deployment_config` → `ParsedDeploymentConfig`, and `PipelineSession` owning one configured `DeploymentPipeline` (presets, GUI attach, start/stop-step resolution, run lifecycle). |
| `cache/` | `PipelineCache` — persisted step-output store keyed per step, with pluggable `LoadStoreStrategy` serializers (basic/JSON, torch model, pickle). The torch-model strategy enforces the prune-parity contract at the boundary: commit + verify on store, fail-loud verify on load. |
| `core/` | Engine and planning: `engine/` (`Pipeline` execution engine, resume/debug helpers), `steps/` (`PipelineStep` base with class-level `REQUIRES`/`PROMISES` contracts plus trainer/tuner step bases), `deployment_plan.py` (`DeploymentPlan` — the single config-resolution layer for every deployment axis), `step_plan.py` (`StepPlan` ordered registry filtered by each step's `applies_to(plan)`, contract-validated at assembly), `pipelines/` (`DeploymentPipeline`, `get_pipeline_step_specs` step ordering), `registry/` (`ModelRegistry`, trainer factory), plus `simulation_factory.py`, `nf_scm_parity.py` (NF↔SCM parity gates), `accuracy_budget.py`, `platform_constraints_resolver.py`, `hybrid_mapping_consumer.py`, `search_mode.py`, `model_config_emit.py`. |
| `pipeline_steps/` | Concrete step implementations grouped by phase: `config/` (architecture search, model configuration/building, torch mapping, weight preloading), `training/` (pretraining), `adaptation/` (activation/clamp/shift/pruning/LIF/TTFS-cycle/noise tuner steps), `quantization/` (normalization fusion, activation/weight quantization, verification), `mapping/` (soft/hard core mapping, core quantization verification), `verification/` (nevresim, Loihi, SANA-FE simulation steps). |

## Dependencies
- `chip_simulation` — spiking-mode semantics and `SpikingDeploymentContract` read by `DeploymentPlan`; `BACKEND_REGISTRY` splices the simulation-step tail; simulation runners, cost extraction, and record comparison for the verification steps and parity gates.
- `mapping` — IR graph/mapping construction, packing, IR pruning, chip quantization, platform constraints, and capacity/majority verification used by the mapping steps and `simulation_factory`.
- `tuning` — tuner classes run by the adaptation/quantization steps; temporal allocation and tuning-budget orchestration read by `DeploymentPlan` and the tuner step base.
- `common` — `best_effort` degrade seam, diagnostics/profiling, env flags, file utils, `DefaultReporter`.
- `model_training` — `BasicTrainer` construction (`registry/trainer_factory.py`), training recipes, weight-loading strategies.
- `data_handling` — data provider/loader factories for the session and steps; test-sample loading for simulation metrics.
- `models` — model layers and decorators used by steps, `SpikingHybridCoreFlow`, perceptron bias-reference refresh.
- `transformations` — normalization fusion, `PerceptronTransformer`, magnitude pruning, quantization bounds, `pruning.committed_masks` commit/verify at the cache store/load boundary.
- `spiking` — cycle-accurate LIF train application and scale-aware boundary calibration.
- `search` — joint arch/HW search problem and result types for `ArchitectureSearchStep`.
- `config_schema` — config defaults and deployment derivation folded into pipeline configs.
- `torch_mapping` — torch-model conversion and conversion probing in `TorchMappingStep`.
- `gui` — JSON-safe serialization and `CompositeReporter` for GUI wiring.
- `visualization` — search-progress visualization during architecture search.

## Dependents
- `run.py` (entry point) — drives a full deployment run through `PipelineSession`.
- `chip_simulation` — reads `DeploymentPlan`, the platform-constraints resolver, the parity record comparator, and the pipeline step classes registered per backend.
- `tuning` — tuners and orchestration read `DeploymentPlan`, `resolve_bias_mode`, and activation utils.
- `models` — builders register themselves via `ModelRegistry`.
- `mapping` — wizard layout verification instantiates builders through `ModelRegistry`.
- `gui` — wizard schema/routes, run monitors, and collectors use `DeploymentPipeline`, step specs, and the model registry.
- `config_schema` — display-view build reads model config schemas and pipeline step specs.

## Exported API
- `Pipeline` — the step-sequencing execution engine (from `core/engine/`).
- `PipelineStep` — the abstract step base with requires/promises data contracts (from `core/steps/`).
