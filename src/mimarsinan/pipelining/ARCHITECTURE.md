# pipelining/ — Pipeline engine: turns a deployment config into an ordered, cache-backed, contract-verified step run

This module owns the end-to-end deployment pipeline: parsing a deployment-config
JSON, resolving it into a declarative `DeploymentPlan`, assembling the ordered
step sequence via a contract-driven `StepPlan`, and executing it with the
`Pipeline` engine (persisted `PipelineCache`, requires/promises data contracts,
per-step metric-retention tolerance, pre/post/step-failed hooks — the failure
hooks observe a dying step with its error and the exception re-raises
untouched). `PipelineSession` in
`session.py` is the composition root the `run.py` entry point drives; the
concrete `PipelineStep` implementations live under `pipeline_steps/`.

## Key files
| File | Purpose |
|---|---|
| `session.py` | Composition root: `parse_deployment_config` → `ParsedDeploymentConfig`, and `PipelineSession` owning one configured `DeploymentPipeline` (presets, GUI attach, start/stop-step resolution, run lifecycle); `apply_determinism` is the `PipelineSession/determinism` owner — seeds every RNG family from `DeploymentPlan.seed` and pins deterministic fp32 math (TF32 off) before any step runs. Surfaces deployment advisories (warnings only, `best_effort`-scoped): config advisories at `run()` start, the post-pretrain envelope gate via a post-step hook. |
| `cache/` | `PipelineCache` — persisted step-output store keyed per step, with pluggable `LoadStoreStrategy` serializers (basic/JSON, torch model, pickle). The torch-model strategy enforces the prune-parity contract at the boundary: commit + verify on store, fail-loud verify on load. |
| `core/` | Engine and planning: `engine/` (`Pipeline` execution engine, resume/debug helpers), `steps/` (`PipelineStep` base with class-level `REQUIRES`/`PROMISES` contracts plus trainer/tuner step bases; the tuner base persists the W3-S1 run-dir adaptation artifacts at commit time — the `ft_pass_walls.json` accumulator + one `retention_ledger.json` entry per tuner-hosting step, via `tuning.orchestration.run_instrumentation`), `deployment_plan.py` (`DeploymentPlan` — the single config-resolution layer for every deployment axis), `step_plan.py` (`StepPlan` ordered registry filtered by each step's `applies_to(plan)`, contract-validated at assembly), `pipelines/` (`DeploymentPipeline`, `get_pipeline_step_specs` step ordering; `merge_pipeline_config` / `apply_provider_facts` / `apply_workload_profiles` are the exact config-merge + provider-fact + profile-fold seam the golden-resolution snapshot harness shares; the plan carries the folded result as `DeploymentPlan.workload`), `registry/` (`ModelRegistry`, trainer factory), plus `simulation_factory.py`, `nf_scm_parity.py` (NF↔SCM parity gates), `spike_count_gate.py` (deployed spike-count certificate gate, `spike_count_parity_samples`), `accuracy_budget.py`, `platform_constraints_resolver.py`, `hybrid_mapping_consumer.py`, `search_mode.py`, `model_config_emit.py`. |
| `pipeline_steps/` | Concrete step implementations grouped by phase: `config/` (architecture search, model configuration/building, torch mapping, weight preloading — which asserts the accuracy the freshly loaded weights produce against the weight set's recorded `expected_accuracy` through `common.measurement`, disarmed by `unusable_baseline_reason` whenever the builder adapted the workload), `training/` (pretraining), `adaptation/` (activation/clamp/shift/pruning/LIF/TTFS-cycle/noise tuner steps, the pre-WQ `LIFAffineFoldStep` [C4], plus the registry-gated `ScaleMigrationStep` — exact cross-layer channel-scale migration between Pruning Adaptation and Activation Analysis, postcondition-checked to preserve the float function — and the exact-QAT-KD-gated `ReferenceTeacherSnapshotStep` — freezes the post-structural float model right after Scale Migration as the exact-QAT distillation teacher), `quantization/` (normalization fusion, activation/weight quantization, the mvm-only `BoundaryQuantizationStep` — platform `activation_bits` calibrates + installs value-grid quantizers at segment entries, verification), `mapping/` (soft/hard core mapping, core quantization verification), `verification/` (nevresim, Loihi, SANA-FE simulation steps; the SANA-FE step reads the declared core capacity + `cores_per_tile`/`tile_grid_*` floorplan keys from `platform_constraints_resolved`, so the simulated NoC floorplan is fixed by the declared platform, not the packed model; the nevresim step's window-count certificate runs its HCM twin on the PIPELINE device and inside `measurement_plane()` — like the runner that fed the chip — so host-op tread ties are decided by lattice values, never by the device/batch GEMM dust of wherever the samples happened to live). |

## Dependencies
- `chip_simulation` — spiking-mode semantics and `SpikingDeploymentContract` read by `DeploymentPlan`; `BACKEND_REGISTRY` splices the simulation-step tail; simulation runners, cost extraction, and record comparison for the verification steps and parity gates.
- `mapping` — IR graph/mapping construction, packing, IR pruning, chip quantization, platform constraints, and capacity/majority verification used by the mapping steps and `simulation_factory`.
- `tuning` — tuner classes run by the adaptation/quantization steps; temporal allocation and tuning-budget orchestration read by `DeploymentPlan` and the tuner step base.
- `common` — `best_effort` degrade seam, diagnostics/profiling, env flags, file utils, `DefaultReporter`, the `measurement` baseline-agreement contract and the `pretrained` weight-set predicates.
- `model_training` — `BasicTrainer` construction (`registry/trainer_factory.py`), training recipes, weight-loading strategies.
- `data_handling` — data provider/loader factories for the session and steps; test-sample loading for simulation metrics.
- `models` — model layers and decorators used by steps, `SpikingHybridCoreFlow`, perceptron bias-reference refresh.
- `transformations` — normalization fusion, `PerceptronTransformer`, magnitude pruning, quantization bounds, `pruning.committed_masks` commit/verify at the cache store/load boundary, `pruning.seed_generators` (the `prune_criterion` seed seam in the soft-core structured-pruning hook), `equalize_channel_scales`/`DEFAULT_CLIP_RATIO` for the Scale Migration step and `DeploymentPlan`.
- `spiking` — cycle-accurate LIF train application and scale-aware boundary calibration.
- `search` — joint arch/HW search problem and result types for `ArchitectureSearchStep`.
- `config_schema` — config defaults and deployment derivation folded into pipeline configs.
- `torch_mapping` — torch-model conversion and conversion probing in `TorchMappingStep`.
- `gui` — JSON-safe serialization and `CompositeReporter` for GUI wiring.
- `visualization` — search-progress visualization during architecture search.
- `advisories` — deployment-advisory evaluation + surfacing (session config/post-pretrain seams, `TorchMappingStep` graph seam); `advisories` reads back only `pipelining.core.platform_constraints_resolver` (acyclic).

## Dependents
- `run.py` (entry point) — drives a full deployment run through `PipelineSession`.
- `chip_simulation` — reads `DeploymentPlan`, the platform-constraints resolver, the parity record comparator, and the pipeline step classes registered per backend.
- `tuning` — tuners and orchestration read `DeploymentPlan`, `resolve_bias_mode`, and activation utils.
- `models` — builders register themselves via `ModelRegistry`.
- `mapping` — wizard layout verification instantiates builders through `ModelRegistry`.
- `gui` — wizard schema/routes, run monitors, and collectors use `DeploymentPipeline`, step specs, and the model registry; the snapshot pruning gate consumes the canonical `PRUNING_ADAPTATION_STEP` step-name constant from `core/step_plan.py` (`core/pipelines/deployment_specs.py` re-exports it and builds the registry entry from it; the gate imports the `step_plan` original because the specs module's star-import of `pipeline_steps` cycles back through `gui`).
- `config_schema` — display-view build reads model config schemas and pipeline step specs.

## Exported API
- `Pipeline` — the step-sequencing execution engine (from `core/engine/`).
- `PipelineStep` — the abstract step base with requires/promises data contracts (from `core/steps/`).
