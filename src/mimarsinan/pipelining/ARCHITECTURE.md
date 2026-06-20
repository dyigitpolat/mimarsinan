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
| `deployment_plan.py` | `DeploymentPlan` | **V1 resolution backbone (`core/`).** `DeploymentPlan.resolve(config)` / `DeploymentPlan.of(pipeline)` is the SINGLE place the scattered `config.get(...)` reads of a deployment flag are resolved — generalises `TtfsAdaptationPlan` / `SpikingDeploymentContract` to EVERY axis (search/model, spiking schedule-derived booleans, act/wt quant, pruning + derived `pruning_enabled`, the `enable_*` backends, tolerances + the `2×tolerance` budget default, sim-metric sampling). Frozen; inline defaults preserved key-for-key (byte-identical — `test_deployment_plan.py`). The spiking-semantics sub-part stays the `SpikingDeploymentContract`, composed lazily via `plan.spiking_contract()` so a plan resolves from a config without `simulation_steps` (step-ordering time). `plan.mode_policy()` returns the V2 `SpikingModePolicy` (schedule-derived, no sim length) and `plan.is_lif_style` (V5) delegates to its `single_step_activation_replacement` — the (firing × sync) decision the step planner reads to pick the activation-adaptation family. Read by `deployment_specs.get_pipeline_step_specs`/`validate_deployment_config`, `deployment_pipeline`, `simulation_factory`, `pipeline_helpers`. |
| `step_plan.py` | `StepPlan`, `StepSpec` | **V5 contract-driven step planning (`core/`).** An ordered registry of `StepSpec(name, step_class, applies=None)`; `StepPlan.resolve(plan)` keeps registry order and drops steps whose `step_class.applies_to(plan)` is false — replacing the former hand-assembled per-flag `append`s in `deployment_specs`. Each `PipelineStep` owns its applicability via `applies_to(plan)` (base returns `True`; conditional steps override with the verbatim former condition). A callable registry entry splices V3's `BACKEND_REGISTRY.selected_step_specs(plan)` tail (which validates every enabled backend against the capability matrix UP-FRONT). Composes with V1 (`DeploymentPlan`) + V2 (`plan.is_lif_style` → policy). Byte-identical to the old assembly across the full config matrix (`test_step_plan.py`). |
| `pipeline_helpers.py` | `require_lif_spiking_mode`, `run_optional_viz`, `safe_warmup_forward` | Loihi/SANA-FE guards (read `DeploymentPlan.of(pipeline).spiking_mode`), non-fatal mapping viz, plain-model warmup (used by `ModelBuildingStep` to initialise lazy modules; `TorchMappingStep` uses the strict `conversion_probe.probe_forward` instead, see [torch_mapping/ARCHITECTURE.md](../torch_mapping/ARCHITECTURE.md)). |
| `platform_constraints_resolver.py` | `build_platform_constraints_resolved` | Single builder for `platform_constraints_resolved` (model config + NAS fixed path). |
| `model_config_emit.py` | `emit_model_config_entries` | Shared `model_builder` / `model_config` cache emission. |
| `simulation_factory.py` | `build_hybrid_mapping_for_pipeline`, `build_identity_mapping_for_pipeline`, `build_deployment_contract`, `run_scm_identity_metric`, `run_hcm_mapping_metric`, `build_spiking_hybrid_flow`, … | Hybrid mapping, the gate-ladder metrics, Loihi/SANA-FE parity. **Rung 2** = `run_scm_identity_metric` over the 1:1 identity mapping (`build_identity_mapping_for_pipeline` = identity build + negative-shift propagation) — IR semantics. **Rung 3** = `run_hcm_mapping_metric` over the packed mapping — packing. SCM no longer caches the packed `hybrid_mapping`; HCM builds it via `load_hybrid_mapping_for_step` when uncached or **stale** (the cached mapping's `source_ir_build_token` must match the step's current `ir_graph.build_token` — a resumed run that regenerated the ir_graph must not simulate the previous run's packed mapping). The old `build_spiking_flow_for_metric` alias is gone (use `build_spiking_hybrid_flow`). `build_deployment_contract` constructs the `SpikingDeploymentContract` SSOT; the flow/reference builders read schedule/sim-length semantics from it, not loose cfg keys. Non-spiking deployment flags (`cycle_accurate_lif_forward`, the metric-sampling knobs, OOM `simulation_batch_size`) come from the resolved `DeploymentPlan` (`DeploymentPlan.of(pipeline)`), not raw `config.get`. |
| `nf_scm_parity.py` | `assert_nf_scm_parity_or_raise`, `nf_scm_parity_enabled`, `NfScmParityError` | **NF↔SCM per-neuron parity gate** (rung 1 ↔ rung 2) for the modes whose NF *is* the deployment kernel: synchronized ttfs_cycle + continuous ttfs. Cascaded has its own per-neuron story; **ttfs_quantized is excluded by design** (its NF trains the floor-staircase + half-step-bias convention — one-step-per-layer agreement with the ceil contract, so per-neuron equality is not its invariant). **Analytic modes** (synchronized ttfs_cycle, continuous ttfs): compares per-perceptron NF activations (normalized, captured via hooks) against identity-mapped contract-runner records grouped by `perceptron_index` — **order-insensitive per (perceptron, sample) row** (`compare_normalized_records`: conv core emission order ≠ torch flatten order; positional wiring is enforced transitively by consumer exactness + rung-3/4); psum partials excluded; fails fast on a stale instance forward (legacy synchronized caches). Config: `nf_scm_parity_samples` (default 2, 0 disables), `nf_scm_parity_atol` (1e-6), `nf_scm_parity_max_mismatch_fraction` (default **0.02 for synchronized** — measured bit-exact 0/122880 once the tuner snaps segment entries — and 0.25 for continuous ttfs; the wrong-NF-dynamics signature measures ~40 %). **Cascaded** (`assert_cascaded_nf_scm_agreement_or_raise`): decision-level argmax agreement between the genuine segment driver and the identity-mapped cascade executor (`nf_scm_parity_samples_cascaded` default 64, `nf_scm_parity_min_agreement` default 0.98; healthy agreement is 1.0 — driver==executor bit-exact, locked by `test_ttfs_segment_node_recorder`; the 2026-06-07 0.85 readings were the stale-bias incident: layer-replacing steps must call `refresh_perceptron_bias_references`, see `transformations/normalization_fusion.py`). NOTE: the contract runner's record for ttfs_cycle is the ANALYTICAL staircase reference (greedy cascade legitimately fires early relative to it) — never use it as the genuine-dynamics reference. `MIMARSINAN_NF_SCM_PARITY_DEBUG=1` prints per-perceptron mismatch stats. Hooked into `SoftCoreMappingStep` BEFORE the model→cpu move (mapper-graph compute modules don't follow `model.to`). |
| `model_registry.py` | `ModelRegistry`, `get_model_types`, `get_model_config_schema` | Registry populated by builders via `@ModelRegistry.register`; builders expose `get_config_schema()` for GUI form generation |

### Subdirectories

| Directory | Purpose |
|-----------|---------|
| `cache/` | Pipeline cache with pluggable serialization strategies |
| `pipelines/` | `DeploymentPipeline` — **`get_pipeline_step_specs(config)`** is the single source of truth for step order; it resolves a `DeploymentPlan` (`core/deployment_plan.py`) and returns `_STEP_PLAN.resolve(plan)` — the V5 `StepPlan` registry (`core/step_plan.py`) where the order IS the pipeline order and each step's `applies_to(plan)` (the former per-flag conditions, now owned by the steps) filters it. `DeploymentPipeline._initialize_config`/`_display_config`/`_assemble_steps` likewise read the plan. See `pipelines/deployment_specs.py`. |
| `pipeline_steps/` | Individual step implementations |

## Dependencies

- **Internal**: `common.file_utils` (`prepare_containing_directory`), `pipelining.cache`.
- **External**: `os`, `json`.

## Dependents

- Entry point (`main.py`) creates and runs `DeploymentPipeline`.
- `gui` registers hooks on `Pipeline` for monitoring.

## Exported API (\_\_init\_\_.py)

`Pipeline`, `PipelineStep`.
