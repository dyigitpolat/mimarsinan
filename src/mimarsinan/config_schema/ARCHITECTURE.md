# config_schema/ -- Deployment Config Schema and Validation

Single source of truth for default deployment parameters, platform constraints,
pipeline-mode presets, and validation of deployment config (JSON for main.py and
merged flat config for pipeline runtime).

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `defaults.py` | `DEFAULT_DEPLOYMENT_PARAMETERS`, `DEFAULT_PLATFORM_CONSTRAINTS`, `PIPELINE_MODE_PRESETS`, `CONFIG_KEYS_SET`, `get_default_*`, `apply_preset` | Defaults and preset merge. **`spiking_mode` default `"lif"`**. Tuner keys: `tuner_target_floor_ratio`, `tuner_calibrate_smooth_tolerance`, etc. Platform: `allow_coalescing`, `allow_per_layer_s` (EW1 RESERVED temporal capability gate), `allow_scheduling`, `max_schedule_passes`, `scheduling_latency_weight`. Temporal allocation (EW1 RESERVED): `s_allocation` (`uniform`/`explicit`/`budget`), `s_allocation_explicit`, `s_allocation_budget`. |
| `deployment_derivation.py` | `derive_deployment_parameters` | Mirrors `gui/static/js/wizard.js` `buildConfig()` rules: float weights → vanilla; LIF disables `activation_quantization`; `ttfs_quantized` enables it. Called from `wizard/config_builder.py` and `runtime.build_flat_pipeline_config`. |
| `runtime.py` | `build_flat_pipeline_config` | Merges defaults + deployment/platform dicts for API preview and wizard (no device I/O). Applies preset then `derive_deployment_parameters`. |
| `validation.py` | `validate_deployment_config`, `validate_merged_config`, `s_allocation_config_errors` | JSON / merged config validation; rejects deprecated coalescing keys via `mapping.coalescing`. **EW2 (Q2 foot-gun closed):** `s_allocation_config_errors(dp, pc)` validates the per-layer-S declaration — `s_allocation ∈ {uniform\|explicit\|budget}`. Only `uniform` is WIRED; `explicit`/`budget` are RESERVED resolver seams that would silently no-op to uniform, so validation **loud-rejects** them at config-validation time (`temporal_allocation.unsupported_s_allocation_error` / the `S_ALLOCATION_SUPPORTED_MODES` SSOT) — BEFORE the silent-uniform resolver path is reachable. Wired into `validate_deployment_config` (so `validate_wizard_state` enforces it). `pc` is retained for call-site stability but no capability gate is consulted. `uniform` (default) ⇒ byte-identical. |
| `display_view.py` / `display_view_meta.py` | `build_config_display_view`, `build_pipeline_config_view`, `load_saved_config_from_run_dir`, `CONFIG_DISPLAY_GROUPS`, `FIELD_DISPLAY_META` | Presentation-only metadata and structured monitor payloads: merges defaults, annotates field provenance (`explicit` / `default` / `derived` / `preset` / `runtime`), expands nested blocks (cores, recipes, model_config, arch_search). Values come from existing defaults/merge helpers — not duplicated. **EW2:** `FIELD_DISPLAY_META` adds the per-layer-S form fields — `allow_per_layer_s` (hardware capability gate) + `s_allocation` / `s_allocation_explicit` / `s_allocation_budget` (tuning group). |
| `namespaced_schema.py` | `CONCERN_GROUPS`, `KeySpec`, `KEY_SPECS`, `LEGACY_KEY_TABLE`, `to_namespaced`, `to_flat`, `provenance_table`, `keys_with_derivation`, `registered_flat_keys`, `unregistered_default_keys` | **V8** — namespaces the flat deployment/platform keys under their §2 concern group and records each key's owner + derivation (provenance registry). One translation table (`LEGACY_KEY_TABLE`) drives a byte-identical flat↔namespaced bijection (`to_namespaced`/`to_flat`). The runtime config stays flat (SSOT); this is the REGISTRY/PROVENANCE layer (concern view), **not** the consumer-side resolver. |
| `__init__.py` | Re-exports | Public API |

### Namespaced concern groups (V8)

The flat runtime config is still the byte-identical SSOT every pipeline step reads.
`namespaced_schema.py` adds the *concern view*: every default deployment/platform key
carries a `KeySpec(group, name, owner, derivation)` so it has an owning §2 group and
recorded provenance. `to_namespaced(flat) → {group: {name: value}}` and its inverse
`to_flat` round-trip byte-identically for every registered key (unregistered/runtime
keys pass through under the `run` group), so existing configs resolve identically.

Groups mirror §2 of `docs/DESIGN_GOALS_and_refactoring_vectors.md`: `workload`,
`spiking`, `hardware` (**migrated**: core grid + `weight_bits` + the `allow_*`
capability gates), `conversion`, `tuning`, `training`, `deployment_target`, `run`.
**Strangler-fig status:** all default keys are registered with provenance and the
shim is byte-identical; the `hardware` group is migrated end-to-end (declared owners +
self-contained bijection), the other groups are registered but their owners still read
the flat keys. The **EW1 RESERVED** per-layer-S axis adds `allow_per_layer_s` to
`hardware` (the capability gate, owner `ChipCapabilities/TemporalAllocation`) and
`s_allocation` / `s_allocation_explicit` / `s_allocation_budget` to `conversion`
(owner `TemporalAllocation`); the per-depth S map is derived by the ConversionPolicy
keystone (research), default `uniform` ⇒ byte-identical.

#### Provenance is real (not all `default`)

`derivation` records *where a key's value comes from* and is no longer uniformly
`"default"`:

| derivation | keys | written by |
|---|---|---|
| `derived` | `pipeline_mode`, `activation_quantization`, `weight_quantization` | `deployment_derivation.derive_deployment_parameters` (wizard parity) |
| `derived` | `firing_mode`, `spike_generation_mode`, `thresholding_mode` | `DeploymentPipeline.__init__` (`setdefault` from `spiking_mode`) |
| `runtime` | `device`, `input_shape`, `input_size`, `num_classes` | `DeploymentPipeline.__init__` (device probe / data provider) |
| `default` | the remaining declared defaults | `DEFAULT_DEPLOYMENT_PARAMETERS` / `DEFAULT_PLATFORM_CONSTRAINTS` |

The derived/runtime keys are *not* in the defaults dict (they have no standalone
default value); they are registered as extra KeySpecs so the provenance table is
complete. The conversion/runtime tags are locked to `display_view_meta.DERIVED_KEYS`
/ `RUNTIME_KEYS` (the established UI provenance truth) by the schema tests, and the
derivation pass's always-written trio is asserted `derived`. `keys_with_derivation(d)`
queries the registry by provenance.

#### This module is the registry; DeploymentPlan is the consumer-side resolver (V1)

`namespaced_schema` is the **REGISTRY / PROVENANCE** layer: it records each key's
owning concern + derivation and offers the namespaced concern view. It is **not** the
place consumers read resolved decisions. **Consumer-side resolution is owned by V1's
`DeploymentPlan`** (`pipelining/core/deployment_plan.py`): steps read
`DeploymentPlan` fields/properties, which are the resolved SSOT. Re-pointing the ~50
flat `config.get` consumers at `to_namespaced(config)[group]` is **deliberately not
done** — it would duplicate `DeploymentPlan` and add a dict-projection per read
without reducing blast radius. Concretely, the `deployment_target` group's backend
gates (`enable_nevresim_simulation` / `enable_loihi_simulation` /
`enable_sanafe_simulation`) are *already* resolved fields on `DeploymentPlan`, and the
residual `sanafe_*` / `nevresim_connectivity_mode` reads are single-caller and
co-located — so the correct future migration target for them is `DeploymentPlan`
fields, not `to_namespaced`. **To finish a group, add resolved fields to
`DeploymentPlan`** (the consumer SSOT) and keep `namespaced_schema` as the provenance
registry that the resolver's keys are declared in.

### Standing rule — pin every external dependency + version-guard at boundaries

The SANA-FE SIGFPE was an unpinned C++ dependency upgrade that silently core-dumped
instead of failing loud (design goal §7/§9). The standing policy, generalized:

- **Pin** every external dependency (and reduction-order-sensitive numerics, e.g.
  cuBLAS) to an exact version so the deployment is deterministic given (config, seed,
  versions).
- **Version/capability-guard at each integration boundary** (each backend/codegen
  seam): validate the dep's version/capabilities *at assembly* and raise an actionable
  error, never let an incompatibility reach the native layer. The sanafe version guard
  is the concrete instance; new backends must add the analogous guard.

## Deployment parameters (selected)

Read by `DeploymentPipeline` / steps (see also `deployment_pipeline.default_deployment_parameters`):

| Key | Role |
|-----|------|
| `spiking_mode` | `"lif"` (default), `"rate"`, `"ttfs"`, `"ttfs_quantized"` |
| `cycle_accurate_lif_forward` | LIF deploys the chip-aligned cross-layer forward (installed at finalize) when true (default **true**) |
| `thresholding_mode` | `"<"` strict vs `"<="` inclusive LIF firing |
| `enable_nevresim_simulation` | Append Nevresim Simulation step (default **true**) |
| `nevresim_connectivity_mode` | `"runtime"` (default) or `"compile_time"` — chip wiring in generated C++ |
| `enable_loihi_simulation` | Append Loihi Simulation step (LIF only) |
| `enable_sanafe_simulation` | Append SANA-FE Simulation step |
| `loihi_parity_sample_index` | Deterministic test index for Loihi parity |
| `sanafe_sample_count`, `sanafe_arch_preset`, `sanafe_custom_arch_path` | SANA-FE step behaviour |
| `activation_quantization`, `weight_quantization`, `pruning`, `pruning_fraction` | Step gating |
| `enable_training_noise` | Optional `NoiseAdaptationStep` after LIF adaptation |
| `max_simulation_samples`, `seed`, `simulation_steps` | Simulation subsampling and cycles |
| `training_recipe`, `tuning_recipe` | AdamW + cosine defaults (ViT-aligned) |

`CONFIG_KEYS_SET` in `defaults.py` lists keys consumed by steps/tuners/simulation (including `enable_nevresim_simulation`, `enable_loihi_simulation`, `sanafe_*`, `cycle_accurate_lif_forward`). Extend it when adding new pipeline config.

## Dependencies

- **Internal**: `mapping.coalescing` (coalescing-key validation), `tuning.orchestration.temporal_allocation` (the `s_allocation` mode constants the EW2 validation enforces).

## Dependents

- `pipelining.pipelines.deployment_pipeline`
- `gui/wizard/schema.py` (wizard forms; labels should match defaults)
- `gui/wizard/config_builder.py` (`derive_deployment_parameters` after preset)
- `gui/server/` (`build_flat_pipeline_config` for `/api/pipeline_steps`; `build_deployment_config_from_state` for `/api/run`)
- `gui/data_collector.py`, `gui/runs.py`, `gui/process_manager.py` (`build_config_display_view` / `build_pipeline_config_view` for monitor `config_view` in pipeline overview)

## Exported API (`__init__.py`)

`DEFAULT_DEPLOYMENT_PARAMETERS`, `DEFAULT_PLATFORM_CONSTRAINTS`, `PIPELINE_MODE_PRESETS`, `CONFIG_KEYS_SET`, `get_default_deployment_parameters`, `get_default_platform_constraints`, `get_pipeline_mode_presets`, `get_config_keys_set`, `apply_preset`, `validate_deployment_config`, `validate_merged_config`, `s_allocation_config_errors`, `build_flat_pipeline_config`, `build_config_display_view`, `build_pipeline_config_view`, `CONCERN_GROUPS`, `KEY_SPECS`, `KeySpec`, `LEGACY_KEY_TABLE`, `keys_with_derivation`, `provenance_table`, `to_flat`, `to_namespaced`.
