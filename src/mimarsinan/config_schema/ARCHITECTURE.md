# config_schema/ -- Deployment Config Schema and Validation

Single source of truth for default deployment parameters, platform constraints,
pipeline-mode presets, and validation of deployment config (JSON for main.py and
merged flat config for pipeline runtime).

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `defaults.py` | `DEFAULT_DEPLOYMENT_PARAMETERS`, `DEFAULT_PLATFORM_CONSTRAINTS`, `PIPELINE_MODE_PRESETS`, `CONFIG_KEYS_SET`, `get_default_*`, `apply_preset` | Defaults and preset merge. **`spiking_mode` default `"lif"`**. Tuner keys: `tuner_target_floor_ratio`, `tuner_calibrate_smooth_tolerance`, etc. Platform: `allow_coalescing`, `allow_scheduling`, `max_schedule_passes`, `scheduling_latency_weight`. |
| `deployment_derivation.py` | `derive_deployment_parameters` | Mirrors `gui/static/js/wizard.js` `buildConfig()` rules: float weights → vanilla; LIF disables `activation_quantization`; `ttfs_quantized` enables it. Called from `wizard/config_builder.py` and `runtime.build_flat_pipeline_config`. |
| `runtime.py` | `build_flat_pipeline_config` | Merges defaults + deployment/platform dicts for API preview and wizard (no device I/O). Applies preset then `derive_deployment_parameters`. |
| `validation.py` | `validate_deployment_config`, `validate_merged_config` | JSON / merged config validation; rejects deprecated coalescing keys via `mapping.coalescing`. |
| `display_view.py` | `build_config_display_view`, `build_pipeline_config_view`, `load_saved_config_from_run_dir`, `CONFIG_DISPLAY_GROUPS`, `FIELD_DISPLAY_META` | Presentation-only metadata and structured monitor payloads: merges defaults, annotates field provenance (`explicit` / `default` / `derived` / `preset` / `runtime`), expands nested blocks (cores, recipes, model_config, arch_search). Values come from existing defaults/merge helpers — not duplicated. |
| `__init__.py` | Re-exports | Public API |

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

- **Internal**: `mapping.coalescing` (validation only).

## Dependents

- `pipelining.pipelines.deployment_pipeline`
- `gui/wizard/schema.py` (wizard forms; labels should match defaults)
- `gui/wizard/config_builder.py` (`derive_deployment_parameters` after preset)
- `gui/server/` (`build_flat_pipeline_config` for `/api/pipeline_steps`; `build_deployment_config_from_state` for `/api/run`)
- `gui/data_collector.py`, `gui/runs.py`, `gui/process_manager.py` (`build_config_display_view` / `build_pipeline_config_view` for monitor `config_view` in pipeline overview)

## Exported API (`__init__.py`)

`DEFAULT_DEPLOYMENT_PARAMETERS`, `DEFAULT_PLATFORM_CONSTRAINTS`, `PIPELINE_MODE_PRESETS`, `CONFIG_KEYS_SET`, `get_default_deployment_parameters`, `get_default_platform_constraints`, `get_pipeline_mode_presets`, `get_config_keys_set`, `apply_preset`, `validate_deployment_config`, `validate_merged_config`, `build_flat_pipeline_config`, `build_config_display_view`, `build_pipeline_config_view`.
