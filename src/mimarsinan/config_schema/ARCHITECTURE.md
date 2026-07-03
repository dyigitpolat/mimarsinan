# config_schema/ — Deployment config SSOT: defaults, derivation, validation, provenance, display views

Every pipeline run consumes one flat config dict merged from deployment parameters and
platform constraints; this module is the single source of truth for its default values,
pipeline-mode presets, and the derivation rules that fold spiking-mode implications and
the `ConversionPolicy` recipe into that dict. It also validates both the deployment
config JSON (`main.py` input) and the merged runtime config, records each key's concern
group and provenance in a namespaced registry, and builds presentation-only config views
for the GUI monitor. `DeploymentPipeline` and the wizard both build configs through these
helpers so they stay in sync.

## Key files
| File | Purpose |
|---|---|
| `defaults.py` | `DEFAULT_DEPLOYMENT_PARAMETERS` / `DEFAULT_PLATFORM_CONSTRAINTS` / training+tuning recipes, `PIPELINE_MODE_PRESETS` + `apply_preset`, `CONFIG_KEYS_SET`, and copy-returning getters (incl. user/system splits driven by exposure). |
| `deployment_derivation.py` | `derive_deployment_parameters` (wizard-parity quantization/pipeline-mode rules; folds the `ConversionPolicy.derive` recipe authoritatively — sim enables, driver, per-mode knobs) and `derive_pipeline_runtime_parameters` (rehydrates firing/spike-generation/thresholding fields). |
| `runtime.py` | `build_flat_pipeline_config` — merges defaults + overrides + preset + both derivation passes the same way `DeploymentPipeline` does, without device I/O. |
| `validation.py` | `validate_deployment_config` (JSON shape, spiking-mode membership, TTFS firing consistency, coalescing-key rejection), `validate_merged_config` (runtime flat config), `s_allocation_config_errors` (loud-rejects reserved `explicit`/`budget` temporal-allocation modes). |
| `namespaced_schema.py` | Provenance registry: `KeySpec` (group/owner/derivation/exposure per flat key), `CONCERN_GROUPS`, `KEY_SPECS`, `LEGACY_KEY_TABLE`, byte-identical `to_namespaced`/`to_flat` bijection, and `keys_with_derivation` / `keys_with_exposure` / `provenance_table` queries. |
| `display_view.py` | `build_config_display_view` / `build_pipeline_config_view` / `load_saved_config_from_run_dir` — structured, JSON-safe monitor payloads with per-field source annotation (explicit/default/derived/preset/runtime). |
| `display_view_meta.py` | Display metadata and resolution helpers: `CONFIG_DISPLAY_GROUPS`, `FIELD_DISPLAY_META`, `RUNTIME_KEYS` / `DERIVED_KEYS` / `TOP_LEVEL_RUN_KEYS`, field-source and default-value resolution. |
| `display_view_build.py` | Nested display-block builders: recipe, model-config (via model registry schema), cores, arch-search blocks, and the pipeline-steps preview. |

## Dependencies
- `chip_simulation` — `spiking_semantics` predicates (`require_known_spiking_mode`, `requires_ttfs_firing`, `forces_activation_quantization`, `is_cycle_based`) used by derivation and validation.
- `tuning` — `orchestration.conversion_policy.ConversionPolicy` (the mode→recipe SSOT folded into derived parameters) and `orchestration.temporal_allocation` (`s_allocation` mode constants and the unsupported-mode error).
- `mapping` — `platform.coalescing.coalescing_config_errors` rejects deprecated coalescing keys during config validation.
- `pipelining` — lazy display-view imports: `core.registry.model_registry.get_model_config_schema` (model-config field schema) and `core.pipelines.deployment_specs` (pipeline-steps preview).
- `gui` — lazy display-view imports: `wizard.config_builder.build_deployment_config_from_state` (expand saved wizard configs) and `wizard.schema.get_wizard_nas_schema` (arch-search block labels).

## Dependents
- `pipelining` — `core.pipelines.deployment_pipeline` (defaults + both derivation passes) and `core.platform_constraints_resolver` (`DEFAULT_PLATFORM_CONSTRAINTS`).
- `gui` — `wizard/schema.py` (form defaults), `wizard/validation.py` (`validate_deployment_config`), `wizard/config_builder.py` (`KEY_SPECS` / `keys_with_exposure`), `server/routes_wizard.py` (`build_flat_pipeline_config` preview), and `runs.py` / `runtime/process_monitor.py` / `runtime/collector/mixins/read_api.py` (monitor config views).

## Exported API
`__init__.py` re-exports:
- Defaults and presets: `DEFAULT_DEPLOYMENT_PARAMETERS`, `DEFAULT_PLATFORM_CONSTRAINTS`, `DEFAULT_TRAINING_RECIPE`, `DEFAULT_TUNING_RECIPE`, `PIPELINE_MODE_PRESETS`, `CONFIG_KEYS_SET`, the `get_default_*` / `get_pipeline_mode_presets` / `get_config_keys_set` getters, `apply_preset`.
- Runtime merge: `build_flat_pipeline_config`.
- Validation: `validate_deployment_config`, `validate_merged_config`, `s_allocation_config_errors`.
- Display views: `build_config_display_view`, `build_pipeline_config_view`.
- Provenance registry: `CONCERN_GROUPS`, `KEY_SPECS`, `KeySpec`, `LEGACY_KEY_TABLE`, `keys_with_exposure`, `keys_with_derivation`, `provenance_table`, `to_flat`, `to_namespaced`.
