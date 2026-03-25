# config_schema/ -- Deployment Config Schema and Validation

Single source of truth for default deployment parameters, platform constraints,
pipeline-mode presets, and validation of deployment config (JSON for main.py and
merged flat config for pipeline runtime).

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `defaults.py` | `DEFAULT_DEPLOYMENT_PARAMETERS`, `DEFAULT_PLATFORM_CONSTRAINTS`, `PIPELINE_MODE_PRESETS`, `CONFIG_KEYS_SET`, `get_default_*`, `apply_preset` | Default values and preset merge logic. `allow_scheduling` (default False) lives in deployment parameters. Platform constraints include `max_schedule_passes` (default 8), `scheduling_latency_weight` (default 1.0) for scheduled mapping. |
| `validation.py` | `validate_deployment_config`, `validate_merged_config` | Validate JSON shape and merged flat config |
| `__init__.py` | Re-exports above | Public API |

## Dependencies

- **Internal**: None (no mimarsinan submodules).
- **External**: `typing`.

## Dependents

- `pipelining.pipelines.deployment_pipeline` uses `get_default_deployment_parameters`, `get_default_platform_constraints`, `apply_preset`.
- Wizard (application layer) will use schema, defaults, and validation.

## Exported API (__init__.py)

`DEFAULT_DEPLOYMENT_PARAMETERS`, `DEFAULT_PLATFORM_CONSTRAINTS`, `PIPELINE_MODE_PRESETS`, `CONFIG_KEYS_SET`, `get_default_deployment_parameters`, `get_default_platform_constraints`, `get_pipeline_mode_presets`, `get_config_keys_set`, `apply_preset`, `validate_deployment_config`, `validate_merged_config`.
