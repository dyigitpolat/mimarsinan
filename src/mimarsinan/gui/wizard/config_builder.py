"""Build minimal deployment config JSON from wizard state."""

from __future__ import annotations

from typing import Any, Dict

from mimarsinan.config_schema.namespaced_schema import KEY_SPECS, keys_with_exposure


_EXTRA_USER_DEPLOYMENT_KEYS = frozenset({
    "model_type",
    "model_config",
    "arch_search",
    "encoding_layer_placement",
    "negative_value_shift",
    "pruning",
    "pruning_fraction",
    "weight_source",
    "finetune_epochs",
    "finetune_lr",
    "batch_size",
    "preprocessing",
    "max_simulation_samples",
})

_EXTRA_USER_PLATFORM_KEYS = frozenset({
    "max_axons",
    "max_neurons",
    "has_bias",
    "search_space",
    "mode",
    "user",
    "auto",
    "fixed",
})


def _is_known_exposure(key: str, *exposures: str) -> bool:
    spec = KEY_SPECS.get(key)
    return spec is not None and spec.exposure in exposures


def _minimal_deployment_parameters(raw: Dict[str, Any]) -> Dict[str, Any]:
    user_keys = set(keys_with_exposure("user")) | set(_EXTRA_USER_DEPLOYMENT_KEYS)
    hidden_keys = (
        set(keys_with_exposure("derived"))
        | set(keys_with_exposure("runtime"))
        | set(keys_with_exposure("system"))
    )
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in user_keys:
            out[key] = value
        elif key not in hidden_keys:
            out[key] = value
    return out


def _minimal_platform_constraints(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    user_keys = set(keys_with_exposure("user")) | set(_EXTRA_USER_PLATFORM_KEYS)
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in {"user", "fixed"} and isinstance(value, dict):
            nested = _minimal_platform_constraints(value)
            if nested:
                out[key] = nested
        elif key == "auto" and isinstance(value, dict):
            auto: Dict[str, Any] = {}
            fixed = value.get("fixed")
            if isinstance(fixed, dict):
                fixed_min = _minimal_platform_constraints(fixed)
                if fixed_min:
                    auto["fixed"] = fixed_min
            if isinstance(value.get("search_space"), dict):
                auto["search_space"] = dict(value["search_space"])
            for auto_key, auto_value in value.items():
                if auto_key not in {"fixed", "search_space"}:
                    auto[auto_key] = auto_value
            if auto:
                out[key] = auto
        elif key in user_keys:
            out[key] = value
        elif not _is_known_exposure(key, "system", "derived", "runtime"):
            out[key] = value
    return out


def build_deployment_config_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build a deployment config dict from wizard state (same shape as the deployment JSON).

    Keeps user-facing knobs only; runtime/derived fields are resolved later.
    """
    state = state or {}
    out: Dict[str, Any] = {}

    out["data_provider_name"] = state.get("data_provider_name", "MNIST_DataProvider")
    out["experiment_name"] = state.get("experiment_name", "experiment")
    out["generated_files_path"] = state.get("generated_files_path", "./generated")
    out["seed"] = state.get("seed", 0)
    out["start_step"] = state.get("start_step")
    if "stop_step" in state:
        out["stop_step"] = state.get("stop_step")
    if "target_metric_override" in state:
        out["target_metric_override"] = state.get("target_metric_override")

    out["deployment_parameters"] = _minimal_deployment_parameters(
        dict(state.get("deployment_parameters") or {})
    )
    out["platform_constraints"] = _minimal_platform_constraints(
        state.get("platform_constraints") or {}
    )

    continue_from = state.get("_continue_from_run_id")
    if continue_from:
        out["_continue_from_run_id"] = continue_from

    return out
