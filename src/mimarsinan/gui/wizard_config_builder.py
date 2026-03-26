"""
Build deployment config JSON from wizard state.

Wizard state uses two independent search toggles:
- ``deployment_parameters.model_config_mode``: ``"user"`` | ``"search"``
- ``deployment_parameters.hw_config_mode``:    ``"fixed"`` | ``"search"``

When hw_config_mode is ``"search"``, the platform_constraints dict may contain
a ``search_space`` sub-dict with HW search bounds.
"""

from __future__ import annotations

from typing import Any, Dict

from mimarsinan.config_schema import (
    get_default_deployment_parameters,
    get_default_platform_constraints,
    apply_preset,
)


def build_deployment_config_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a deployment config dict from wizard state (same shape as main.py JSON).

    - Applies pipeline_mode preset to deployment_parameters (setdefault).
    - Fills missing top-level keys with defaults where applicable.
    - Does not mutate state; returns a new dict.

    The returned dict can be serialized to JSON and passed to main.py.
    """
    state = state or {}
    out: Dict[str, Any] = {}

    out["data_provider_name"] = state.get("data_provider_name", "MNIST_DataProvider")
    out["experiment_name"] = state.get("experiment_name", "experiment")
    out["generated_files_path"] = state.get("generated_files_path", "./generated")
    out["seed"] = state.get("seed", 0)
    out["pipeline_mode"] = state.get("pipeline_mode", "phased")
    out["start_step"] = state.get("start_step")
    out["stop_step"] = state.get("stop_step")
    out["target_metric_override"] = state.get("target_metric_override")

    dp_defaults = get_default_deployment_parameters()
    dp = dict(state.get("deployment_parameters") or {})
    for k, v in dp_defaults.items():
        dp.setdefault(k, v)
    apply_preset(out["pipeline_mode"], dp)

    # Ensure the two search toggles have defaults
    dp.setdefault("model_config_mode", "user")
    dp.setdefault("hw_config_mode", "fixed")

    out["deployment_parameters"] = dp

    pc = state.get("platform_constraints")
    if pc is None:
        pc = dict(get_default_platform_constraints())
    elif isinstance(pc, dict):
        pc = dict(pc)
        for k, v in get_default_platform_constraints().items():
            pc.setdefault(k, v)
    else:
        pc = dict(pc)
    out["platform_constraints"] = pc

    return out
