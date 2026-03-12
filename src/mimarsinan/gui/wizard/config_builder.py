"""
Build deployment config JSON from wizard state.

Wizard state has the same shape as the deployment config (main.py input).
The builder applies defaults and pipeline_mode preset so the output is valid
and can be saved to a file and passed to main.py.
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

    # Top-level
    out["data_provider_name"] = state.get("data_provider_name", "MNIST_DataProvider")
    out["experiment_name"] = state.get("experiment_name", "experiment")
    out["generated_files_path"] = state.get("generated_files_path", "./generated")
    out["seed"] = state.get("seed", 0)
    out["pipeline_mode"] = state.get("pipeline_mode", "phased")
    out["start_step"] = state.get("start_step")
    out["stop_step"] = state.get("stop_step")
    out["target_metric_override"] = state.get("target_metric_override")

    # deployment_parameters: merge defaults, then state, then apply preset
    dp_defaults = get_default_deployment_parameters()
    dp = dict(state.get("deployment_parameters") or {})
    for k, v in dp_defaults.items():
        dp.setdefault(k, v)
    apply_preset(out["pipeline_mode"], dp)
    out["deployment_parameters"] = dp

    # platform_constraints: pass through (user or {mode, user, auto}); optionally fill defaults for flat user
    pc = state.get("platform_constraints")
    if pc is None:
        pc = dict(get_default_platform_constraints())
    elif isinstance(pc, dict):
        pc = dict(pc)
        if "mode" not in pc:
            for k, v in get_default_platform_constraints().items():
                pc.setdefault(k, v)
        elif pc.get("mode") == "user":
            # Ensure "user" key exists for main.py: copy flat keys into user if missing
            if "user" not in pc or not isinstance(pc.get("user"), dict):
                user = {k: v for k, v in pc.items() if k != "mode"}
                for k, v in get_default_platform_constraints().items():
                    user.setdefault(k, v)
                pc["user"] = user
        elif pc.get("mode") == "auto":
            auto = pc.get("auto")
            if isinstance(auto, dict):
                if "fixed" not in auto or not isinstance(auto.get("fixed"), dict):
                    auto["fixed"] = dict(get_default_platform_constraints())
                # search_space left as-is from state
            else:
                pc["auto"] = {"fixed": dict(get_default_platform_constraints()), "search_space": {}}
    else:
        pc = dict(pc)
    out["platform_constraints"] = pc

    return out
