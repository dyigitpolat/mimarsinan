"""
Wizard flow: step order and branching.

Steps are identified by string ids. The UI can use this to render the stepper
and decide which step is next/previous. Branching (user vs NAS) is expressed
by conditional steps.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Step ids in order (before branching)
WIZARD_STEP_IDS: List[str] = [
    "experiment_basics",
    "pipeline_mode",
    "configuration_mode",
    "user_model",       # user branch: model type + model_config
    "nas_options",      # nas branch: optimizer + arch_search
    "platform_constraints",
    "spiking_quantization",
    "training",
    "run_control",
    "review",
]


def get_step_index(step_id: str) -> int:
    """Return the index of a step (0-based), or -1 if unknown."""
    try:
        return WIZARD_STEP_IDS.index(step_id)
    except ValueError:
        return -1


def get_next_step_id(state: Dict[str, Any], current_step_id: str) -> str | None:
    """
    Return the next step id given current state and current step.

    Handles branching: after configuration_mode, user -> user_model, nas -> nas_options.
    From user_model we go to platform_constraints (skip nas_options); from nas_options to platform_constraints.
    """
    idx = get_step_index(current_step_id)
    if idx < 0 or idx >= len(WIZARD_STEP_IDS) - 1:
        return None
    config_mode = (state.get("deployment_parameters") or {}).get("configuration_mode", "user")
    if current_step_id == "configuration_mode":
        return "user_model" if config_mode == "user" else "nas_options"
    if current_step_id == "user_model" or current_step_id == "nas_options":
        return "platform_constraints"
    return WIZARD_STEP_IDS[idx + 1]


def get_previous_step_id(state: Dict[str, Any], current_step_id: str) -> str | None:
    """Return the previous step id (inverse of get_next_step_id)."""
    idx = get_step_index(current_step_id)
    if idx <= 0:
        return None
    config_mode = (state.get("deployment_parameters") or {}).get("configuration_mode", "user")
    if current_step_id == "platform_constraints":
        return "user_model" if config_mode == "user" else "nas_options"
    if current_step_id in ("user_model", "nas_options"):
        return "configuration_mode"
    return WIZARD_STEP_IDS[idx - 1]
