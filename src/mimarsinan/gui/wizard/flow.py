"""
Wizard flow: step order and branching.

Steps are identified by string ids. The UI can use this to render the stepper
and decide which step is next/previous.

Two independent search toggles control branching:
- model_config_mode: "user" | "search"
- hw_config_mode: "fixed" | "search"

The search options step (nas_options) is shown whenever either toggle is set
to "search". The platform_constraints step is shown whenever HW is NOT being
searched (user provides fixed HW config).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


WIZARD_STEP_IDS: List[str] = [
    "experiment_basics",
    "pipeline_mode",
    "search_toggles",       # replaces old "configuration_mode"
    "user_model",           # model type + model_config
    "nas_options",          # search parameters (shown when any search is active)
    "platform_constraints", # HW config (shown when hw_config_mode != "search")
    "spiking_quantization",
    "training",
    "run_control",
    "review",
]


def _get_model_search(state: Dict[str, Any]) -> bool:
    dp = state.get("deployment_parameters") or {}
    return dp.get("model_config_mode", "user") == "search"


def _get_hw_search(state: Dict[str, Any]) -> bool:
    dp = state.get("deployment_parameters") or {}
    return dp.get("hw_config_mode", "fixed") == "search"


def _any_search_active(state: Dict[str, Any]) -> bool:
    return _get_model_search(state) or _get_hw_search(state)


def get_step_index(step_id: str) -> int:
    """Return the index of a step (0-based), or -1 if unknown."""
    try:
        return WIZARD_STEP_IDS.index(step_id)
    except ValueError:
        return -1


def get_next_step_id(state: Dict[str, Any], current_step_id: str) -> Optional[str]:
    """Return the next step id given current state and current step."""
    idx = get_step_index(current_step_id)
    if idx < 0 or idx >= len(WIZARD_STEP_IDS) - 1:
        return None

    if current_step_id == "search_toggles":
        # Always go to user_model (need to pick model type)
        return "user_model"

    if current_step_id == "user_model":
        if _any_search_active(state):
            return "nas_options"
        return "platform_constraints"

    if current_step_id == "nas_options":
        if _get_hw_search(state):
            # HW is being searched — skip platform_constraints
            return "spiking_quantization"
        return "platform_constraints"

    if current_step_id == "platform_constraints":
        return "spiking_quantization"

    return WIZARD_STEP_IDS[idx + 1]


def get_previous_step_id(state: Dict[str, Any], current_step_id: str) -> Optional[str]:
    """Return the previous step id (inverse of get_next_step_id)."""
    idx = get_step_index(current_step_id)
    if idx <= 0:
        return None

    if current_step_id == "spiking_quantization":
        if _get_hw_search(state):
            return "nas_options"
        return "platform_constraints"

    if current_step_id == "platform_constraints":
        if _any_search_active(state):
            return "nas_options"
        return "user_model"

    if current_step_id == "nas_options":
        return "user_model"

    if current_step_id == "user_model":
        return "search_toggles"

    return WIZARD_STEP_IDS[idx - 1]
