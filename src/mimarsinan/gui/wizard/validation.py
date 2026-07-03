"""Validate wizard state (deployment config shape) via config_schema.validate_deployment_config."""

from __future__ import annotations

from typing import Any, Dict, List

from mimarsinan.config_schema import validate_deployment_config


def validate_wizard_state(state: Dict[str, Any]) -> List[str]:
    """Validate wizard state (same shape as deployment config JSON); returns error messages (empty if valid)."""
    if not state:
        return ["Wizard state is empty"]
    return validate_deployment_config(state)
