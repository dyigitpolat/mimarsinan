"""
Configuration wizard application layer.

Builds deployment config JSON from wizard state and validates state/config.
Wizard state mirrors the deployment config shape (top-level + deployment_parameters + platform_constraints).
"""

from mimarsinan.gui.wizard.config_builder import build_deployment_config_from_state
from mimarsinan.gui.wizard.validation import validate_wizard_state

__all__ = [
    "build_deployment_config_from_state",
    "validate_wizard_state",
]
