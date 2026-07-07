"""Build deployment config JSON from wizard state (explicit-keys-only emission)."""

from __future__ import annotations

from typing import Any, Dict

from mimarsinan.gui.wizard.emit import emit_deployment_config


def build_deployment_config_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build a deployment config dict from wizard state (same shape as the deployment JSON).

    Explicit keys survive verbatim — including schema-unknown ones (never
    silently dropped); only runtime keys and the non-declarable derived
    ``activation_quantization`` are removed. Missing run-identity keys are
    filled from schema defaults.
    """
    return emit_deployment_config(state or {})
