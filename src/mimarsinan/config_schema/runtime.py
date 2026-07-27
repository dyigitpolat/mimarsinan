"""Build flat runtime config dicts for pipeline, wizard, and API preview."""

from __future__ import annotations

from typing import Dict, Optional

from mimarsinan.config_schema.defaults import (
    apply_preset,
    get_default_deployment_parameters,
    get_default_platform_constraints,
)
from mimarsinan.config_schema.deployment_derivation import (
    derive_deployment_parameters,
    derive_pipeline_runtime_parameters,
    derive_platform_constraints,
)


def build_flat_pipeline_config(
    deployment_parameters: Optional[Dict[str, object]] = None,
    platform_constraints: Optional[Dict[str, object]] = None,
    *,
    pipeline_mode: str = "vanilla",
) -> Dict[str, object]:
    """Merge defaults + overrides the same way ``DeploymentPipeline`` does (without device I/O)."""
    dp = dict(get_default_deployment_parameters())
    if deployment_parameters:
        dp.update(deployment_parameters)
    apply_preset(pipeline_mode, dp)
    dp.setdefault("pipeline_mode", pipeline_mode)

    pc = dict(get_default_platform_constraints())
    if platform_constraints:
        pc.update(platform_constraints)
    derive_platform_constraints(
        pc, cores_declared=bool(platform_constraints and "cores" in platform_constraints)
    )

    # Derive on the FLAT merge, exactly like ``merge_pipeline_config`` — the
    # derivation may consult platform keys (activation_bits arms mvm AQ).
    config: Dict[str, object] = {}
    config.update(dp)
    config.update(pc)
    derive_deployment_parameters(
        config,
        explicit_keys=set(deployment_parameters or {}) | set(platform_constraints or {}),
    )
    derive_pipeline_runtime_parameters(config)
    return config
