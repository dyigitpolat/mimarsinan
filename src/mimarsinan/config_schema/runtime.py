"""Build flat runtime config dicts for pipeline, wizard, and API preview."""

from __future__ import annotations

from typing import Any, Dict, Optional

from mimarsinan.config_schema.defaults import (
    apply_preset,
    get_default_deployment_parameters,
    get_default_platform_constraints,
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

    pc = dict(get_default_platform_constraints())
    if platform_constraints:
        pc.update(platform_constraints)

    config: Dict[str, object] = {}
    config.update(dp)
    config.update(pc)
    return config
