"""Build ``platform_constraints_resolved`` from flat pipeline config."""

from __future__ import annotations

from typing import Any, cast

from mimarsinan.config_schema.defaults import DEFAULT_PLATFORM_CONSTRAINTS
from mimarsinan.mapping.platform.coalescing import CANONICAL_KEY, normalize_coalescing_config


def build_platform_constraints_resolved(
    pipeline_config: dict[str, Any],
    *,
    include_neuron_splitting: bool = True,
) -> dict[str, Any]:
    """Single source for resolved platform constraints dict."""
    cores = pipeline_config.get("cores")
    if cores is None:
        cores = list(cast("list[dict[str, Any]]", DEFAULT_PLATFORM_CONSTRAINTS["cores"]))

    global_has_bias = pipeline_config.get("platform_constraints", {}).get(
        "has_bias", True
    )
    cores = [dict(ct) for ct in cores]
    for ct in cores:
        ct.setdefault("has_bias", global_has_bias)

    pcfg: dict[str, Any] = {"cores": cores}
    if include_neuron_splitting:
        pcfg["allow_neuron_splitting"] = bool(
            pipeline_config.get("allow_neuron_splitting", False)
        )
    pcfg["allow_scheduling"] = bool(pipeline_config.get("allow_scheduling", False))

    if "target_tq" in pipeline_config:
        pcfg["target_tq"] = pipeline_config["target_tq"]
    if "weight_bits" in pipeline_config:
        pcfg["weight_bits"] = pipeline_config["weight_bits"]

    if CANONICAL_KEY in pipeline_config:
        pcfg[CANONICAL_KEY] = bool(pipeline_config[CANONICAL_KEY])
    else:
        pcfg[CANONICAL_KEY] = False
    normalize_coalescing_config(pcfg)
    return pcfg


def resolve_bias_mode(pipeline_config: dict[str, Any]) -> str:
    """Deployment bias delivery (``"on_chip"`` / ``"param_encoded"``) for this config.

    Single source shared by the tuners and the mapping step: reuses the same
    ``all(has_bias)`` resolution as ``SoftCoreMappingStep`` so training-time nodes and
    the deployed mapping agree on the declared mode.
    """
    from mimarsinan.mapping.platform.platform_constraints import (
        resolve_platform_mapping_params,
    )
    from mimarsinan.models.nn.activations.bias_mode import bias_mode_from_hardware_bias

    cores = build_platform_constraints_resolved(pipeline_config)["cores"]
    params = resolve_platform_mapping_params(cores)
    return bias_mode_from_hardware_bias(params.hardware_bias)


def resolve_wq_two_scale_projection(config: dict[str, Any]) -> bool:
    """Effective two-scale WQ flag: the ``wq_two_scale_projection`` key AND the
    platform's on-chip bias capability — a parameter-encoded bias rides the
    core matrix as an always-on axon row and must obey the ±q_max
    weight-register contract on the weight grid, so two-scale is not mappable
    there (``wq_cascade_crater_repair.md`` §5, backend audit)."""
    if not bool(config.get("wq_two_scale_projection", False)):
        return False
    return resolve_bias_mode(config) == "on_chip"
