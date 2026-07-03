"""Shared model-repr extraction and layout verification for wizard API and snapshots."""

from __future__ import annotations

from typing import Any

import torch

from mimarsinan.common.best_effort import best_effort
from mimarsinan.mapping.verification.verifier import (
    MappingVerificationResult,
    verify_soft_core_mapping,
)
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.platform.platform_constraints import (
    resolve_platform_mapping_params,
    resolve_scalar_mapping_params,
)
from mimarsinan.mapping.layout.layout_plan import build_layout_plan
from mimarsinan.models.builders import BUILDERS_REGISTRY
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


def model_repr_from_wizard_body(body: dict) -> Any:
    """Build mapper repr from a wizard-style request body."""
    model_type = body.get("model_type", "simple_mlp")
    input_shape = tuple(int(x) for x in body.get("input_shape", [1, 28, 28]))
    num_classes = int(body.get("num_classes", 10))
    model_config = body.get("model_config", {})
    placement = str(body.get("encoding_layer_placement", "subsume"))
    pipeline_config = {
        "target_tq": int(body.get("target_tq", 32)),
        "device": "cpu",
    }

    builder_cls = BUILDERS_REGISTRY.get(model_type)
    if builder_cls is None:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    builder = builder_cls(
        device=torch.device("cpu"),
        input_shape=input_shape,
        num_classes=num_classes,
        pipeline_config=pipeline_config,
    )
    raw_model = builder.build(model_config)
    category = ModelRegistry.get_category(model_type)

    if category == "torch":
        raw_model.eval()
        with torch.no_grad(), best_effort("wizard torch-model warm-up forward"):
            raw_model(torch.randn(1, *input_shape))
        supermodel = convert_torch_model(
            raw_model,
            input_shape=input_shape,
            num_classes=num_classes,
            device="cpu",
            encoding_layer_placement=placement,
        )
        model_repr = supermodel.get_mapper_repr()
    else:
        raw_model.eval()
        with torch.no_grad(), best_effort("wizard native-model warm-up forward"):
            raw_model(torch.randn(2, *input_shape))
        model_repr = raw_model.get_mapper_repr()
        mark_encoding_layers(model_repr, placement=placement)

    if hasattr(model_repr, "assign_perceptron_indices"):
        model_repr.assign_perceptron_indices()
    return model_repr


def model_repr_from_model(
    model: Any,
    *,
    input_shape: tuple | list | None = None,
    num_classes: int = 10,
) -> Any | None:
    """Extract mapper repr from a built model (native or torch); None when extraction fails."""
    model_repr = None
    with best_effort("snapshot mapper-repr extraction"):
        if hasattr(model, "get_mapper_repr"):
            model_repr = model.get_mapper_repr()
        else:
            if input_shape is None:
                return None
            supermodel = convert_torch_model(
                model,
                input_shape=tuple(input_shape),
                num_classes=int(num_classes),
                device="cpu",
            )
            model_repr = supermodel.get_mapper_repr()
    if model_repr is None:
        return None
    if hasattr(model_repr, "assign_perceptron_indices"):
        with best_effort("snapshot perceptron-index assignment"):
            model_repr.assign_perceptron_indices()
    return model_repr


def resolve_tiling_params_from_body(
    body: dict,
    *,
    tiling_max_axons: int | None = None,
    tiling_max_neurons: int | None = None,
):
    """Return ``(effective_max_axons, effective_max_neurons, hardware_bias, allow_coalescing)``."""
    allow_coalescing = bool(body.get("allow_coalescing", False))
    core_types = body.get("core_types") or body.get("cores")
    if core_types:
        pmap = resolve_platform_mapping_params(
            core_types, allow_coalescing=allow_coalescing
        )
        return (
            pmap.effective_max_axons,
            pmap.effective_max_neurons,
            pmap.hardware_bias,
            pmap.allow_coalescing,
        )
    scalar = resolve_scalar_mapping_params(
        max_axons=int(
            tiling_max_axons if tiling_max_axons is not None else body.get("max_axons", 1024)
        ),
        max_neurons=int(
            tiling_max_neurons if tiling_max_neurons is not None else body.get("max_neurons", 1024)
        ),
        hardware_bias=bool(body.get("hardware_bias", False)),
        allow_coalescing=allow_coalescing,
    )
    return (
        scalar.effective_max_axons,
        scalar.effective_max_neurons,
        scalar.hardware_bias,
        scalar.allow_coalescing,
    )


def verify_layout_for_model_repr(
    model_repr,
    *,
    max_axons: int,
    max_neurons: int,
    allow_coalescing: bool = False,
    hardware_bias: bool = False,
) -> MappingVerificationResult:
    return verify_soft_core_mapping(
        model_repr,
        max_axons=max_axons,
        max_neurons=max_neurons,
        allow_coalescing=allow_coalescing,
        hardware_bias=hardware_bias,
    )


def verify_planned_mapping_performance(
    model_repr,
    platform_constraints: dict,
) -> dict | None:
    """Wizard-shaped mapping performance dict for snapshot panel."""
    cores = platform_constraints.get("cores") or []
    if not cores or model_repr is None:
        return None

    capabilities = ChipCapabilities.from_platform_constraints(platform_constraints)
    pmap = resolve_platform_mapping_params(
        cores, allow_coalescing=capabilities.allow_coalescing
    )
    if pmap.effective_max_axons <= 0 or pmap.effective_max_neurons <= 0:
        return None

    soft = verify_layout_for_model_repr(
        model_repr,
        max_axons=pmap.effective_max_axons,
        max_neurons=pmap.effective_max_neurons,
        allow_coalescing=pmap.allow_coalescing,
        hardware_bias=pmap.hardware_bias,
    )
    if not soft.feasible:
        return {"feasible": False}

    core_types_dicts = [
        {
            "max_axons": int(ct.get("max_axons", 0)),
            "max_neurons": int(ct.get("max_neurons", 0)),
            "count": int(ct.get("count", 0)),
        }
        for ct in cores
    ]
    plan = build_layout_plan(
        soft,
        core_types_dicts,
        **capabilities.permission_kwargs(),
    )
    stats_out: dict = plan.stats.to_dict()
    stats_out.setdefault("host_side_segment_count", plan.host_side_segment_count)
    stats_out.setdefault("layout_preview", plan.layout_preview)
    si = plan.schedule_info or {}
    if si.get("per_segment_passes"):
        stats_out["per_segment_passes"] = si["per_segment_passes"]
    stats_out["feasible"] = bool(plan.feasible)
    return stats_out
