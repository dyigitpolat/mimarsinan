"""Shared model-repr extraction and layout verification for wizard API and snapshots."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

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
from mimarsinan.models.builders import BUILDERS_REGISTRY, build_model
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


def require_workload_fields(body: dict) -> tuple[str, tuple[int, ...], int]:
    """The workload identity of a layout request; every field must be explicit
    (the framework assumes no dataset or architecture)."""
    missing = [key for key in ("model_type", "input_shape", "num_classes") if body.get(key) is None]
    if missing:
        raise ValueError(
            f"layout request is missing workload fields {missing}: pass the "
            "provider/builder facts explicitly (no framework defaults exist)."
        )
    return (
        str(body["model_type"]),
        tuple(int(x) for x in body["input_shape"]),
        int(body["num_classes"]),
    )


def model_repr_from_wizard_body(body: dict) -> Any:
    """Build mapper repr from a wizard-style request body."""
    model_type, input_shape, num_classes = require_workload_fields(body)
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
    raw_model = build_model(builder, model_config, encoding_placement=placement)
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
        # Placement already resolved at flow birth by ``build_model``.
        model_repr = raw_model.get_mapper_repr()

    if hasattr(model_repr, "assign_perceptron_indices"):
        model_repr.assign_perceptron_indices()
    return model_repr


def materialise_lazy_parameters(model: Any, input_shape) -> None:
    """Give a lazy module the one batch it needs before its weights exist.

    A ``LazyBatchNorm`` that has never seen a forward carries uninitialized
    buffers, and reading its affine parameters raises — which a layout call
    would otherwise have to report as 'this model does not fit the chip'.
    Inert (and side-effect-free) for every model that has already run.
    """
    if input_shape is None:
        return
    uninitialized = any(
        isinstance(t, (UninitializedBuffer, UninitializedParameter))
        for t in list(model.parameters()) + list(model.buffers())
    )
    if not uninitialized:
        return
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.randn(2, *tuple(int(d) for d in input_shape)))
    finally:
        model.train(was_training)


def model_repr_from_model(
    model: Any,
    *,
    input_shape: tuple | list | None = None,
    num_classes: int | None = None,
    encoding_placement: str = "subsume",
) -> Any | None:
    """Extract mapper repr from a built model (native or torch); None when extraction fails.

    ``encoding_placement`` is the RUN's configured placement, and it applies only
    to the branch that has to birth a flow of its own (a torch module before
    conversion). A model that already carries a mapper graph carries the
    deployment's own resolved marking — read it, never re-resolve it.
    """
    model_repr = None
    with best_effort("snapshot mapper-repr extraction"):
        if hasattr(model, "get_mapper_repr"):
            materialise_lazy_parameters(model, input_shape)
            model_repr = model.get_mapper_repr()
        else:
            if input_shape is None or num_classes is None:
                return None
            supermodel = convert_torch_model(
                model,
                input_shape=tuple(input_shape),
                num_classes=int(num_classes),
                device="cpu",
                encoding_layer_placement=encoding_placement,
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
        **capabilities.layout_kwargs(),
    )
    stats_out: dict = plan.stats.to_dict()
    stats_out.setdefault("host_side_segment_count", plan.host_side_segment_count)
    stats_out.setdefault("layout_preview", plan.layout_preview)
    si = plan.schedule_info or {}
    if si.get("per_segment_passes"):
        stats_out["per_segment_passes"] = si["per_segment_passes"]
    stats_out["feasible"] = bool(plan.feasible)
    return stats_out
