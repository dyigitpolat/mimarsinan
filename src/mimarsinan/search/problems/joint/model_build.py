"""Candidate model construction for joint search (build, warm up, convert)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch.nn.parameter import UninitializedParameter

from mimarsinan.models.builders import build_model
from mimarsinan.torch_mapping.converter import convert_torch_model


def build_raw_model(
    *,
    builder_factory: Any,
    device: Any,
    input_shape: Tuple[int, ...],
    num_classes: int,
    target_tq: int,
    model_config: Dict,
    pcfg: Dict,
    placement: str,
) -> Tuple[Any, float]:
    """Build and warm up a raw model. Returns (model, total_params) or raises."""
    builder = builder_factory(
        device, input_shape, num_classes, {**pcfg, "target_tq": int(target_tq)},
    )
    model = build_model(
        builder, model_config, encoding_placement=placement
    ).to(device)

    model.eval()
    with torch.no_grad():
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = device
        dummy = torch.zeros((1, *tuple(input_shape)), device=model_device)
        _ = model(dummy)

    if any(isinstance(p, UninitializedParameter) for p in model.parameters()):
        raise RuntimeError("Model has uninitialised parameters after forward pass")

    total_params = float(sum(int(p.numel()) for p in model.parameters()))
    return model, total_params


def convert_to_mapper_repr(
    model: Any,
    *,
    input_shape: Tuple[int, ...],
    num_classes: int,
    device: Any,
    target_tq: int,
    placement: str,
) -> Any:
    """Convert via torch mapping if the model lacks ``get_mapper_repr``.

    A native builder's flow already had its placement resolved by
    ``build_model``; a torch module's flow is born here and resolves the
    same one, so a candidate's core count is the deployed model's.
    """
    if hasattr(model, "get_mapper_repr"):
        return model
    return convert_torch_model(
        model,
        input_shape=tuple(input_shape),
        num_classes=num_classes,
        device=device,
        Tq=target_tq,
        encoding_layer_placement=placement,
    )
