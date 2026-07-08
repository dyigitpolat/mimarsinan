"""Shared helpers for adapting torchvision image-model builders."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from mimarsinan.common.pretrained import PretrainedWeightSet

TORCHVISION_SOURCE = "torchvision"
# torchvision ships its weights under its own licence unless a weight declares one.
TORCHVISION_LICENSE = "BSD-3-Clause (torchvision)"
IMAGE_CLASSIFICATION_TASK = "image classification"


def torchvision_weight_set(member: Any, **overrides: Any) -> PretrainedWeightSet:
    """One registered weight set, with EVERY fact read from the torchvision
    weight enum — no number is hand-copied, so none can drift.

    ``adapts_*`` defaults to True: these builders project the patch/stem conv
    onto the provider's channel count and the loading strategy skips the
    shape-mismatched head, so a geometry or class-count difference is an
    adaptation the builder implements, not an incompatibility.
    """
    meta = dict(member.meta)
    metrics = meta.get("_metrics") or {}
    dataset = next(iter(metrics), "unknown")
    top1 = (metrics.get(dataset) or {}).get("acc@1")
    transform = member.transforms()
    crop = int(transform.crop_size[0])
    facts: dict[str, Any] = {
        "id": member.name.lower(),
        "label": f"{dataset} · {member.name}",
        "task": IMAGE_CLASSIFICATION_TASK,
        "dataset": dataset,
        "input_shape": (3, crop, crop),
        "num_classes": len(meta.get("categories") or ()),
        "source": TORCHVISION_SOURCE,
        "expected_accuracy": None if top1 is None else float(top1) / 100.0,
        "license": meta.get("license") or TORCHVISION_LICENSE,
        "num_parameters": meta.get("num_params"),
        "recipe": meta.get("recipe"),
        "preprocessing": {
            "resize_to": int(transform.resize_size[0]),
            "crop_to": crop,
            "mean": [float(x) for x in transform.mean],
            "std": [float(x) for x in transform.std],
            "interpolation": str(transform.interpolation.value),
        },
        "adapts_input_shape": True,
        "adapts_num_classes": True,
    }
    facts.update(overrides)
    return PretrainedWeightSet(**facts)


def registered_weight_set(builder_cls: Any, weight_set_id: str | None) -> PretrainedWeightSet:
    """The set a weight-set id names, judged against what the BUILDER registered;
    ``None`` selects the builder's default (its first registration)."""
    declared = builder_cls.workload_profile().pretrained_weight_sets
    if weight_set_id is None:
        return declared[0]
    for weight_set in declared:
        if weight_set.id == str(weight_set_id):
            return weight_set
    raise ValueError(
        f"pretrained weight set {weight_set_id!r} is not registered for "
        f"{builder_cls.__name__}; registered: {[w.id for w in declared]}"
    )


def torchvision_weights(builder_cls: Any, weights_enum: Any, weight_set_id: str | None) -> Any:
    """The torchvision weight enum member of a registered weight set."""
    return weights_enum[registered_weight_set(builder_cls, weight_set_id).id.upper()]


def parse_image_input_shape(input_shape, *, model_name: str) -> tuple[int, int, int]:
    """Validate and normalize a ``(C, H, W)`` input shape."""
    shape = tuple(int(x) for x in input_shape)
    if len(shape) != 3:
        raise ValueError(f"{model_name} expects input_shape (C, H, W), got {shape!r}")
    c, h, w = shape
    if c <= 0 or h <= 0 or w <= 0:
        raise ValueError(f"{model_name} needs positive input dimensions, got {shape!r}")
    return c, h, w


def resize_conv_input_weights(weight: torch.Tensor, in_channels: int) -> torch.Tensor:
    """Project pretrained conv weights to a new input-channel count."""
    if weight.shape[1] == in_channels:
        return weight.detach().clone()
    if in_channels == 1:
        return weight.mean(dim=1, keepdim=True)
    if in_channels < weight.shape[1]:
        return weight[:, :in_channels].detach().clone()
    extra = in_channels - weight.shape[1]
    mean = weight.mean(dim=1, keepdim=True).repeat(1, extra, 1, 1)
    return torch.cat([weight.detach().clone(), mean], dim=1)


def adapt_conv_in_channels(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Clone ``conv`` with a new input-channel count and projected weights."""
    if conv.in_channels == in_channels:
        return conv
    # Conv2d stores its 2-tuple geometry as tuple[int, ...]; cast back to the ctor's _size_2_t.
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=cast("tuple[int, int]", conv.kernel_size),
        stride=cast("tuple[int, int]", conv.stride),
        padding=cast("str | tuple[int, int]", conv.padding),
        dilation=cast("tuple[int, int]", conv.dilation),
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight.copy_(resize_conv_input_weights(conv.weight, in_channels))
        if conv.bias is not None:
            assert new_conv.bias is not None
            new_conv.bias.copy_(conv.bias)
    return new_conv
