"""Frozen-teacher snapshots for knowledge-distillation recovery."""

from __future__ import annotations

import copy

import torch.nn as nn


def freeze_module(module: nn.Module) -> nn.Module:
    """Put ``module`` in eval mode and disable grad on every parameter."""
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


REFERENCE_TEACHER_CACHE_SUFFIX = ".reference_teacher_model"


def origin_teacher_kd_active(config) -> bool:
    """The pipeline-wide origin-teacher KD lever (calculus §13.2 L-A)."""
    return bool(config.get("origin_teacher_kd", False))


def find_reference_teacher(pipeline) -> nn.Module | None:
    """The cached ORIGIN (reference) teacher, or None. Key-suffix scan: the
    producing step owns the namespaced cache key."""
    for key in pipeline.cache.keys():
        if str(key).endswith(REFERENCE_TEACHER_CACHE_SUFFIX):
            return pipeline.cache.get(key)
    return None


def resolve_conversion_teacher(pipeline, model: nn.Module) -> nn.Module:
    """The KD anchor for a conversion tuner: the frozen ORIGIN teacher when
    the lever is armed (fail-loud when the snapshot step never cached one —
    a silent per-step fallback would defeat the anchor), else a frozen
    snapshot of ``model`` (the historical per-step anchor)."""
    device = pipeline.config["device"]
    if origin_teacher_kd_active(pipeline.config):
        teacher = find_reference_teacher(pipeline)
        if teacher is None:
            raise RuntimeError(
                "origin_teacher_kd is armed but no cached reference teacher "
                "exists; the 'Reference Teacher Snapshot' step must run "
                "before the first conversion tuner (resume from it, or drop "
                "start_step past it)."
            )
        teacher.to(device)
        return freeze_module(teacher)
    return snapshot_frozen_teacher(model, device)


def snapshot_frozen_teacher(model: nn.Module, device) -> nn.Module:
    """Deep-copy ``model`` into a frozen, eval-mode teacher on ``device``.

    The deepcopy runs on CPU (the model is moved there and back) so a large model
    need not fit twice in accelerator memory during the copy.
    """
    model.to("cpu")
    teacher = copy.deepcopy(model)
    model.to(device)
    teacher.to(device)
    return freeze_module(teacher)
