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
