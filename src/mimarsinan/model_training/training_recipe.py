"""Composable training recipe: optimizer + scheduler + grad clip + LLRD."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

import torch


_BLOCK_PATTERNS = [
    re.compile(r"encoder_layer_(\d+)"),
    re.compile(r"(?:^|\.)blocks?\.(\d+)\."),
    re.compile(r"(?:^|\.)layers?\.(\d+)\."),
    re.compile(r"(?:^|\.)layer(\d+)(?:\.|_)"),
]


@dataclass(frozen=True)
class TrainingRecipe:
    """Declarative training recipe consumed by :class:`BasicTrainer`."""

    optimizer: str = "adam"
    weight_decay: float = 5e-5
    betas: Sequence[float] = (0.9, 0.99)
    momentum: float = 0.9
    scheduler: str = "cosine"
    warmup_ratio: float = 0.0
    grad_clip_norm: float = 0.0
    layer_wise_lr_decay: float = 1.0
    no_decay_keywords: Sequence[str] = (
        "bias",
        "norm",
        "pos_embed",
        "position_embedding",
        "class_token",
        "cls_token",
    )
    label_smoothing: Optional[float] = None


DEFAULT_RECIPE_FIELDS = {f.name for f in TrainingRecipe.__dataclass_fields__.values()}


def build_recipe(config: dict, key: str = "training_recipe") -> Optional[TrainingRecipe]:
    """Return a ``TrainingRecipe`` from ``config[key]``, or ``None`` for legacy behavior.

    Recipes are strictly opt-in with no cross-fallback: a fine-tuning recipe can
    destabilize rate-based SNN adaptation, so callers set each key explicitly.
    """
    block = config.get(key)
    if not block:
        return None
    if not isinstance(block, dict):
        raise TypeError(f"{key} must be a dict, got {type(block)!r}")

    kwargs = {k: v for k, v in block.items() if k in DEFAULT_RECIPE_FIELDS}
    if "betas" in kwargs and kwargs["betas"] is not None:
        kwargs["betas"] = tuple(kwargs["betas"])
    if "no_decay_keywords" in kwargs and kwargs["no_decay_keywords"] is not None:
        kwargs["no_decay_keywords"] = tuple(kwargs["no_decay_keywords"])
    return TrainingRecipe(**kwargs)


def _infer_block_depth(param_name: str) -> Optional[int]:
    for pattern in _BLOCK_PATTERNS:
        m = pattern.search(param_name)
        if m is not None:
            return int(m.group(1))
    return None


def _max_block_depth(model: torch.nn.Module) -> int:
    max_depth = -1
    for name, _ in model.named_parameters():
        depth = _infer_block_depth(name)
        if depth is not None and depth > max_depth:
            max_depth = depth
    return max_depth


def _is_no_decay(name: str, keywords: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(k in lowered for k in keywords)


def build_param_groups(
    model: torch.nn.Module,
    base_lr: float,
    recipe: TrainingRecipe,
) -> list[dict]:
    """Split model params into optimizer groups by weight-decay exclusion and layer-wise LR decay.

    Block depth is inferred from param names; params without a depth are treated as
    head (``base_lr``) unless they match stem/embedding keywords (deepest-decayed LR).
    """
    decay = recipe.weight_decay
    llrd = float(recipe.layer_wise_lr_decay)
    no_decay_keywords = tuple(recipe.no_decay_keywords)
    stem_keywords = ("conv_proj", "patch_embed", "patch_embedding", "stem", "cls_token", "class_token", "pos_embed", "position_embedding")

    max_depth = _max_block_depth(model) if llrd < 1.0 else -1
    use_llrd = llrd < 1.0 and max_depth >= 0

    groups: dict[tuple[float, bool], dict] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = _is_no_decay(name, no_decay_keywords)

        if use_llrd:
            depth = _infer_block_depth(name)
            if depth is not None:
                scale = llrd ** (max_depth - depth + 1)
            else:
                lowered = name.lower()
                if any(k in lowered for k in stem_keywords):
                    scale = llrd ** (max_depth + 2)
                else:
                    scale = 1.0
            lr = base_lr * scale
        else:
            lr = base_lr

        key = (lr, no_decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": lr,
                "weight_decay": 0.0 if no_decay else decay,
            }
        groups[key]["params"].append(param)

    return list(groups.values())


def build_optimizer(
    model: torch.nn.Module,
    base_lr: float,
    recipe: TrainingRecipe,
) -> torch.optim.Optimizer:
    param_groups = build_param_groups(model, base_lr, recipe)
    name = recipe.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(param_groups, lr=base_lr, betas=tuple(recipe.betas))
    if name == "adamw":
        return torch.optim.AdamW(param_groups, lr=base_lr, betas=tuple(recipe.betas))
    if name == "sgd":
        return torch.optim.SGD(param_groups, lr=base_lr, momentum=recipe.momentum)
    raise ValueError(f"Unknown optimizer: {recipe.optimizer!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    recipe: TrainingRecipe,
    total_steps: int,
) -> tuple[torch.optim.lr_scheduler._LRScheduler, int]:
    """Return ``(scheduler, warmup_steps)`` scheduled over ``total_steps``.

    ``scheduler.step()`` is expected to be called once per step unit that
    ``total_steps`` counts (epochs or minibatches -- caller's choice).
    """
    total = max(1, int(total_steps))
    warmup_steps = max(0, int(round(recipe.warmup_ratio * total)))

    scheduler_name = recipe.scheduler.lower()
    if scheduler_name == "constant":
        main = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
    elif scheduler_name == "cosine":
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total - warmup_steps), eta_min=0.0
        )
    else:
        raise ValueError(f"Unknown scheduler: {recipe.scheduler!r}")

    if warmup_steps <= 0:
        return main, 0

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
    )
    combined = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, main], milestones=[warmup_steps]
    )
    return combined, warmup_steps
