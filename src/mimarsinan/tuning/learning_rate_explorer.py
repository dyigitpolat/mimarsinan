"""Validation-accuracy-aware LR range search.

Each candidate LR is tested over ``steps_per_probe`` gradient steps (using
``trainer.train_n_steps``), then evaluated by **validation accuracy** (via
a caller-supplied ``validate_fn``).  This avoids the failure mode where a
high LR minimises training-batch loss by overfitting a single batch while
destroying generalisation -- the validation accuracy criterion rejects
such LRs automatically.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import torch

from mimarsinan.tuning.tuning_budget import TuningBudget


def clone_state_for_trainer(trainer) -> Any:
    """Snapshot trainable weights to CPU; supports BasicTrainer and aux-model trainers.

    Cloning to CPU keeps GPU memory free for training buffers (optimizer
    momentum, gradients, activations).  ``load_state_dict`` uses in-place
    ``copy_()`` which handles CPU→GPU transfer transparently on restore.
    """
    if hasattr(trainer, "aux_model"):
        return (
            {k: v.detach().clone().cpu() for k, v in trainer.aux_model.state_dict().items()},
            {k: v.detach().clone().cpu() for k, v in trainer.model.state_dict().items()},
        )
    return {k: v.detach().clone().cpu() for k, v in trainer.model.state_dict().items()}


def restore_state_for_trainer(trainer, state: Any) -> None:
    if isinstance(state, tuple):
        trainer.aux_model.load_state_dict(state[0])
        trainer.model.load_state_dict(state[1])
    else:
        trainer.model.load_state_dict(state)


class LRRangeFinder:
    """Exponential sweep selecting the largest non-destructive LR.

    The heuristic picks the highest LR whose validation accuracy does not
    drop below ``baseline - margin`` (where *margin* is typically the
    accuracy standard error from the tuning budget).  This maximises
    recovery speed while staying within the noise floor.
    """

    def __init__(
        self,
        *,
        trainer,
        clone_state: Callable[[], Any],
        restore_state: Callable[[Any], None],
        lr_min: float,
        lr_max: float,
        num_probes: int,
        steps_per_probe: int,
        validate_fn: Callable[[], float],
        max_total_steps: int | None = None,
        margin: float = 0.005,
    ):
        self.trainer = trainer
        self.clone_state = clone_state
        self.restore_state = restore_state
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        self.num_probes = max(2, int(num_probes))
        self.steps_per_probe = max(1, int(steps_per_probe))
        self.validate_fn = validate_fn
        self.max_total_steps = max_total_steps
        self.margin = float(margin)

    def find_best_lr(self) -> float:
        state = self.clone_state()
        try:
            baseline = float(self.validate_fn())

            accs: list[float] = []
            lrs: list[float] = []
            cumulative_steps = 0
            for i in range(self.num_probes):
                self.restore_state(state)
                lr = self.lr_min * (self.lr_max / self.lr_min) ** (
                    i / max(1, self.num_probes - 1)
                )
                self.trainer.train_n_steps(lr, self.steps_per_probe, constant_lr=True)
                cumulative_steps += self.steps_per_probe
                acc = float(self.validate_fn())
                accs.append(acc)
                lrs.append(float(lr))

                # Free optimizer/scaler memory from train_n_steps before next probe.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if acc < baseline * 0.1 and i > 0:
                    break
                if self.max_total_steps and cumulative_steps >= self.max_total_steps:
                    break

            threshold = baseline - self.margin
            non_destructive = [
                (lr, acc) for lr, acc in zip(lrs, accs) if acc >= threshold
            ]
            if non_destructive:
                return max(non_destructive, key=lambda x: x[0])[0]
            return max(zip(lrs, accs), key=lambda x: x[1])[0]
        finally:
            self.restore_state(state)



def find_lr_range_for_trainer(
    trainer,
    pipeline,
    budget: TuningBudget,
    *,
    validate_fn: Callable[[], float],
    anchor_lr: float | None = None,
) -> float:
    """Run :class:`LRRangeFinder` with budget-derived probe parameters.

    When *anchor_lr* is provided the sweep range is centred on that LR
    (one order of magnitude each direction) instead of spanning the full
    config range.  This keeps probes relevant when ``pipeline_lr`` is far
    from the default ``[1e-5, 1e-1]`` band (e.g. ImageNet at 1e-4).
    """
    cfg = pipeline.config
    if anchor_lr is not None:
        lr_min = anchor_lr / 100.0
        lr_max = anchor_lr * 10.0
    else:
        lr_min = float(cfg.get("lr_range_min", 1e-5))
        lr_max = float(cfg.get("lr_range_max", 1e-1))

    margin = budget.accuracy_se()

    return LRRangeFinder(
        trainer=trainer,
        clone_state=lambda: clone_state_for_trainer(trainer),
        restore_state=lambda s: restore_state_for_trainer(trainer, s),
        lr_min=lr_min,
        lr_max=lr_max,
        num_probes=budget.lr_num_probes,
        steps_per_probe=budget.lr_steps_per_probe,
        validate_fn=validate_fn,
        max_total_steps=budget.max_lr_exploration_steps,
        margin=margin,
    ).find_best_lr()
