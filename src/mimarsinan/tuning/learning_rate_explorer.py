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

from mimarsinan.tuning.tuning_budget import TuningBudget


def clone_state_for_trainer(trainer) -> Any:
    """Snapshot trainable weights; supports BasicTrainer and aux-model trainers."""
    if hasattr(trainer, "aux_model"):
        return (
            copy.deepcopy(trainer.aux_model.state_dict()),
            copy.deepcopy(trainer.model.state_dict()),
        )
    return copy.deepcopy(trainer.model.state_dict())


def restore_state_for_trainer(trainer, state: Any) -> None:
    if isinstance(state, tuple):
        trainer.aux_model.load_state_dict(state[0])
        trainer.model.load_state_dict(state[1])
    else:
        trainer.model.load_state_dict(state)


class LRRangeFinder:
    """Exponential sweep selecting the LR with the best validation accuracy."""

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
    ):
        self.trainer = trainer
        self.clone_state = clone_state
        self.restore_state = restore_state
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        self.num_probes = max(2, int(num_probes))
        self.steps_per_probe = max(1, int(steps_per_probe))
        self.validate_fn = validate_fn

    def find_best_lr(self) -> float:
        state = self.clone_state()
        try:
            baseline = float(self.validate_fn())

            accs: list[float] = []
            lrs: list[float] = []
            for i in range(self.num_probes):
                self.restore_state(state)
                lr = self.lr_min * (self.lr_max / self.lr_min) ** (
                    i / max(1, self.num_probes - 1)
                )
                self.trainer.train_n_steps(lr, self.steps_per_probe)
                acc = float(self.validate_fn())
                accs.append(acc)
                lrs.append(float(lr))

            sm = _smooth(accs)
            best_val = max(sm)
            worst_val = min(sm)

            if baseline > 1e-6 and best_val < baseline * 0.9:
                return lrs[0]

            if best_val - worst_val < 1e-4:
                return (self.lr_min * self.lr_max) ** 0.5

            best_i = max(range(len(sm)), key=lambda j: sm[j])
            return lrs[best_i]
        finally:
            self.restore_state(state)


def _smooth(values: list[float], window: int = 5) -> list[float]:
    if not values:
        return []
    w = max(1, min(window, len(values)))
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - w // 2)
        hi = min(len(values), lo + w)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def find_lr_range_for_trainer(
    trainer, pipeline, budget: TuningBudget, *, validate_fn: Callable[[], float]
) -> float:
    """Run :class:`LRRangeFinder` with budget-derived probe parameters."""
    cfg = pipeline.config
    lr_min = float(cfg.get("lr_range_min", 1e-5))
    lr_max = float(cfg.get("lr_range_max", 1e-1))
    return LRRangeFinder(
        trainer=trainer,
        clone_state=lambda: clone_state_for_trainer(trainer),
        restore_state=lambda s: restore_state_for_trainer(trainer, s),
        lr_min=lr_min,
        lr_max=lr_max,
        num_probes=budget.lr_num_probes,
        steps_per_probe=budget.lr_steps_per_probe,
        validate_fn=validate_fn,
    ).find_best_lr()
