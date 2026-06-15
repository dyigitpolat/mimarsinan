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

from mimarsinan.tuning.orchestration.tuning_budget import TuningBudget


def _clone(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def clone_state_for_trainer(trainer) -> Any:
    """Snapshot trainable weights on the model's device; supports aux-model trainers."""
    if hasattr(trainer, "aux_model"):
        return (_clone(trainer.aux_model), _clone(trainer.model))
    return _clone(trainer.model)


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
        coarse_signal: Callable[[], float] | None = None,
        coarse_top_k: int = 3,
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
        self.coarse_signal = coarse_signal
        self.coarse_top_k = max(1, int(coarse_top_k))

    def _probe_lr(self, i: int) -> float:
        return self.lr_min * (self.lr_max / self.lr_min) ** (
            i / max(1, self.num_probes - 1)
        )

    def _select(self, lrs, accs, baseline) -> float:
        threshold = baseline - self.margin
        non_destructive = [
            (lr, acc) for lr, acc in zip(lrs, accs) if acc >= threshold
        ]
        if non_destructive:
            return max(non_destructive, key=lambda x: x[0])[0]
        return max(zip(lrs, accs), key=lambda x: x[1])[0]

    def find_best_lr(self) -> float:
        if self.coarse_signal is not None:
            return self._find_best_lr_coarse()
        state = self.clone_state()
        try:
            baseline = float(self.validate_fn())

            accs: list[float] = []
            lrs: list[float] = []
            cumulative_steps = 0
            for i in range(self.num_probes):
                self.restore_state(state)
                lr = self._probe_lr(i)
                self.trainer.train_n_steps(lr, self.steps_per_probe, constant_lr=True)
                cumulative_steps += self.steps_per_probe
                acc = float(self.validate_fn())
                accs.append(acc)
                lrs.append(float(lr))

                if acc < baseline * 0.1 and i > 0:
                    break
                if self.max_total_steps and cumulative_steps >= self.max_total_steps:
                    break

            return self._select(lrs, accs, baseline)
        finally:
            self.restore_state(state)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _find_best_lr_coarse(self) -> float:
        """Cheap loss-slope coarse pass; full validation only for the top-K.

        Every probe is scored by ``coarse_signal`` (lower = better, a
        training-loss / loss-slope signal); only the ``coarse_top_k`` best
        candidates by that signal pay a full ``validate_fn`` call.  The
        final selection reuses the largest-non-destructive heuristic over
        the validated candidates.  Restore-after-probe and the lr range
        are identical to the full-validation path.
        """
        state = self.clone_state()
        try:
            baseline = float(self.validate_fn())

            signals: list[float] = []
            lrs: list[float] = []
            cumulative_steps = 0
            for i in range(self.num_probes):
                self.restore_state(state)
                lr = self._probe_lr(i)
                self.trainer.train_n_steps(lr, self.steps_per_probe, constant_lr=True)
                cumulative_steps += self.steps_per_probe
                signals.append(float(self.coarse_signal()))
                lrs.append(float(lr))
                if self.max_total_steps and cumulative_steps >= self.max_total_steps:
                    break

            order = sorted(range(len(lrs)), key=lambda j: signals[j])
            top = sorted(order[: self.coarse_top_k])

            fine_lrs: list[float] = []
            fine_accs: list[float] = []
            for j in top:
                self.restore_state(state)
                self.trainer.train_n_steps(
                    lrs[j], self.steps_per_probe, constant_lr=True
                )
                fine_lrs.append(lrs[j])
                fine_accs.append(float(self.validate_fn()))

            return self._select(fine_lrs, fine_accs, baseline)
        finally:
            self.restore_state(state)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()



def make_loss_slope_signal(trainer):
    """Cheap 'lower is better' coarse LR score: training-batch loss.

    Ranks LR probes by a single forward-pass loss (no validation sweep); reads one
    fresh training batch per call, never touches the test set. Returns ``None`` for
    trainers that cannot evaluate a training-batch loss (some stubs), so the LR
    finder falls back to full-validation scoring.
    """
    if not (hasattr(trainer, "evaluate_loss_on_batch")
            and hasattr(trainer, "next_training_batch")):
        return None

    def signal() -> float:
        return float(trainer.evaluate_loss_on_batch(trainer.next_training_batch()))

    return signal


def find_lr_range_for_trainer(
    trainer,
    pipeline,
    budget: TuningBudget,
    *,
    validate_fn: Callable[[], float],
    anchor_lr: float | None = None,
    coarse_signal: Callable[[], float] | None = None,
) -> float:
    """Run :class:`LRRangeFinder` with budget-derived probe parameters.

    When *anchor_lr* is provided the sweep range is centred on that LR
    (one order of magnitude each direction) instead of spanning the full
    config range.  This keeps probes relevant when ``pipeline_lr`` is far
    from the default ``[1e-5, 1e-1]`` band (e.g. ImageNet at 1e-4).

    When *coarse_signal* is provided the sweep scores every probe by that
    cheap signal and reserves full ``validate_fn`` scoring for the top
    candidates (see :meth:`LRRangeFinder._find_best_lr_coarse`); ``None``
    keeps the full-validation behavior unchanged.
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
        coarse_signal=coarse_signal,
    ).find_best_lr()
