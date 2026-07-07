"""Step-based training APIs for :class:`BasicTrainer`."""

from __future__ import annotations

import warmup_scheduler
import torch

from mimarsinan.tuning.learning_rate_explorer import (
    clone_state_for_trainer,
    restore_state_for_trainer,
)


_TARGET_CONFIRM_FACTOR = 4
"""Confirmation-window multiple for a target-reach read: validation windows
are not difficulty-uniform, so one easy progress window must not end a stage
(measured: an armed 16k floor truncated by a single window read)."""


def _recipe_recovery_enabled(trainer) -> bool:
    """Whether the step recovery routes through ``tuning_recipe`` (warmup+cosine).

    Defaults ``False`` for duck-typed step trainers that lack the predicate.
    """
    predicate = getattr(trainer, "_recipe_step_recovery_enabled", None)
    return bool(predicate()) if callable(predicate) else False


def train_n_steps(
    trainer,
    lr,
    steps: int,
    warmup_steps: int = 0,
    *,
    constant_lr: bool = False,
    optimizer=None,
):
    owns_optimizer = optimizer is None
    if owns_optimizer:
        optimizer, scheduler, scaler = trainer._get_optimizer_and_scheduler_steps(lr, steps)
    else:
        scheduler, scaler = trainer._scheduler_and_scaler_for_optimizer(optimizer, lr, steps)
    if constant_lr:
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=0
        )
    if warmup_steps > 0:
        scheduler = warmup_scheduler.GradualWarmupScheduler(
            optimizer,
            multiplier=1.0,
            total_epoch=warmup_steps,
            after_scheduler=scheduler,
        )
    total = int(steps) + int(warmup_steps)
    for _ in range(total):
        try:
            x, y = next(trainer.train_iter)
        except StopIteration:
            trainer.train_iter = iter(trainer.train_loader)
            x, y = next(trainer.train_iter)
        x, y = x.to(trainer.device), y.to(trainer.device)
        trainer._optimize(x, y, optimizer, scaler)
        scheduler.step()
        trainer._report("LR", optimizer.param_groups[0]["lr"])
    del scheduler, scaler
    if owns_optimizer:
        del optimizer


def _rebuild_scheduler_at_reduced_lr(
    trainer, optimizer, reduced_lr, max_steps, *, factor=None
):
    """Drop the optimizer LR and rebuild its scheduler so the reduction sticks.

    A plateau reduction scales each group's peak; clearing ``initial_lr`` stops
    the rebuilt scheduler from re-stamping the stale peak and walking the LR back up.
    """
    use_recipe = _recipe_recovery_enabled(trainer)
    for group in optimizer.param_groups:
        if use_recipe and factor is not None:
            reduced_peak = float(group.get("initial_lr", group["lr"])) * float(factor)
            group["lr"] = reduced_peak
        else:
            group["lr"] = float(reduced_lr)
        group.pop("initial_lr", None)
    scheduler, _scaler = trainer._scheduler_and_scaler_for_optimizer(
        optimizer, reduced_lr, max_steps, constant_lr=True
    )
    return scheduler


def train_steps_until_target(
    trainer,
    lr,
    max_steps,
    target_accuracy,
    warmup_steps=0,
    *,
    validation_n_batches: int = 1,
    check_interval: int = 1,
    patience: int = 3,
    min_steps: int = 0,
    min_improvement: float = 1e-3,
    optimizer=None,
    plateau_lr_factor: float = 1.0,
    plateau_lr_reductions: int = 0,
    return_steps: bool = False,
    cosine_decay: bool = False,
    final_validation: bool = True,
):
    # Reproducibility contract: the budget is STEPS only — no wall-clock cap.
    # Identical configs train identical step counts on any hardware.
    recipe_recovery = _recipe_recovery_enabled(trainer)
    use_constant = (not cosine_decay) and not recipe_recovery
    owns_optimizer = optimizer is None
    if owns_optimizer:
        optimizer, scheduler, scaler = trainer._get_optimizer_and_scheduler_steps(
            lr, max_steps, constant_lr=use_constant
        )
    else:
        scheduler, scaler = trainer._scheduler_and_scaler_for_optimizer(
            optimizer, lr, max_steps, constant_lr=use_constant
        )
    if warmup_steps > 0:
        scheduler = warmup_scheduler.GradualWarmupScheduler(
            optimizer, multiplier=1.0, total_epoch=warmup_steps, after_scheduler=scheduler
        )
    total = int(max_steps) + int(warmup_steps)
    n_val = max(1, int(validation_n_batches))
    interval = max(1, int(check_interval))
    min_s = max(0, int(min_steps))
    imp_eps = float(min_improvement)
    plateau_factor = float(plateau_lr_factor)
    plateau_reductions_left = max(0, int(plateau_lr_reductions))
    plateau_enabled = plateau_factor < 1.0 and plateau_reductions_left > 0
    current_base_lr = float(lr)

    # [M-guard] keep-best anchors at the ENTRY state and metric, never 0.0: a
    # run that never beats entry restores entry exactly (full state_dict incl.
    # buffers), so a wrecked recovery becomes a no-op step, not a committed wreck.
    entry_state = clone_state_for_trainer(trainer)
    best_acc = float(trainer.validate_n_batches(n_val))
    best_state = entry_state
    stale_checks = 0
    steps_run = 0

    for step_idx in range(total):
        x, y = trainer.next_training_batch()
        x, y = x.to(trainer.device), y.to(trainer.device)
        trainer._optimize(x, y, optimizer, scaler)
        steps_run += 1
        scheduler.step()
        trainer._report("LR", optimizer.param_groups[0]["lr"])

        if (step_idx + 1) % interval == 0 or step_idx == total - 1:
            acc = trainer.validate_n_batches(n_val)
            if acc >= target_accuracy:
                # Confirm on a larger independent window before ending the
                # stage; a refuted reach continues with the better estimate.
                confirm = trainer.validate_n_batches(
                    _TARGET_CONFIRM_FACTOR * n_val
                )
                if confirm >= target_accuracy:
                    best_state = clone_state_for_trainer(trainer)
                    for _ in range(2):
                        x, y = trainer.next_training_batch()
                        x, y = x.to(trainer.device), y.to(trainer.device)
                        trainer._optimize(x, y, optimizer, scaler)
                        steps_run += 1
                        scheduler.step()
                    break
                acc = confirm
            if acc > best_acc + imp_eps:
                best_acc = acc
                best_state = clone_state_for_trainer(trainer)
                stale_checks = 0
            else:
                stale_checks += 1
                if step_idx + 1 >= min_s and stale_checks >= patience:
                    if plateau_enabled and plateau_reductions_left > 0:
                        current_base_lr *= plateau_factor
                        scheduler = _rebuild_scheduler_at_reduced_lr(
                            trainer, optimizer, current_base_lr, max_steps,
                            factor=plateau_factor,
                        )
                        plateau_reductions_left -= 1
                        stale_checks = 0
                        continue
                    break

    if best_state is not None:
        restore_state_for_trainer(trainer, best_state)
    del scheduler, scaler, best_state, entry_state
    if owns_optimizer:
        del optimizer
    # A4 eval consolidation: callers that re-measure with their own basis skip
    # the trailing eval; the keep-best restore above is unaffected.
    final_acc = trainer.validate_n_batches(n_val) if final_validation else None
    if return_steps:
        return final_acc, steps_run
    return final_acc


def train_one_step(
    trainer,
    lr,
    *,
    batch=None,
    eval_batch=None,
    return_post_update_loss: bool = False,
):
    optimizer, _, scaler = trainer._get_optimizer_and_scheduler(lr, epochs=0)
    if batch is None:
        x, y = trainer.next_training_batch()
    else:
        x, y = batch
    x, y = x.to(trainer.device), y.to(trainer.device)
    loss = trainer._optimize(x, y, optimizer, scaler)
    if return_post_update_loss:
        probe_batch = eval_batch if eval_batch is not None else (x, y)
        return trainer.evaluate_loss_on_batch(probe_batch)
    return float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)
