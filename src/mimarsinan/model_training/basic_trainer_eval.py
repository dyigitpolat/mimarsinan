"""Evaluation and validation helpers for :class:`BasicTrainer`."""

from __future__ import annotations

import contextlib

import torch


def _eval_autocast(device):
    if torch.device(device).type == "cuda":
        return torch.amp.autocast("cuda")
    return contextlib.nullcontext()


def _build_gpu_val_cache(trainer):
    # ``.clone()`` is load-bearing: FFCV's IndexedLoader yields views into a
    # small rotating buffer pool, so aliased references silently corrupt
    # cached batches once the pool wraps.
    trainer._gpu_val_cache = [
        (x.to(trainer.device, non_blocking=True).clone(),
         y.to(trainer.device, non_blocking=True).clone())
        for x, y in trainer.validation_loader
    ]
    trainer._gpu_val_cursor = 0


def iter_validation_batches(trainer, n_batches: int):
    if getattr(trainer, "_gpu_val_cache", None) is None:
        _build_gpu_val_cache(trainer)
    cache = trainer._gpu_val_cache
    if not cache:
        return
    for _ in range(n_batches):
        yield cache[trainer._gpu_val_cursor % len(cache)]
        trainer._gpu_val_cursor += 1


def test(trainer, max_batches: int | None = None):
    total = 0
    correct = 0
    with torch.no_grad(), _eval_autocast(trainer.device):
        for batch_idx, (x, y) in enumerate(trainer.test_loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            trainer.model.eval()
            trainer.model = trainer.model.to(trainer.device)
            x, y = x.to(trainer.device), y.to(trainer.device)
            _, predicted = trainer.model(x).max(1)
            total += float(y.size(0))
            correct += float(predicted.eq(y).sum().item())

    if total <= 0:
        return 0.0
    acc = correct / total
    trainer._report("Test accuracy", acc)
    return acc


def validate_on_loader(trainer, x, y):
    total = 0
    correct = 0
    with torch.no_grad():
        trainer.model = trainer.model.to(trainer.device)
        x, y = x.to(trainer.device), y.to(trainer.device)
        _, predicted = trainer.model(x).max(1)
        total += float(y.size(0))
        correct += float(predicted.eq(y).sum().item())
    return correct / total


def validate(trainer):
    x, y = trainer.next_validation_batch()
    trainer.model.eval()
    acc = validate_on_loader(trainer, x.to(trainer.device), y.to(trainer.device))
    trainer._report(trainer._validation_metric_name("Validation accuracy"), acc)
    return acc


def validate_n_batches(trainer, n_batches: int) -> float:
    if n_batches <= 0:
        return 0.0
    trainer.model.eval()
    total = 0
    correct = 0
    with torch.no_grad(), _eval_autocast(trainer.device):
        for x, y in trainer.iter_validation_batches(int(n_batches)):
            x, y = x.to(trainer.device), y.to(trainer.device)
            _, predicted = trainer.model(x).max(1)
            total += float(y.size(0))
            correct += float(predicted.eq(y).sum().item())
    acc = correct / total if total else 0.0
    trainer._report(trainer._validation_metric_name("Validation accuracy"), acc)
    return acc


def validate_train(trainer):
    x, y = trainer.next_training_batch()
    trainer.model.train()
    acc = validate_on_loader(trainer, x.to(trainer.device), y.to(trainer.device))
    trainer._report(
        trainer._validation_metric_name("Validation accuracy on train set"), acc
    )
    return acc


def evaluate_loss_on_batch(trainer, batch) -> float:
    x, y = batch
    trainer.model.eval()
    with torch.no_grad():
        x, y = x.to(trainer.device), y.to(trainer.device)
        loss = trainer.loss_function(trainer.model, x, y)
    return float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)
