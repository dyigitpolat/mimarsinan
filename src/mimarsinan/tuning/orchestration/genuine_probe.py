"""Pure helpers for probing an arbitrary forward over the validation set."""

import copy
from typing import Iterable, Tuple

import torch


def iter_val_batches(
    trainer, n_batches: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """Typed view of the trainer's ``iter_validation_batches`` (x, y) pairs."""
    return trainer.iter_validation_batches(int(n_batches))


def eval_forward_over_val(trainer, forward_obj, model, n_batches, device) -> float:
    """Top-1 accuracy of ``forward_obj`` over ``n_batches`` val batches; never installs it."""
    n_batches = int(n_batches)
    if n_batches <= 0:
        return 0.0
    model.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for x, y in iter_val_batches(trainer, n_batches):
            x, y = x.to(device), y.to(device)
            _, predicted = forward_obj(x).max(1)
            total += float(y.size(0))
            correct += float(predicted.eq(y).sum().item())
    return correct / total if total else 0.0


def genuine_acc_on_clone(model, device, *, prepare, build_forward, evaluate) -> float:
    """Genuine accuracy on a deepcopy: prepare → build forward → evaluate; live model untouched."""
    clone = copy.deepcopy(model).to(device)
    prepare(clone)
    forward_obj = build_forward(clone)
    return evaluate(forward_obj, clone)
