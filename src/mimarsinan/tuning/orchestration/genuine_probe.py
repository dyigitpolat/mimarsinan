"""Pure helpers for probing an arbitrary forward over the validation set."""

import copy
from typing import Iterable, Tuple

import torch


def iter_val_batches(
    trainer, n_batches: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """Typed view of the trainer's ``iter_validation_batches`` (x, y) pairs."""
    return trainer.iter_validation_batches(int(n_batches))


def _forward_adaptive_chunks(forward_obj, x):
    """[C4'] full-batch forward, halving into chunks on CUDA OOM: the genuine
    spike-train eval materializes S x batch x features, so a val batch sized
    for the fp32 loaders can exceed VRAM at deploy-eval time (measured
    36.94 GiB = 32 cycles x 512 batch on a ViT). Value-identical (pure eval,
    recomputed per retry); an OOM at chunk 1 fails loud."""
    chunk = int(x.size(0))
    while True:
        try:
            if chunk >= x.size(0):
                return forward_obj(x)
            return torch.cat([
                forward_obj(part) for part in torch.split(x, chunk)
            ])
        except torch.OutOfMemoryError:
            if chunk <= 1:
                raise
            chunk = max(1, chunk // 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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
            _, predicted = _forward_adaptive_chunks(forward_obj, x).max(1)
            total += float(y.size(0))
            correct += float(predicted.eq(y).sum().item())
    return correct / total if total else 0.0


def genuine_acc_on_clone(model, device, *, prepare, build_forward, evaluate) -> float:
    """Genuine accuracy on a deepcopy: prepare → build forward → evaluate; live model untouched."""
    clone = copy.deepcopy(model).to(device)
    prepare(clone)
    forward_obj = build_forward(clone)
    return evaluate(forward_obj, clone)
