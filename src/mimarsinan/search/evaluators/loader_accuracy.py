"""The shared measurement seam of the NAS evaluators: verified accuracy over a loader.

Both evaluators end in the same loop -- a full pass over a validation loader,
top-1, no grad -- and both DECIDE on the result: it is the fitness that ranks
candidate architectures. A corrupted read here does not print a wrong number, it
silently reorders the search. So the pass goes through ``batch_integrity`` exactly
like the trainer's reporting and decision paths do.
"""

from __future__ import annotations

import torch

from mimarsinan.data_handling import batch_integrity


def _pass(model, loader, device, source: str) -> tuple[float, float]:
    correct = 0.0
    total = 0.0
    for x, y in loader:
        batch_integrity.verify_batch(x, source=source)
        x = x.to(device)
        y = y.to(device)
        _, predicted = model(x).max(1)
        total += float(y.size(0))
        correct += float(predicted.eq(y).sum().item())
    return correct, total


@torch.no_grad()
def accuracy_over_loader(model, loader, device, *, source: str = "the NAS validation loader") -> float:
    """Top-1 accuracy over a full pass, restarted whole on a corrupted read.

    ``model`` is left in ``eval`` mode by this call; callers that keep training
    restore ``train`` themselves.
    """
    model.eval()
    correct, total = batch_integrity.verified_pass(
        lambda: _pass(model, loader, device, source), source=source,
    )
    return float(correct / total) if total > 0 else 0.0
