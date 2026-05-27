"""Deterministic test subsample evaluation for :class:`BasicTrainer`."""

from __future__ import annotations

import os

import torch


def test_on_subsample(trainer, *, max_samples: int, seed: int = 0) -> float:
    """Run test over a deterministic subsample of the test set."""
    from mimarsinan.chip_simulation.test_subsample import (
        compute_test_subsample_indices,
    )

    try:
        total_samples = len(trainer.data_provider._get_test_dataset())
    except Exception:
        total_samples = None

    if total_samples is None or total_samples <= 0:
        xs_all: list[torch.Tensor] = []
        ys_all: list[torch.Tensor] = []
        with torch.no_grad():
            for x, y in trainer.test_loader:
                for i in range(x.shape[0]):
                    xs_all.append(x[i])
                    ys_all.append(y[i])
        total_samples = len(xs_all)
        if total_samples == 0:
            return 0.0
        indices = compute_test_subsample_indices(
            total_samples=total_samples,
            seed=int(seed),
            max_samples=int(max_samples),
        )
        if len(indices) < total_samples:
            xs_all = [xs_all[i] for i in indices]
            ys_all = [ys_all[i] for i in indices]
    else:
        indices = compute_test_subsample_indices(
            total_samples=total_samples,
            seed=int(seed),
            max_samples=int(max_samples),
        )
        selected = set(indices) if len(indices) < total_samples else None

        xs_all = []
        ys_all = []
        with torch.no_grad():
            global_idx = 0
            for x, y in trainer.test_loader:
                bsz = int(x.shape[0])
                for i in range(bsz):
                    if selected is None or global_idx in selected:
                        xs_all.append(x[i])
                        ys_all.append(y[i])
                    global_idx += 1
                if selected is not None and len(xs_all) >= len(selected):
                    break
        if not xs_all:
            return 0.0

    bs = int(trainer.test_batch_size)
    total = 0
    correct = 0
    _probe = os.environ.get("MIMARSINAN_VRAM_PROBE") == "1"
    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, len(xs_all), bs)):
            x = torch.stack(xs_all[start:start + bs]).to(trainer.device)
            y = torch.stack(ys_all[start:start + bs]).to(trainer.device)
            trainer.model.eval()
            trainer.model = trainer.model.to(trainer.device)
            if _probe and torch.cuda.is_available():
                torch.cuda.synchronize()
                alc = torch.cuda.memory_allocated()
                rsv = torch.cuda.memory_reserved()
                peak = torch.cuda.max_memory_allocated()
                print(
                    f"[VRAM::batch {batch_idx:03d}] pre_forward  "
                    f"alc={alc/1e6:8.1f} MB  rsv={rsv/1e6:8.1f} MB  "
                    f"peak={peak/1e6:8.1f} MB",
                    flush=True,
                )
            _, predicted = trainer.model(x).max(1)
            if _probe and torch.cuda.is_available():
                torch.cuda.synchronize()
                alc = torch.cuda.memory_allocated()
                rsv = torch.cuda.memory_reserved()
                peak = torch.cuda.max_memory_allocated()
                print(
                    f"[VRAM::batch {batch_idx:03d}] post_forward "
                    f"alc={alc/1e6:8.1f} MB  rsv={rsv/1e6:8.1f} MB  "
                    f"peak={peak/1e6:8.1f} MB",
                    flush=True,
                )
            total += float(y.size(0))
            correct += float(predicted.eq(y).sum().item())
    if total <= 0:
        return 0.0
    acc = correct / total
    trainer._report("Test accuracy (subsample)", acc)
    return acc
