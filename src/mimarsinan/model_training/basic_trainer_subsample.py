"""Deterministic test subsample evaluation for :class:`BasicTrainer`."""

from __future__ import annotations

import torch

from mimarsinan.chip_simulation.subsample import compute_test_subsample_indices
from mimarsinan.common.env import vram_probe_enabled
from mimarsinan.data_handling import batch_integrity


def _collect_all(trainer, source: str):
    """Every test sample, as OWNED per-sample copies, verified batch by batch.

    ``x[i]`` is a VIEW into the loader's batch buffer, and a producer thread
    refills that buffer, so a list of views held across the whole pass is a list
    of samples the pipeline is still free to overwrite -- the exact aliasing
    hazard ``own_batch``'s clone exists for. Cloning per sample also drops the
    hidden retention of every batch buffer a single selected view would pin.

    Own THEN verify (W0.8b finding 3): the per-sample clones are taken from the
    OWNED batch, so the verdict covers the bytes that end up in the subsample
    rather than a buffer that may already have been refilled by the time the
    clones are made.
    """
    xs_all: list[torch.Tensor] = []
    ys_all: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in trainer.test_loader:
            x, y = batch_integrity.own_verified_batch(x, y, source=source)
            for i in range(x.shape[0]):
                xs_all.append(x[i].clone())
                ys_all.append(y[i].clone())
    return xs_all, ys_all


def _collect_selected(trainer, selected, source: str):
    """The selected test samples, as OWNED per-sample copies, verified batch by batch.

    Owned before verified, for the reason spelled out in :func:`_collect_all`."""
    xs_all: list[torch.Tensor] = []
    ys_all: list[torch.Tensor] = []
    with torch.no_grad():
        global_idx = 0
        for x, y in trainer.test_loader:
            x, y = batch_integrity.own_verified_batch(x, y, source=source)
            bsz = int(x.shape[0])
            for i in range(bsz):
                if selected is None or global_idx in selected:
                    xs_all.append(x[i].clone())
                    ys_all.append(y[i].clone())
                global_idx += 1
            if selected is not None and len(xs_all) >= len(selected):
                break
    return xs_all, ys_all


def test_on_subsample(trainer, *, max_samples: int, seed: int = 0) -> float:
    """Run test over a deterministic subsample of the test set.

    Every collected batch is proven finite as it is read, and the whole
    collection pass restarts on a corrupted read: this is what
    ``PipelineStep.pipeline_metric`` reports AND progresses the pipeline on, so a
    poisoned batch here is a wrong decision, not just a wrong print.
    """
    try:
        total_samples = len(trainer.data_provider._get_test_dataset())
    except (TypeError, NotImplementedError):
        total_samples = None

    # The cap covers the whole set: defer to the exact full test() path so a cap
    # >= the dataset size is byte-identical (no subsampling, same fp/order).
    if total_samples is not None and 0 < total_samples <= int(max_samples):
        return trainer.test()

    source = f"the test loader ({type(getattr(trainer, 'test_loader', None)).__name__})"
    if total_samples is None or total_samples <= 0:
        xs_all, ys_all = batch_integrity.verified_pass(
            lambda: _collect_all(trainer, source), source=source,
        )
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

        xs_all, ys_all = batch_integrity.verified_pass(
            lambda: _collect_selected(trainer, selected, source), source=source,
        )
        if not xs_all:
            return 0.0

    bs = int(trainer.test_batch_size)
    total = 0
    correct = 0
    _probe = vram_probe_enabled()
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
