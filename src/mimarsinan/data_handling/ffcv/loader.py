"""FFCV loader subclass: exposes per-batch indices and substitutes labels.

FFCV's stock ``IntDecoder`` is unreliable on ViT-class workloads under any
fused multi-tensor-apply optimizer state — the label tensor comes back
filled with CUDA allocator-fill bytes (allocated, never written). We
bypass it: preload all labels as a ``torch.LongTensor`` once at loader
construction, capture FFCV's internal per-batch indices via a subclassed
``EpochIterator``, and look up the real labels in the preloaded tensor.

The loader's iterator yields ``(x.clone(), label_lookup[indices])``
directly — no separate shim, no torch-DataLoader-compat layer. The clone
on ``x`` is the contract that owns FFCV's rotating-buffer pool: FFCV
yields tensor views into a small pre-allocated pool that gets overwritten
as iteration advances, so downstream consumers that hold a yielded tensor
across iteration boundaries would silently see corrupted data without it.
"""

from __future__ import annotations

from queue import Queue
from typing import Any, Iterator

import numpy as np
import torch
from ffcv.loader.epoch_iterator import EpochIterator
from ffcv.loader.loader import Loader
from ffcv.pipeline.compiler import Compiler


class _IndexedEpochIterator(EpochIterator):
    """``EpochIterator`` that records ``batch_indices`` for every produced batch."""

    def __init__(self, loader: Loader, order):
        # Queue exists before super().__init__ kicks off the worker thread.
        self.batch_index_queue: Queue = Queue()
        super().__init__(loader, order)

    def run_pipeline(self, b_ix, batch_indices, batch_slot, cuda_event):
        # Copy: the source buffer may be reused by the next batch.
        self.batch_index_queue.put(np.ascontiguousarray(batch_indices).copy())
        return super().run_pipeline(b_ix, batch_indices, batch_slot, cuda_event)


class IndexedLoader(Loader):
    """FFCV ``Loader`` that yields ``(x_clone, lookup[batch_indices])``.

    ``label_lookup`` is the preloaded on-device tensor (one entry per
    sample in the beton's order). Required; the lookup path is what makes
    this loader reliable on ViT-class workloads.
    """

    def __init__(self, *args, label_lookup: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self._label_lookup = label_lookup

    def __iter__(self) -> Iterator:
        Compiler.set_num_threads(self.num_workers)
        order = self.next_traversal_order()
        selected_order = order[: len(self) * self.batch_size]
        self.next_epoch += 1
        if self.code is None or self.recompile:
            self.generate_code()
        ep_iter = _IndexedEpochIterator(self, selected_order)
        return self._yield(ep_iter)

    def _yield(self, ep_iter: _IndexedEpochIterator) -> Iterator:
        for batch in ep_iter:
            x = batch[0].clone()
            idx_np = ep_iter.batch_index_queue.get_nowait()
            idx_t = torch.as_tensor(idx_np, dtype=torch.long, device=self._label_lookup.device)
            y = self._label_lookup.index_select(0, idx_t)
            yield x, y


def _unwrap(dataset: Any) -> Any:
    """Strip view-only wrappers that delegate ``__getitem__`` to a base."""
    while hasattr(dataset, "_base") and not hasattr(dataset, "targets"):
        dataset = dataset._base
    return dataset


def preload_labels(dataset: Any) -> torch.LongTensor:
    """Return all integer labels of ``dataset`` as a CPU ``torch.LongTensor``.

    Walks ``Subset`` and ``_AsRGB`` wrappers; uses fast ``.targets`` /
    ``.labels`` metadata when available, falls back to per-sample
    ``__getitem__`` as a last resort.
    """
    dataset = _unwrap(dataset)
    from torch.utils.data import Subset

    if isinstance(dataset, Subset):
        base = _unwrap(dataset.dataset)
        idxs = list(dataset.indices)
        if hasattr(base, "targets"):
            t = base.targets
            if isinstance(t, torch.Tensor):
                return t[torch.as_tensor(idxs, dtype=torch.long)].long().contiguous()
            return torch.as_tensor([int(t[i]) for i in idxs], dtype=torch.long)
        if hasattr(base, "labels"):
            return torch.as_tensor([int(base.labels[i]) for i in idxs], dtype=torch.long)
        return torch.as_tensor([int(dataset[i][1]) for i in range(len(dataset))], dtype=torch.long)

    if hasattr(dataset, "targets"):
        t = dataset.targets
        if isinstance(t, torch.Tensor):
            return t.long().contiguous()
        return torch.as_tensor(list(t), dtype=torch.long)

    if hasattr(dataset, "labels"):
        return torch.as_tensor(list(dataset.labels), dtype=torch.long)

    return torch.as_tensor([int(dataset[i][1]) for i in range(len(dataset))], dtype=torch.long)
