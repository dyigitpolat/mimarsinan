"""FFCV loader subclass that exposes per-batch indices and substitutes preloaded labels."""

from __future__ import annotations

from queue import Queue
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Subset
from ffcv.loader.epoch_iterator import EpochIterator
from ffcv.loader.loader import Loader
from ffcv.pipeline.compiler import Compiler


class _IndexedEpochIterator(EpochIterator):
    """``EpochIterator`` that records ``batch_indices`` for every produced batch."""

    def __init__(self, loader: Loader, order):
        # The queue must exist before super().__init__ starts the worker thread that fills it.
        self.batch_index_queue: Queue = Queue()
        super().__init__(loader, order)

    def run_pipeline(self, b_ix, batch_indices, batch_slot, cuda_event):
        # Copy: the source buffer may be reused by the next batch.
        self.batch_index_queue.put(np.ascontiguousarray(batch_indices).copy())
        return super().run_pipeline(b_ix, batch_indices, batch_slot, cuda_event)


class IndexedLoader(Loader):
    """FFCV ``Loader`` that yields ``(x_clone, lookup[batch_indices])``.

    ``label_lookup`` is the required preloaded on-device label tensor, in the
    beton's sample order; the lookup path replaces FFCV's unreliable IntDecoder.
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
            # clone: FFCV yields views into a rotating buffer pool overwritten as iteration advances.
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

    Walks Subset/_AsRGB wrappers, preferring ``.targets`` / ``.labels``
    metadata and falling back to per-sample ``__getitem__``.
    """
    dataset = _unwrap(dataset)
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
