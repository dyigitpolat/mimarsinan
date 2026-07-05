"""Byte accounting for the hybrid flow's retained segment tensor cache."""

from __future__ import annotations

import torch

# Retention budget for uploaded segment tensors; tier-0 vehicles fit in a few
# tens of MB (float64), large vehicles fall back to per-stage eviction.
SEGMENT_CACHE_MAX_BYTES = 256 * 1024 * 1024


def segment_entry_nbytes(entry: dict) -> int:
    """Device bytes held by one cache entry, deduped by tensor storage."""
    seen: set[int] = set()
    total = 0
    tensors: list = list(entry.get("core_params", []) or [])
    tensors += [b for b in entry.get("hw_biases", []) or [] if b is not None]
    tensors += list(entry.get("thresholds", []) or [])
    tensors += list((entry.get("bank_tensors", {}) or {}).values())
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            continue
        storage = t.untyped_storage()
        key = storage.data_ptr()
        if key in seen:
            continue
        seen.add(key)
        total += int(storage.nbytes())
    return total
