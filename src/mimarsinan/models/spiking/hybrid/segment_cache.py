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
    for plan in entry.get("axon_fill_plans", []) or []:
        tensors += list(plan.tensors())
    # [F4] the packed-cycle executor memoises its PackedStage into THIS entry
    # (``seg["packed"]``); each bucket holds a DENSE stack of the per-core
    # weight views. Uncounted, the budget under-reports by exactly the bytes
    # that make large stages expensive.
    packed = entry.get("packed")
    if packed is not None:
        tensors.append(getattr(packed, "theta_flat", None))
        for bucket in getattr(packed, "buckets", []) or []:
            tensors += [
                getattr(bucket, name, None) for name in (
                    "weights", "bias", "on_dst", "inp_dst", "inp_src",
                    "buf_dst", "buf_src",
                )
            ]
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
