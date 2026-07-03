"""SSOT producing the seeded spiking-sim test-subsample index list (SCM/HCM/nevresim must share it)."""

from __future__ import annotations

import numpy as np


def compute_test_subsample_indices(
    *,
    total_samples: int,
    seed: int,
    max_samples: int,
) -> list[int]:
    """Return a deterministic, seeded subsample index list (the full range when subsampling is disabled)."""
    if total_samples <= 0:
        return []
    if max_samples is None or max_samples <= 0 or max_samples >= total_samples:
        return list(range(total_samples))
    rng = np.random.RandomState(int(seed))
    return [
        int(i)
        for i in rng.choice(total_samples, size=int(max_samples), replace=False)
    ]
