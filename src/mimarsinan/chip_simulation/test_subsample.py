"""Single source of truth for the spiking-sim test subsample.

SCM (``BasicTrainer.test_on_subsample``), HCM (same), and nevresim
(``SimulationRunner``) must all evaluate on the **same** test samples for
their accuracies to be directly comparable. Inlining the
``np.random.RandomState(seed).choice(...)`` call in two places risked
silent drift if either copy reads a slightly different ``total_samples``
(e.g. loader ``drop_last`` vs the raw dataset length). This helper is
the one place that produces the subsample index list.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_test_subsample_indices(
    *,
    total_samples: int,
    seed: int,
    max_samples: int,
) -> list[int]:
    """Return a deterministic, seeded subsample index list.

    Returns the full ``[0, total_samples)`` range when subsampling is
    disabled (``max_samples <= 0`` or ``>= total_samples``), so callers
    can iterate the result unconditionally.
    """
    if total_samples <= 0:
        return []
    if max_samples is None or max_samples <= 0 or max_samples >= total_samples:
        return list(range(total_samples))
    rng = np.random.RandomState(int(seed))
    return [
        int(i)
        for i in rng.choice(total_samples, size=int(max_samples), replace=False)
    ]
