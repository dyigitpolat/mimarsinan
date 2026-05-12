"""Spike-encoding primitives shared by simulator backends.

``uniform_rate_encode`` was originally a private helper inside
``lava_loihi_runner.py``; it is reused by the SANA-FE runner to inject
rate-coded input spike trains into each neural segment.  Lifting it to a
neutral module keeps both runners pointed at one implementation so a
behavioural drift can only happen here.
"""

from __future__ import annotations

import numpy as np


def uniform_rate_encode(rates: np.ndarray, T: int) -> np.ndarray:
    """Uniform-rate spike encoding matching SCM ``spike_mode='Uniform'``.

    Parameters
    ----------
    rates
        ``(N, D)`` array of non-negative values; clipped into ``[0, 1]``.
    T
        Number of cycles.

    Returns
    -------
    ``(N, D, T)`` binary spike train.  For each ``(n, d)``,
    ``N_d = round(rate * T)`` spikes are placed at uniformly-spaced cycle
    indices; ``rate == 1.0`` saturates to a spike at every cycle.
    """
    rates = np.clip(rates, 0.0, 1.0)
    N_samples, D = rates.shape
    spikes = np.zeros((N_samples, D, T), dtype=np.float32)
    for cycle in range(T):
        n = np.round(rates * T).astype(np.int64)  # (N, D)
        mask_full = (n == T)
        mask_active = (n != 0) & (n != T) & (cycle < T)
        n_safe = np.maximum(n, 1)
        spacing = T / n_safe.astype(np.float64)
        fire = mask_active & (np.floor(cycle / spacing) < n_safe) & (np.floor(cycle % spacing) == 0)
        spikes[:, :, cycle] = fire.astype(np.float32)
        spikes[:, :, cycle][mask_full] = 1.0
    return spikes
