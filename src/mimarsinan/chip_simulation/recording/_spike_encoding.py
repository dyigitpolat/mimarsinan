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

    Uses torch ``to_uniform_spikes`` per cycle so batch output matches HCM.
    """
    import torch

    from mimarsinan.chip_simulation.recording import spike_modes

    clipped = np.clip(rates, 0.0, 1.0)
    tensor = torch.tensor(clipped, dtype=torch.float32)
    n_samples, d = tensor.shape
    spikes = np.zeros((n_samples, d, T), dtype=np.float32)
    for cycle in range(T):
        spikes[:, :, cycle] = spike_modes.to_uniform_spikes(
            tensor, cycle, T,
        ).numpy()
    return spikes


def deterministic_rate_encode(rates: np.ndarray, T: int) -> np.ndarray:
    """Deterministic encoding: fires every cycle when rate > 0.5."""
    import torch

    from mimarsinan.chip_simulation.recording import spike_modes

    clipped = np.clip(rates, 0.0, 1.0)
    tensor = torch.tensor(clipped, dtype=torch.float32)
    n_samples, d = tensor.shape
    spikes = np.zeros((n_samples, d, T), dtype=np.float32)
    per_cycle = spike_modes.to_deterministic_spikes(tensor).numpy()
    for cycle in range(T):
        spikes[:, :, cycle] = per_cycle
    return spikes


def front_loaded_rate_encode(rates: np.ndarray, T: int) -> np.ndarray:
    """Front-loaded encoding: spike when round(rate * T) > cycle."""
    rates = np.clip(rates, 0.0, 1.0)
    N_samples, D = rates.shape
    spikes = np.zeros((N_samples, D, T), dtype=np.float32)
    n = np.round(rates * T).astype(np.int64)
    for cycle in range(T):
        spikes[:, :, cycle] = (n > cycle).astype(np.float32)
    return spikes


def stochastic_rate_encode(
    rates: np.ndarray,
    T: int,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Stochastic Bernoulli encoding per cycle."""
    rates = np.clip(rates, 0.0, 1.0)
    N_samples, D = rates.shape
    if rng is None:
        rng = np.random.default_rng()
    random = rng.random((N_samples, D, T))
    return (random < rates[..., None]).astype(np.float32)


def spike_train_rate_encode(rates: np.ndarray, T: int) -> np.ndarray:
    """Materialize a full spike train for ``spike_mode='SpikeTrain'``.

    Matches HCM ``materialized_spike_train`` / ``uniform_spike_train`` so host
    replay (Nevresim ``SpikeTrainSpikeGenerator``, Lava, SANA-FE) sees the same
    cycle-major train HCM recorded.
    """
    return uniform_rate_encode(rates, T)


def flatten_spike_train_sample(encoded_sample: np.ndarray) -> np.ndarray:
    """Flatten ``(D, T)`` encoded spikes to Nevresim cycle-major layout."""
    if encoded_sample.ndim != 2:
        raise ValueError(
            f"expected encoded sample shape (D, T); got {encoded_sample.shape}"
        )
    return np.transpose(encoded_sample, (1, 0)).reshape(-1).astype(np.float64)


def encode_segment_input(
    rates: np.ndarray,
    T: int,
    spike_mode: str,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Batch-encode segment input rates for Lava/SANA-FE injection."""
    if spike_mode == "TTFS":
        raise ValueError(
            "encode_segment_input does not support spike_mode='TTFS'; "
            "use the TTFS encoding path"
        )
    if spike_mode == "Uniform":
        return uniform_rate_encode(rates, T)
    if spike_mode == "Deterministic":
        return deterministic_rate_encode(rates, T)
    if spike_mode == "FrontLoaded":
        return front_loaded_rate_encode(rates, T)
    if spike_mode == "Stochastic":
        rng = np.random.default_rng(seed)
        return stochastic_rate_encode(rates, T, rng=rng)
    if spike_mode == "SpikeTrain":
        return spike_train_rate_encode(rates, T)
    raise ValueError(f"Invalid spike_mode: {spike_mode!r}")
