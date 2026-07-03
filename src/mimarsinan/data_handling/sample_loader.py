"""Load deterministic test-set samples by index for parity steps."""

from __future__ import annotations

from typing import List, Sequence

import torch

from mimarsinan.data_handling.data_loader_factory import (
    DataLoaderFactory,
    shutdown_data_loader,
)


def load_test_samples_by_index(
    data_provider_factory,
    indices: Sequence[int],
    *,
    num_workers: int = 4,
) -> List[torch.Tensor]:
    """Return one batch tensor per index in ``indices`` (order preserved)."""
    wanted = set(int(i) for i in indices)
    if not wanted:
        return []
    if any(i < 0 for i in wanted):
        raise ValueError("all sample indices must be >= 0")

    factory = DataLoaderFactory(data_provider_factory, num_workers=num_workers)
    provider = factory.create_data_provider()
    loader = factory.create_test_loader(provider.get_test_batch_size(), provider)

    out: dict[int, torch.Tensor] = {}
    seen = 0
    try:
        for xs, _ys in loader:
            batch_size = int(xs.shape[0])
            for local in range(batch_size):
                if seen in wanted:
                    out[seen] = xs[local : local + 1]
                    wanted.discard(seen)
                seen += 1
                if not wanted:
                    break
            if not wanted:
                break
    finally:
        shutdown_data_loader(loader)

    if wanted:
        raise IndexError(
            f"sample indices {sorted(wanted)} exceed the test set size (seen {seen})"
        )
    return [out[i] for i in indices]


def load_test_sample_by_index(
    data_provider_factory,
    sample_index: int,
    *,
    num_workers: int = 4,
) -> torch.Tensor:
    """Return a single-sample batch tensor for ``sample_index``."""
    if sample_index < 0:
        raise ValueError("sample_index must be >= 0")
    return load_test_samples_by_index(
        data_provider_factory, [sample_index], num_workers=num_workers
    )[0]
