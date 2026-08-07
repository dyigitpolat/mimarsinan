"""Load deterministic test-set samples by index for parity steps.

These samples ARE the deployment claim: ``loihi_simulation_step`` and
``sanafe_simulation_step`` feed them to a chip simulator and compare the result
against a torch reference, so the number that comes out is a parity verdict. A
corrupted read here does not print a wrong accuracy -- it passes or fails a parity
check on data that was never the dataset. The read is therefore guarded exactly
like every other measured read (``batch_integrity``), and the samples are owned
snapshots rather than views the producer may still refill.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch

from mimarsinan.data_handling import batch_integrity
from mimarsinan.data_handling.data_loader_factory import (
    DataLoaderFactory,
    shutdown_data_loader,
)


def _collect_wanted(
    loader: Iterable, wanted: set[int], source: str,
) -> Tuple[dict[int, torch.Tensor], set[int], int]:
    """One verified pass, stopping as soon as every wanted index is owned.

    A positional read cannot repair by skipping (index ``k`` means sample ``k``),
    so this is a ``verified_pass``: restarted whole, or raised.
    """
    remaining = set(wanted)
    out: dict[int, torch.Tensor] = {}
    seen = 0
    for xs, ys in loader:
        xs, _ys = batch_integrity.own_verified_batch(xs, ys, source=source)
        for local in range(int(xs.shape[0])):
            if seen in remaining:
                # Cloned off the owned batch so a single retained sample does not
                # pin the whole batch buffer alive behind it.
                out[seen] = xs[local : local + 1].clone()
                remaining.discard(seen)
            seen += 1
            if not remaining:
                break
        if not remaining:
            break
    return out, remaining, seen


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
    source = f"the parity sample loader ({type(loader).__name__})"

    try:
        out, remaining, seen = batch_integrity.verified_pass(
            lambda: _collect_wanted(loader, wanted, source), source=source,
        )
    finally:
        shutdown_data_loader(loader)

    if remaining:
        raise IndexError(
            f"sample indices {sorted(remaining)} exceed the test set size (seen {seen})"
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
