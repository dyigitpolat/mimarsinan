"""Load deterministic test-set (or validation-set) samples by index for parity and census steps.

These samples ARE the deployment claim: ``loihi_simulation_step`` and
``sanafe_simulation_step`` feed them to a chip simulator and compare the result
against a torch reference, so the number that comes out is a parity verdict. A
corrupted read here does not print a wrong accuracy -- it passes or fails a parity
check on data that was never the dataset. The read is therefore guarded exactly
like every other measured read (``batch_integrity``), and the samples are owned
snapshots rather than views the producer may still refill.

[U-HV] The split is a declaration: ``load_pairs_by_index(split=...)`` reads the development split
(``DEVELOPMENT_SPLIT``, the provider's validation carve) or the test split, which only a run whose declaration names
it asks for; any other name is refused by name (``UnknownSampleSplit``) before a provider is built.
``sample_read`` is the record of such a read that bundles, step entries and the deployment record print.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.data_handling import batch_integrity
from mimarsinan.data_handling.data_loader_factory import (
    DataLoaderFactory,
    shutdown_data_loader,
)

#: [U-HV] The development split: the provider's validation carve, the split every measurement reads by default.
DEVELOPMENT_SPLIT = "validation"
#: [U-HV] The splits a verified read can serve: the development split, then the test split (read only by name).
SAMPLE_SPLITS = (DEVELOPMENT_SPLIT, "test")


class UnknownSampleSplit(ToolchainRefusal, ValueError):
    """[U-HV] A sample split that names neither the development nor the test split."""


def require_sample_split(split: Any, *, declared_by: str = "split") -> str:
    """``split`` when it names a split of ``SAMPLE_SPLITS``; otherwise a refusal naming ``declared_by`` and the value."""
    if isinstance(split, str) and split in SAMPLE_SPLITS:
        return split
    raise UnknownSampleSplit(
        f"{declared_by}={split!r} names no sample split: the splits are {SAMPLE_SPLITS!r} (the development split "
        f"{DEVELOPMENT_SPLIT!r}, or the test split by name); no sample was read")


def sample_read(split: str, indices: Iterable[int], *, declared_by: str) -> Dict[str, Any]:
    """[U-HV] The record of one read: the declaring key, the split and the per-split sample indices, in read order."""
    return {"declared_by": str(declared_by), "split": require_sample_split(split, declared_by=declared_by),
            "indices": [int(index) for index in indices]}


def _collect_wanted(
    loader: Iterable, wanted: set[int], source: str,
) -> Tuple[dict[int, Tuple[torch.Tensor, torch.Tensor]], set[int], int]:
    """One verified pass, stopping as soon as every wanted index is owned.

    A positional read cannot repair by skipping (index ``k`` means sample ``k``),
    so this is a ``verified_pass``: restarted whole, or raised.
    """
    remaining = set(wanted)
    out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    seen = 0
    for xs, ys in loader:
        xs, ys = batch_integrity.own_verified_batch(xs, ys, source=source)
        for local in range(int(xs.shape[0])):
            if seen in remaining:
                # Cloned off the owned batch so a single retained sample does not
                # pin the whole batch buffer alive behind it.
                out[seen] = (
                    xs[local : local + 1].clone(), ys[local : local + 1].clone())
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
    return [sample for sample, _label in _pairs_by_index(
        data_provider_factory, indices, num_workers=num_workers, split="test")]


def load_test_pairs_by_index(
    data_provider_factory,
    indices: Sequence[int],
    *,
    num_workers: int = 4,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return one ``(sample, label)`` batch pair per index, order preserved.

    The LABEL half is what turns a parity read into a deployed-accuracy claim;
    it rides the same verified pass rather than a second, unguarded read.
    """
    return _pairs_by_index(data_provider_factory, indices, num_workers=num_workers, split="test")


def load_pairs_by_index(
    data_provider_factory,
    indices: Sequence[int],
    *,
    split: str,
    num_workers: int = 4,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """[U-HV] One ``(sample, label)`` batch pair per index of the DECLARED split, order preserved.

    ``split`` is ``DEVELOPMENT_SPLIT`` or ``"test"``; any other value is refused by name before a provider is built.
    """
    return _pairs_by_index(data_provider_factory, indices, num_workers=num_workers, split=split)


def load_validation_samples_by_index(
    data_provider_factory,
    indices: Sequence[int],
    *,
    num_workers: int = 4,
) -> List[torch.Tensor]:
    """[U-S] The same verified read on the validation split: the inputs of a measurement that must never read the
    test split (the event census)."""
    return [sample for sample, _label in _pairs_by_index(
        data_provider_factory, indices, num_workers=num_workers, split="validation")]


def _pairs_by_index(data_provider_factory, indices: Sequence[int], *, num_workers: int,
                    split: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    # [U-HV] Refused by name first: an unrecognised split used to fall through to the test loader.
    split = require_sample_split(split)
    wanted = set(int(i) for i in indices)
    if not wanted:
        return []
    if any(i < 0 for i in wanted):
        raise ValueError("all sample indices must be >= 0")

    factory = DataLoaderFactory(data_provider_factory, num_workers=num_workers)
    provider = factory.create_data_provider()
    if split == DEVELOPMENT_SPLIT:
        loader = factory.create_validation_loader(provider.get_validation_batch_size(), provider)
    else:
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
            f"sample indices {sorted(remaining)} exceed the {split} set size (seen {seen})"
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
