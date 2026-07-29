"""Batch-integrity SSOT: no number this program publishes is measured on a corrupted read.

W0.7 established the defect. FFCV's ``EpochIterator`` hands out views into device
buffers filled by a producer thread on a side CUDA stream; a batch that waited in
the queue across unrelated GPU work has been observed coming back ENTIRELY
non-finite (measured: 393216/393216 NaN) while the identical batch, re-read from a
fresh iterator microseconds later, was bit-correct. Measured on, a NaN batch makes
the model emit one constant class, so the "accuracy" lands at that class's share of
the batch -- 11/128 = 0.0859375 on CIFAR-10 validation batch 0, which is exactly the
figure a converted VGG-8 reported seconds after the same weights tested 0.9266.

Evaluation inputs are finite by construction (decoded, normalized images), so a
non-finite input batch is a corrupted READ of an immutable source, never data. This
module is the one place that says so, and every seam that turns batches into a
reported or decided number goes through it. Three primitives:

``own_batch``   snapshot first. The check must apply to the exact tensor the model
                is measured on; verifying a live view into a rotating producer
                buffer proves nothing about what the forward pass later reads.
``read_verified_batch``  a SINGLE-batch read, repaired and re-read on corruption.
``verified_pass``        a FULL loader pass, restarted whole on corruption.

The split is not cosmetic. A single-batch reader may repair by reading a DIFFERENT
position (any validation batch is an equally valid draw). A full pass may not: it
measures every position, so skipping one silently changes the measured set. Its only
honest repair is to redo the whole pass, and its only honest failure is to raise.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import torch

CORRUPT_BATCH_RETRIES = 2
"""Repaired re-reads allowed before a non-finite batch is called a broken pipeline."""

_T = TypeVar("_T")


class CorruptedBatchError(RuntimeError):
    """A batch that cannot be a measurement of the model.

    Raised instead of returning a number, and deliberately NOT a subclass of
    anything a ``best_effort``-style fallback swallows: a corrupted read must end
    the measurement, not degrade it.
    """


def is_corrupt_batch(x: Any) -> bool:
    """Whether ``x`` is a corrupted read: a float input carrying non-finite values.

    Integer/boolean inputs cannot be non-finite, so they are never probed (and
    ``torch.isfinite`` would be a pointless device round-trip on them).
    """
    if not isinstance(x, torch.Tensor):
        return False
    return bool(torch.is_floating_point(x)) and not bool(torch.isfinite(x).all())


def own_batch(x: Any, y: Any) -> tuple[Any, Any]:
    """An OWNED snapshot of ``(x, y)`` -- the tensor the verification is about.

    Load-bearing, not defensive: the loader yields views into a rotating buffer
    pool that the producer thread refills. Verifying a view and then measuring on
    that same view leaves a window in which the buffer is refilled between the
    check and the forward pass, so the check would guarantee nothing. Cloning
    first makes the verified bytes immutable, and the verdict permanent.
    """
    return (
        x.clone() if isinstance(x, torch.Tensor) else x,
        y.clone() if isinstance(y, torch.Tensor) else y,
    )


def _describe(x: torch.Tensor, source: str) -> str:
    return (
        f"{int((~torch.isfinite(x)).sum())}/{x.numel()} non-finite input values "
        f"read from {source}"
    )


def verify_batch(x: Any, *, source: str) -> None:
    """Raise :class:`CorruptedBatchError` if ``x`` is a corrupted read."""
    if is_corrupt_batch(x):
        raise CorruptedBatchError(
            f"{_describe(x, source)}. Evaluation images are finite by "
            "construction, so this batch is a corrupted read, not data."
        )


def read_verified_batch(
    read: Callable[[], tuple[Any, Any]],
    *,
    source: str,
    repair: Callable[[int], None] | None = None,
    retries: int = CORRUPT_BATCH_RETRIES,
) -> tuple[Any, Any]:
    """One verified batch from ``read``, repaired and re-read while it comes back corrupt.

    ``read()`` must return an OWNED batch (see :func:`own_batch`). ``repair(attempt)``
    receives the 1-based attempt number and must make the next ``read()`` draw from
    somewhere the corruption has not already been observed -- see
    ``BasicTrainer._repair_validation_read`` for the escalation this contract exists
    for. Exhausting the retries raises rather than returning a number that is not a
    measurement of the model.
    """
    for attempt in range(retries + 1):
        x, y = read()
        if not is_corrupt_batch(x):
            return x, y
        print(
            f"[batch_integrity] CORRUPTED batch discarded "
            f"(attempt {attempt + 1}/{retries + 1}): {_describe(x, source)}. "
            f"The batch is a corrupted read, not data; re-reading.",
            flush=True,
        )
        if repair is not None:
            repair(attempt + 1)
    raise CorruptedBatchError(
        f"{source} kept returning non-finite batches after {retries} repaired "
        "re-reads. Evaluation images are finite by construction, so this is a "
        "broken input pipeline, not data -- refusing to report a number measured "
        "on it."
    )


def verified_pass(
    run_pass: Callable[[], _T],
    *,
    source: str,
    retries: int = CORRUPT_BATCH_RETRIES,
) -> _T:
    """Run a FULL loader pass, restarting it whole when a batch comes back corrupt.

    ``run_pass()`` iterates the loader from the start and must
    :func:`verify_batch` every batch it consumes, so a corrupted read aborts it
    with :class:`CorruptedBatchError` before any partial result escapes. The
    restart re-iterates from a fresh iterator, which is what abandons the poisoned
    in-flight queue.

    There is no position to skip here: a pass measures every position, so a
    corruption that survives every restart is positionally deterministic -- the
    pipeline, not a transient -- and the pass raises rather than returning a number
    measured on a silently reduced set.
    """
    for attempt in range(retries + 1):
        try:
            return run_pass()
        except CorruptedBatchError as err:
            if attempt == retries:
                raise
            print(
                f"[batch_integrity] {err} Restarting the {source} pass "
                f"(attempt {attempt + 2}/{retries + 1}).",
                flush=True,
            )
    raise AssertionError("unreachable: the final attempt either returns or raises")


def verify_owned_batches(batches, *, source: str) -> None:
    """Verify an already-materialized, OWNED batch list (e.g. a pooled eval cache).

    No repair exists for owned tensors -- they are snapshots, so a re-read cannot
    change them -- which is exactly why this raises instead of retrying.
    """
    for index, (x, _y) in enumerate(batches):
        if is_corrupt_batch(x):
            raise CorruptedBatchError(
                f"{_describe(x, f'{source} entry {index}')}. A materialized eval "
                "cache is an owned snapshot, so this cannot be repaired by "
                "re-reading -- refusing to decide on it."
            )
