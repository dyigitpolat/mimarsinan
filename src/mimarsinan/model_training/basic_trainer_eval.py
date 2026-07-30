"""Evaluation and validation helpers for :class:`BasicTrainer`."""

from __future__ import annotations

import contextlib
import random

import torch

from mimarsinan.data_handling import batch_integrity

# Fixed so the decision subsample is identical across every validation in a run and reproducible across runs.
_VAL_SUBSAMPLE_SEED = 1234


def _val_source(trainer) -> str:
    return f"the validation loader ({type(getattr(trainer, 'validation_loader', None)).__name__})"


def _test_source(trainer) -> str:
    return f"the test loader ({type(getattr(trainer, 'test_loader', None)).__name__})"


def metric_grade_eval(device):
    """fp32 measurement seam: disables any ambient autocast on ``device`` so
    reported/gate metrics never inherit fp16/bf16 kernels from a surrounding
    training region (docs/research/findings/numerical_boundary_consistency.md §2 RC2)."""
    device_type = torch.device(device).type
    if device_type not in ("cuda", "cpu"):
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, enabled=False)


def _verified_to_device(trainer, x, y, source: str):
    """An owned device batch that has been PROVEN finite.

    Snapshot first, verify second -- delegated to ``own_verified_batch`` so the
    ordering is stated in one place rather than re-implemented per seam.
    """
    return batch_integrity.own_verified_batch(
        x, y, source=source, device=trainer.device, non_blocking=True,
    )


def _shared_eval_cache_key(trainer, max_batches):
    return (
        "val",
        int(trainer.validation_batch_size),
        str(trainer.device),
        None if max_batches is None else int(max_batches),
    )


def _shared_eval_cache_seam(trainer):
    """(get, put) on the trainer's factory pool, or ``None`` outside pooled mode.

    The cached batches are inputs only (model-independent) and the val loader is
    unshuffled, so pooled content is bit-identical to a per-trainer rebuild.
    """
    factory = getattr(trainer, "data_loader_factory", None)
    owns = getattr(factory, "owns_loaders", None)
    if not (callable(owns) and owns()):
        return None
    return factory


def _materialize_val_cache(trainer, max_batches, source: str):
    """One full pass over the validation loader into an OWNED, VERIFIED device cache.

    Every batch that enters the cache is proven finite as it is materialized, so
    the pass aborts on a corrupted read before a partial cache escapes. Batches
    the reservoir rejects are never materialized and never probed: they cannot
    reach a number, and probing them would buy a device sync per batch.
    """
    # Caps the on-device cache to a seeded reservoir subsample so the full validation set is never materialized on the device.
    if max_batches is None:
        return [
            _verified_to_device(trainer, x, y, source)
            for x, y in trainer.validation_loader
        ]
    cap = int(max_batches)
    # Re-seeded per attempt so a restarted pass selects the identical subsample.
    rng = random.Random(_VAL_SUBSAMPLE_SEED)
    cache = []
    for i, (x, y) in enumerate(trainer.validation_loader):
        if i < cap:
            cache.append(_verified_to_device(trainer, x, y, source))
        else:
            j = rng.randint(0, i)
            if j < cap:
                cache[j] = _verified_to_device(trainer, x, y, source)
    return cache


def _build_gpu_val_cache(trainer):
    """Install the validation cache every DECISION read is served from.

    This is the tuner's path (``validate_n_batches`` -> ``iter_validation_batches``
    -> here, and ``validate_correctness_on_indices``), so a corrupted read here
    does not merely print a wrong metric: it calibrates the acceptance baseline
    and drives every ladder commit/rollback. It carries the same detect-and-repair
    discipline as the single-batch reporting path, and the discipline is CHEAPER
    to keep here: the cache is owned, so verifying it once makes every later read
    of it provably clean for the trainer's whole life.
    """
    max_batches = getattr(trainer, "_val_cache_max_batches", None)
    factory = _shared_eval_cache_seam(trainer)
    key = None if factory is None else _shared_eval_cache_key(trainer, max_batches)
    source = _val_source(trainer)
    if factory is not None:
        shared = factory.get_eval_cache(key)
        if shared is not None:
            # Pooled caches are only ever produced below, hence already verified.
            # Re-checking makes that a LOCAL fact rather than a cross-call-site
            # assumption, for one on-device scan per trainer.
            batch_integrity.verify_owned_batches(
                shared, source=f"the pooled eval cache for {source}"
            )
            trainer._gpu_val_cache = shared
            trainer._gpu_val_cursor = 0
            return

    cache = batch_integrity.verified_pass(
        lambda: _materialize_val_cache(trainer, max_batches, source),
        source=source,
    )
    if factory is not None:
        factory.put_eval_cache(key, cache)
    trainer._gpu_val_cache = cache
    trainer._gpu_val_cursor = 0


def iter_validation_batches(trainer, n_batches: int):
    if getattr(trainer, "_gpu_val_cache", None) is None:
        _build_gpu_val_cache(trainer)
    cache = trainer._gpu_val_cache
    if not cache:
        return
    for _ in range(n_batches):
        yield cache[trainer._gpu_val_cursor % len(cache)]
        trainer._gpu_val_cursor += 1


def validate_correctness_on_indices(trainer, batch_indices):
    """Per-example correctness (bool list) over fixed validation-cache batches.

    Reads only the validation cache (never the test set) and scores the same
    examples each call so reference and candidate are paired.
    """
    if getattr(trainer, "_gpu_val_cache", None) is None:
        _build_gpu_val_cache(trainer)
    cache = trainer._gpu_val_cache
    if not cache:
        return []
    trainer.model.eval()
    correct: list[bool] = []
    with torch.no_grad(), metric_grade_eval(trainer.device):
        for idx in batch_indices:
            x, y = cache[idx % len(cache)]
            x, y = x.to(trainer.device), y.to(trainer.device)
            _, predicted = trainer.model(x).max(1)
            correct.extend(bool(v) for v in predicted.eq(y).tolist())
    return correct


def _test_pass(trainer, max_batches, source: str) -> tuple[float, float]:
    total = 0.0
    correct = 0.0
    with torch.no_grad(), metric_grade_eval(trainer.device):
        for batch_idx, (x, y) in enumerate(trainer.test_loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            trainer.model.eval()
            trainer.model = trainer.model.to(trainer.device)
            x, y = _verified_to_device(trainer, x, y, source)
            _, predicted = trainer.model(x).max(1)
            total += float(y.size(0))
            correct += float(predicted.eq(y).sum().item())
    return correct, total


def test(trainer, max_batches: int | None = None):
    """The published test number. Restarted whole on a corrupted read; never partial.

    ``_report`` fires once, outside the retry, so a restarted pass cannot emit a
    number for an aborted pass.
    """
    source = _test_source(trainer)
    correct, total = batch_integrity.verified_pass(
        lambda: _test_pass(trainer, max_batches, source), source=source,
    )
    if total <= 0:
        return 0.0
    acc = correct / total
    trainer._report("Test accuracy", acc)
    return acc


def validate_on_loader(trainer, x, y):
    total = 0
    correct = 0
    with torch.no_grad(), metric_grade_eval(trainer.device):
        trainer.model = trainer.model.to(trainer.device)
        x, y = x.to(trainer.device), y.to(trainer.device)
        _, predicted = trainer.model(x).max(1)
        total += float(y.size(0))
        correct += float(predicted.eq(y).sum().item())
    return correct / total


def validate(trainer):
    x, y = trainer.next_validation_batch()
    trainer.model.eval()
    acc = validate_on_loader(trainer, x.to(trainer.device), y.to(trainer.device))
    trainer._report(trainer._validation_metric_name("Validation accuracy"), acc)
    return acc


def validate_n_batches(trainer, n_batches: int) -> float:
    if n_batches <= 0:
        return 0.0
    trainer.model.eval()
    total = 0
    correct = 0
    with torch.no_grad(), metric_grade_eval(trainer.device):
        for x, y in trainer.iter_validation_batches(int(n_batches)):
            x, y = x.to(trainer.device), y.to(trainer.device)
            _, predicted = trainer.model(x).max(1)
            total += float(y.size(0))
            correct += float(predicted.eq(y).sum().item())
    acc = correct / total if total else 0.0
    trainer._report(trainer._validation_metric_name("Validation accuracy"), acc)
    return acc


def _repair_training_read(trainer, _attempt: int) -> None:
    """Fresh training iterator: abandons the poisoned in-flight queue.

    No positional advance here, unlike the validation read: the training loader is
    SHUFFLED, so a fresh iterator already draws a different batch on every attempt
    and progress through unexamined data is guaranteed without one -- hence a skip
    of 0 through the shared repair primitive.
    """
    trainer.train_iter = batch_integrity.restart_and_advance(
        lambda: iter(trainer.train_loader), 0,
    )


def validate_train(trainer):
    """Reported "Validation accuracy on train set" -- guarded like every other
    reported number. The guard sits HERE and not in ``next_training_batch``
    because that method is also the training loop's hot path, where a finiteness
    probe would cost a device sync per optimizer step; a non-finite training batch
    is already handled there by the AMP scaler, which skips the step."""
    x, y = batch_integrity.read_verified_batch(
        lambda: batch_integrity.own_batch(*trainer.next_training_batch()),
        source=f"the training loader ({type(getattr(trainer, 'train_loader', None)).__name__})",
        repair=lambda attempt: _repair_training_read(trainer, attempt),
    )
    trainer.model.train()
    acc = validate_on_loader(trainer, x.to(trainer.device), y.to(trainer.device))
    trainer._report(
        trainer._validation_metric_name("Validation accuracy on train set"), acc
    )
    return acc


def evaluate_loss_on_batch(trainer, batch) -> float:
    x, y = batch
    trainer.model.eval()
    with torch.no_grad():
        x, y = x.to(trainer.device), y.to(trainer.device)
        loss = trainer.loss_function(trainer.model, x, y)
    return float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)
