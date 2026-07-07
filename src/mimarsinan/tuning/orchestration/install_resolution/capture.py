"""[MBH-A6] channel-resolved activation capture for install-resolution gauges."""

from __future__ import annotations

from typing import Iterable, List

import torch

from mimarsinan.models.nn.layers import TransformedActivation
from mimarsinan.tuning.orchestration.mbh_ledger import _measurement_guard

_MAX_ROWS_PER_BATCH = 4096


class ChannelStatsAccumulator:
    """Duck-typed activation decorator accumulating channel-resolved positives.

    Rows are deterministically subsampled per batch; values live on CPU so the
    capture never pins GPU activations.
    """

    def __init__(self, max_rows_per_batch: int = _MAX_ROWS_PER_BATCH):
        self._max_rows = int(max_rows_per_batch)
        self._chunks: List[torch.Tensor] = []

    def input_transform(self, x):
        return x

    def output_transform(self, x):
        if x.dim() > 1:
            rows = x.detach().to(torch.float32).moveaxis(1, -1).reshape(-1, x.shape[1])
            if rows.shape[0] > self._max_rows:
                idx = torch.linspace(
                    0, rows.shape[0] - 1, steps=self._max_rows, device=rows.device
                ).round().long()
                rows = rows.index_select(0, idx)
            self._chunks.append(rows.cpu())
        return x

    def _rows(self) -> torch.Tensor:
        if not self._chunks:
            return torch.empty(0, 0)
        return torch.cat(self._chunks, dim=0)

    def per_channel_q99(self) -> List[float]:
        """Count-based q99 of each channel's positives (no positives -> 0.0)."""
        rows = self._rows()
        out: List[float] = []
        for c in range(rows.shape[1]):
            positives = rows[:, c][rows[:, c] > 0.0]
            if positives.numel() == 0:
                out.append(0.0)
            else:
                out.append(
                    float(torch.quantile(positives, 0.99, interpolation="higher"))
                )
        return out

    def positive_values(self) -> List[float]:
        rows = self._rows()
        if rows.numel() == 0:
            return []
        flat = rows.reshape(-1)
        return flat[flat > 0.0].tolist()

    def mean_positive(self) -> float:
        values = self.positive_values()
        if not values:
            return 0.0
        return float(sum(values) / len(values))


def attach_activation_decorator(perceptron, decorator):
    """Attach a duck-typed decorator to a perceptron activation; return cleanup."""
    activation = perceptron.activation
    if hasattr(activation, "decorate") and hasattr(activation, "pop_decorator"):
        activation.decorate(decorator)
        return lambda: activation.pop_decorator()
    wrapped = TransformedActivation(activation, [decorator])
    perceptron.set_activation(wrapped)
    return lambda: perceptron.set_activation(activation)


def capture_install_stats(tuner, n_batches: int = 2) -> List[tuple]:
    """Cursor-isolated channel-stats capture of the tuner's LIVE model at the
    install anchor: RNG is forked and the trainer's validation cursor restored,
    so the live trajectory is untouched.

    A fresh trainer has no cursor yet; the cache content is deterministic, so
    pre-building it and rewinding to 0 is bit-invariant for consumers.
    """
    prev_cursor = getattr(tuner.trainer, "_gpu_val_cursor", None)
    with _measurement_guard(tuner.trainer):
        batches = [
            x for x, _ in tuner.trainer.iter_validation_batches(int(n_batches))
        ]
        stats = collect_channel_stats(
            tuner.model, batches, tuner.pipeline.config["device"],
        )
    tuner.trainer._gpu_val_cursor = 0 if prev_cursor is None else prev_cursor
    return stats


def collect_channel_stats(
    model, input_batches: Iterable, device
) -> List[tuple]:
    """Run ``input_batches`` through ``model`` with per-perceptron accumulators.

    Returns ``[(perceptron, ChannelStatsAccumulator), ...]``; the activation
    stack is restored even when the forward raises.
    """
    perceptrons = list(model.get_perceptrons())
    accumulators = [ChannelStatsAccumulator() for _ in perceptrons]
    cleanups = [
        attach_activation_decorator(perceptron, acc)
        for perceptron, acc in zip(perceptrons, accumulators)
    ]
    was_training = bool(getattr(model, "training", False))
    try:
        model.eval()
        with torch.no_grad():
            for x in input_batches:
                model(x.to(device))
    finally:
        for cleanup in reversed(cleanups):
            cleanup()
        if was_training:
            model.train()
    return list(zip(perceptrons, accumulators))
