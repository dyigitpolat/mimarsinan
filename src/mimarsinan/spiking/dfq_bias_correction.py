"""Mode-agnostic DFQ per-neuron bias-correction core (channel-axis-correct, mask-aware, probe-ratcheted)."""

from __future__ import annotations

from typing import Callable

import torch

from mimarsinan.models.perceptron_mixer.perceptron import activation_channel_axis
from mimarsinan.transformations.pruning.committed_masks import (
    commit_perceptron_pruning,
)


def channel_mean(t: torch.Tensor, axis: int) -> torch.Tensor:
    """Mean over all dims except ``axis`` (the per-output-channel mean)."""
    if axis in (-1, t.dim() - 1):
        return t.reshape(-1, t.shape[-1]).float().mean(0)
    dims = [d for d in range(t.dim()) if d != axis]
    return t.float().mean(dim=dims)


def perceptron_channel_mean(perceptron, t: torch.Tensor) -> torch.Tensor:
    """``channel_mean`` on the perceptron's owner-declared channel axis (fails loud when ambiguous)."""
    return channel_mean(t, activation_channel_axis(perceptron, t))


def teacher_activation_samples(teacher, cal_x) -> dict:
    """Capture each teacher perceptron's activation output on ``cal_x``."""
    samples: dict = {}
    handles = [
        perceptron.activation.register_forward_hook(
            lambda _m, _i, out, k=k: samples.__setitem__(k, out.detach())
        )
        for k, perceptron in enumerate(teacher.get_perceptrons())
        if hasattr(perceptron, "activation")
    ]
    try:
        with torch.no_grad():
            teacher(cal_x)
    finally:
        for handle in handles:
            handle.remove()
    return samples


def teacher_channel_means(teacher, cal_x) -> dict:
    """Per-perceptron teacher activation channel-mean keyed by perceptron index."""
    perceptrons = list(teacher.get_perceptrons())
    return {
        k: perceptron_channel_mean(perceptrons[k], v)
        for k, v in teacher_activation_samples(teacher, cal_x).items()
    }


def mean_abs_gap(perceptrons, ann_mean: dict, cascade: dict) -> float:
    """Mean over perceptrons of the per-channel ``|cascade_mean - ann_mean|``."""
    perceptrons = list(perceptrons)
    gaps = []
    for k, ann_mu in ann_mean.items():
        cascade_value = cascade.get(k)
        if cascade_value is None:
            continue
        cm = perceptron_channel_mean(perceptrons[k], cascade_value)
        n = min(cm.numel(), ann_mu.numel())
        gaps.append((cm[:n] - ann_mu[:n]).abs().mean().item())
    return sum(gaps) / max(1, len(gaps))


def _live_bias_delta(perceptron, delta: torch.Tensor) -> torch.Tensor:
    """Zero the delta on structurally-dead rows (committed ``prune_bias_mask``)."""
    mask = getattr(perceptron.layer, "prune_bias_mask", None)
    if mask is None:
        return delta
    return delta * (~mask[: delta.numel()]).to(delta.device, delta.dtype)


def _bias_snapshot(perceptrons) -> dict:
    return {
        k: p.layer.bias.detach().clone()
        for k, p in enumerate(perceptrons)
        if getattr(p.layer, "bias", None) is not None
    }


def _restore_biases(perceptrons, snapshot: dict) -> None:
    with torch.no_grad():
        for k, saved in snapshot.items():
            perceptrons[k].layer.bias.data.copy_(saved)


def dfq_correct_biases(
    model,
    ann_mean: dict,
    cascade_means_fn,
    *,
    bias_iters: int,
    eta: float,
    probe: Callable[[], float] | None = None,
    probe_patience: int | None = None,
) -> dict:
    """Run up to ``bias_iters`` rounds of DFQ per-neuron bias correction in place.

    ``cascade_means_fn()`` returns a fresh ``{perceptron_index: decoded_value}``
    map, re-measured each round since the biases move. With a ``probe`` (a
    deployed-behavior accuracy read), the loop keeps the best-probe bias state
    (never worse than iteration-0) and stops early after ``probe_patience``
    iterations without a new best.
    """
    bias_iters = max(0, int(bias_iters))
    eta = float(eta)
    perceptrons = list(model.get_perceptrons())

    # W-CAL-2: the deployed cascade reads raw params (hooks never fire there),
    # so calibration must start from the committed-pruning state.
    for perceptron in perceptrons:
        commit_perceptron_pruning(perceptron)

    gap_before = mean_abs_gap(perceptrons, ann_mean, cascade_means_fn())

    best_probe = -float("inf")
    best_iter = 0
    best_snapshot: dict = {}
    probe_entry = None
    probe_curve: list[float] = []
    iters_run = 0
    if probe is not None:
        probe_entry = float(probe())
        best_probe = probe_entry
        best_snapshot = _bias_snapshot(perceptrons)

    for iteration in range(1, bias_iters + 1):
        cascade = cascade_means_fn()
        for k, perceptron in enumerate(perceptrons):
            cascade_value = cascade.get(k)
            bias = getattr(perceptron.layer, "bias", None)
            if cascade_value is None or k not in ann_mean or bias is None:
                continue
            cm = perceptron_channel_mean(perceptron, cascade_value)
            ann_mu = ann_mean[k]
            n = min(cm.numel(), ann_mu.numel(), bias.numel())
            delta = _live_bias_delta(
                perceptron, eta * (ann_mu[:n] - cm[:n]).to(bias.device, bias.dtype),
            )
            with torch.no_grad():
                bias[:n] += delta
        iters_run = iteration
        if probe is None:
            continue
        probe_value = float(probe())
        probe_curve.append(probe_value)
        if probe_value > best_probe:
            best_probe = probe_value
            best_iter = iteration
            best_snapshot = _bias_snapshot(perceptrons)
        elif probe_patience is not None and iteration - best_iter >= int(probe_patience):
            break

    stats = {}
    if probe is not None:
        _restore_biases(perceptrons, best_snapshot)
        stats.update({
            "probe_entry": probe_entry,
            "probe_curve": probe_curve,
            "probe_best": best_probe,
            "probe_best_iter": best_iter,
            "probe_iters_run": iters_run,
        })

    gap_after = mean_abs_gap(perceptrons, ann_mean, cascade_means_fn())
    stats.update({"mean_gap_before": gap_before, "mean_gap_after": gap_after})
    return stats
