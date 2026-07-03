"""Mode-agnostic DFQ per-neuron bias-correction core."""

from __future__ import annotations

import torch


def channel_mean(t: torch.Tensor) -> torch.Tensor:
    """Mean over all dims except the last (per output channel)."""
    return t.reshape(-1, t.shape[-1]).float().mean(0)


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
    return {k: channel_mean(v) for k, v in teacher_activation_samples(teacher, cal_x).items()}


def mean_abs_gap(ann_mean: dict, cascade: dict) -> float:
    """Mean over perceptrons of the per-channel ``|cascade_mean - ann_mean|``."""
    gaps = []
    for k, ann_mu in ann_mean.items():
        cascade_value = cascade.get(k)
        if cascade_value is None:
            continue
        cm = channel_mean(cascade_value)
        n = min(cm.numel(), ann_mu.numel())
        gaps.append((cm[:n] - ann_mu[:n]).abs().mean().item())
    return sum(gaps) / max(1, len(gaps))


def dfq_correct_biases(
    model, ann_mean: dict, cascade_means_fn, *, bias_iters: int, eta: float,
) -> dict:
    """Run ``bias_iters`` rounds of DFQ per-neuron bias correction in place.

    ``cascade_means_fn()`` returns a fresh ``{perceptron_index: decoded_value}``
    map, re-measured each round since the biases move.
    """
    bias_iters = max(0, int(bias_iters))
    eta = float(eta)
    perceptrons = list(model.get_perceptrons())

    gap_before = mean_abs_gap(ann_mean, cascade_means_fn())
    for _ in range(bias_iters):
        cascade = cascade_means_fn()
        for k, perceptron in enumerate(perceptrons):
            cascade_value = cascade.get(k)
            bias = getattr(perceptron.layer, "bias", None)
            if cascade_value is None or k not in ann_mean or bias is None:
                continue
            cm = channel_mean(cascade_value)
            ann_mu = ann_mean[k]
            n = min(cm.numel(), ann_mu.numel(), bias.numel())
            with torch.no_grad():
                bias[:n] += eta * (ann_mu[:n] - cm[:n]).to(bias.device, bias.dtype)
    gap_after = mean_abs_gap(ann_mean, cascade_means_fn())

    return {"mean_gap_before": gap_before, "mean_gap_after": gap_after}
