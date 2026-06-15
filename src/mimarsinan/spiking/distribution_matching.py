"""Distribution-matching calibration for the genuine TTFS single-spike cascade.

Two stages, both grounded in the teacher ANN's activation distribution:

1. **Scale-aware boundaries** — each block's ``theta_out`` is the teacher's
   per-perceptron activation quantile, so the [0,1] TTFS window normalizes that
   block's output (and the downstream input un-normalizes it).
2. **DFQ per-neuron bias correction** — a few rounds of matching each
   perceptron's *cascade* channel-mean to the ANN's by nudging ``layer.bias``.
   Raising a starved neuron's membrane baseline revives it (reversing the
   death-cascade collapse) while matching the first moment.

Together they shrink the cascade↔ANN first-moment gap, turning the full TTFS
transform into a smoothly recoverable teacher->genuine ramp. The model must
already be in the deployed single-spike-cascade TTFS state (the caller runs the
finalize rebuild before calibrating).
"""

from __future__ import annotations

import torch

from mimarsinan.spiking.scale_aware_boundaries import calibrate_scale_aware_boundaries
from mimarsinan.spiking.segment_partition import perceptron_of


def _channel_mean(t: torch.Tensor) -> torch.Tensor:
    """Mean over all dims except the last (per output channel)."""
    return t.reshape(-1, t.shape[-1]).float().mean(0)


def _teacher_activation_samples(teacher, cal_x):
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


def _cascade_channel_means(model, cal_x, T):
    """Per-perceptron cascade decoded value, keyed by perceptron index."""
    # Imported lazily: ``ttfs_segment_forward`` pulls in ``mimarsinan.spiking``,
    # so a module-level import here would close an import cycle.
    from mimarsinan.models.spiking.training.ttfs_segment_forward import (
        TTFSSegmentForward,
    )

    with torch.no_grad():
        _, node_values = TTFSSegmentForward(
            model.get_mapper_repr(), T
        ).forward_with_node_values(cal_x)
    by_perceptron = {
        id(perceptron_of(node)): value
        for node, value in node_values.items()
        if perceptron_of(node) is not None
    }
    out: dict = {}
    for k, perceptron in enumerate(model.get_perceptrons()):
        value = by_perceptron.get(id(perceptron))
        if value is not None:
            out[k] = value
    return out


def _mean_abs_gap(model, ann_mean, cal_x, T):
    cascade = _cascade_channel_means(model, cal_x, T)
    gaps = []
    for k, ann_mu in ann_mean.items():
        cascade_value = cascade.get(k)
        if cascade_value is None:
            continue
        cm = _channel_mean(cascade_value)
        n = min(cm.numel(), ann_mu.numel())
        gaps.append((cm[:n] - ann_mu[:n]).abs().mean().item())
    return sum(gaps) / max(1, len(gaps))


def _dead_fraction(model, cal_x, T):
    cascade = _cascade_channel_means(model, cal_x, T)
    return {
        k: (value.abs() < 1e-6).float().mean().item()
        for k, value in cascade.items()
    }


def match_activation_distributions(
    model,
    teacher,
    cal_x,
    T,
    *,
    quantile: float = 0.99,
    bias_iters: int = 15,
    eta: float = 0.7,
):
    """Match the deployed TTFS cascade's activation distribution to the teacher ANN's.

    Calibrates scale-aware [0,1] boundaries from the teacher's per-perceptron
    activation ``quantile``, then runs ``bias_iters`` rounds of DFQ per-neuron
    bias correction so each perceptron's cascade channel-mean tracks the ANN's
    (``bias += eta * (ann_mean - cascade_mean)``). The model is mutated in place
    and must already be in the deployed single-spike-cascade TTFS state.

    Returns a small stats dict. ``mean_gap_before``/``mean_gap_after`` bracket
    the DFQ loop (both measured after boundary calibration, the prototype's
    "pre-correction" anchor): the cascade↔ANN channel-mean ``|gap|`` shrinks as
    the per-neuron means converge. ``dead_fraction_before``/``after`` are the
    matching total %dead over the same window.
    """
    T = int(T)
    perceptrons = list(model.get_perceptrons())
    n_perceptrons = len(perceptrons)

    ann_samples = _teacher_activation_samples(teacher, cal_x)
    ann_mean = {k: _channel_mean(v) for k, v in ann_samples.items()}
    theta_out = [
        max(float(torch.quantile(ann_samples[k].abs().float().flatten(), quantile)), 1e-2)
        if k in ann_samples else 1.0
        for k in range(n_perceptrons)
    ]

    calibrate_scale_aware_boundaries(model, theta_out)

    gap_before = _mean_abs_gap(model, ann_mean, cal_x, T)
    dead_before = _dead_fraction(model, cal_x, T)

    for _ in range(int(bias_iters)):
        cascade = _cascade_channel_means(model, cal_x, T)
        for k, perceptron in enumerate(perceptrons):
            cascade_value = cascade.get(k)
            bias = getattr(perceptron.layer, "bias", None)
            if cascade_value is None or k not in ann_mean or bias is None:
                continue
            cm = _channel_mean(cascade_value)
            ann_mu = ann_mean[k]
            n = min(cm.numel(), ann_mu.numel(), bias.numel())
            with torch.no_grad():
                bias[:n] += eta * (ann_mu[:n] - cm[:n]).to(bias.device, bias.dtype)

    gap_after = _mean_abs_gap(model, ann_mean, cal_x, T)
    dead_after = _dead_fraction(model, cal_x, T)

    return {
        "mean_gap_before": gap_before,
        "mean_gap_after": gap_after,
        "dead_fraction_before": sum(dead_before.values()),
        "dead_fraction_after": sum(dead_after.values()),
        "quantile": float(quantile),
        "bias_iters": int(bias_iters),
        "eta": float(eta),
        "num_perceptrons": n_perceptrons,
    }
