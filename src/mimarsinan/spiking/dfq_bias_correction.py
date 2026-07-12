"""Mode-agnostic DFQ per-neuron bias-correction core (channel-axis-correct, mask-aware, probe-ratcheted)."""

from __future__ import annotations

from typing import Callable

import torch

from mimarsinan.models.perceptron_mixer.perceptron import activation_channel_axis
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)
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


def _activation_hook_samples(model, cal_x, capture, indices=None) -> dict:
    """One no-grad forward with per-perceptron activation hooks; ``capture``
    picks the tensor from ``(hook_inputs, hook_output)``."""
    wanted = None if indices is None else set(indices)
    samples: dict = {}
    handles = [
        perceptron.activation.register_forward_hook(
            lambda _m, inp, out, k=k: samples.__setitem__(k, capture(inp, out))
        )
        for k, perceptron in enumerate(model.get_perceptrons())
        if hasattr(perceptron, "activation") and (wanted is None or k in wanted)
    ]
    try:
        with torch.no_grad():
            model(cal_x)
    finally:
        for handle in handles:
            handle.remove()
    return samples


def teacher_activation_samples(teacher, cal_x) -> dict:
    """Capture each teacher perceptron's activation output on ``cal_x``."""
    return _activation_hook_samples(
        teacher, cal_x, lambda _inp, out: out.detach(),
    )


def teacher_channel_means(teacher, cal_x) -> dict:
    """Per-perceptron teacher activation channel-mean keyed by perceptron index."""
    perceptrons = list(teacher.get_perceptrons())
    return {
        k: perceptron_channel_mean(perceptrons[k], v)
        for k, v in teacher_activation_samples(teacher, cal_x).items()
    }


def perceptron_preactivation_samples(model, cal_x, indices=None) -> dict:
    """Capture each perceptron's activation INPUT (the value-domain
    pre-activation) on ``cal_x`` — the input-side twin of
    :func:`teacher_activation_samples`. ``indices`` restricts the capture."""
    return _activation_hook_samples(
        model, cal_x, lambda inp, _out: inp[0].detach(), indices,
    )


def preactivation_channel_means(model, cal_x) -> dict:
    """Per-perceptron pre-activation channel-mean keyed by perceptron index."""
    perceptrons = list(model.get_perceptrons())
    return {
        k: perceptron_channel_mean(perceptrons[k], v)
        for k, v in perceptron_preactivation_samples(model, cal_x).items()
    }


def _effective_bias_shift(perceptron, value_delta: torch.Tensor) -> None:
    """Subtract a value-domain pre-activation ``value_delta`` via the effective
    bias (θ-normalized), per-channel-θ aware."""
    theta = torch.as_tensor(perceptron.activation_scale).detach()
    theta = theta.to(value_delta.device, value_delta.dtype)
    shift = value_delta / (
        theta.reshape(-1) if theta.numel() == value_delta.numel() else theta
    )
    PerceptronTransformer().apply_effective_bias_transform(
        perceptron, lambda b, s=shift: b - s.to(b.device, b.dtype),
    )


def sequential_first_moment_fold(
    model,
    float_preact_mean: dict,
    cal_x,
    *,
    own_offsets: dict,
    hop_order=None,
    baked_flag: str = "_first_moment_folded",
) -> dict:
    """[S3] Sequential per-hop first-moment bias fold, input→output.

    Per hop k: fold ``b_eff -= (E[preact_deployed - preact_float] - own)/θ``,
    the deployed estimate measured through the ALREADY-FOLDED prefix (a fresh
    forward per hop). ``own_offsets[k]`` is the hop's INTENTIONAL value-domain
    offset (the baked +θ/(2S) half-step): omitting it folds the raw gap and
    cancels the mid-tread compensation — the §3.2 sign trap (0.93→0.59).
    """
    perceptrons = list(model.get_perceptrons())
    # W-CAL-2: calibration must start from the committed-pruning state.
    for perceptron in perceptrons:
        commit_perceptron_pruning(perceptron)

    order = list(hop_order) if hop_order is not None else list(range(len(perceptrons)))
    folded = 0
    skipped = 0
    per_hop_abs_delta: dict = {}
    for k in order:
        perceptron = perceptrons[k]
        mu_float = float_preact_mean.get(k)
        bias = getattr(perceptron.layer, "bias", None)
        if (
            mu_float is None
            or bias is None
            or getattr(perceptron, baked_flag, False)
        ):
            skipped += 1
            continue
        value = perceptron_preactivation_samples(model, cal_x, indices=(k,)).get(k)
        if value is None:
            skipped += 1
            continue
        cm = perceptron_channel_mean(perceptron, value)
        mu = mu_float.to(cm.device, cm.dtype)
        n = min(cm.numel(), mu.numel(), bias.numel())
        own = torch.as_tensor(
            own_offsets.get(k, 0.0), dtype=cm.dtype, device=cm.device,
        )
        delta = (cm[:n] - mu[:n]) - (own.reshape(-1)[:n] if own.dim() else own)
        delta = _live_bias_delta(perceptron, delta)
        full = torch.zeros(bias.numel(), dtype=bias.dtype, device=bias.device)
        full[:n] = delta.to(bias.device, bias.dtype)
        _effective_bias_shift(perceptron, full)
        setattr(perceptron, baked_flag, True)
        folded += 1
        per_hop_abs_delta[k] = float(delta.abs().mean())

    return {
        "folded": folded,
        "skipped": skipped,
        "per_hop_abs_delta": per_hop_abs_delta,
        "mean_abs_delta": (
            sum(per_hop_abs_delta.values()) / len(per_hop_abs_delta)
            if per_hop_abs_delta else 0.0
        ),
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
