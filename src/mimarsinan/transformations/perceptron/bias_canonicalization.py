"""Guarded empirical bias canonicalization at the QAT entry (data-verified)."""

from __future__ import annotations

import torch

from mimarsinan.models.perceptron_mixer.perceptron import activation_channel_axis
from mimarsinan.transformations.perceptron.bias_saturation import (
    _EMPIRICAL_MARGIN,
    _STARVATION_ZERO_FRAC,
    clip_off_saturated_effective_bias,
    empirical_bias_shift,
    predicted_weight_zero_fraction,
)
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)


@torch.no_grad()
def _channel_preactivation_range(perceptron, model, batches, device):
    """Observed per-channel (min, max) of the perceptron's activation input."""
    v_min = v_max = None
    handle = None

    def pre_hook(_m, inp):
        nonlocal v_min, v_max
        t = inp[0].detach()
        axis = activation_channel_axis(perceptron, t)
        dims = [d for d in range(t.dim()) if d != axis]
        lo = t.amin(dim=dims).float().cpu()
        hi = t.amax(dim=dims).float().cpu()
        v_min = lo if v_min is None else torch.minimum(v_min, lo)
        v_max = hi if v_max is None else torch.maximum(v_max, hi)

    handle = perceptron.activation.register_forward_pre_hook(pre_hook)
    try:
        for x in batches:
            model(x.to(device))
    finally:
        handle.remove()
    return v_min, v_max


@torch.no_grad()
def _decisions(model, batches, device):
    return [model(x.to(device)).argmax(-1).cpu() for x in batches]


def _apply_decision_verified(perceptron, model, batches, device, reference, apply) -> bool:
    """Run ``apply()`` and verify decision-invariance on the calibration batches;
    any flip restores the perceptron's weight/bias/normalization state exactly.
    Returns True when the change is kept."""
    snapshot = (
        perceptron.layer.weight.data.clone(),
        perceptron.layer.bias.data.clone(),
        {k: v.clone() for k, v in perceptron.normalization.state_dict().items()},
    )
    apply()
    verified = _decisions(model, batches, device)
    if all(torch.equal(a, b) for a, b in zip(reference, verified)):
        return True
    perceptron.layer.weight.data.copy_(snapshot[0])
    perceptron.layer.bias.data.copy_(snapshot[1])
    perceptron.normalization.load_state_dict(snapshot[2])
    return False


def canonicalize_starved_bias_outliers(
    model,
    calibration_batches,
    *,
    bits: int,
    ceiling: float = 1.0,
    margin: float = _EMPIRICAL_MARGIN,
) -> dict:
    """Guarded, VERIFIED canonicalization of grid-starving bias outliers the provable clip cannot reach.

    Arms only perceptrons whose predicted weight-zero fraction at the target grid
    is >= 0.8; each guarded rung is VERIFIED (a decision flip restores state exactly).
    """
    report = {"clipped": 0, "restored": 0, "removed": 0, "removal_restored": 0}
    batches = [x for x in calibration_batches]
    if not batches:
        return report
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    transformer = PerceptronTransformer()
    try:
        armed = []
        for perceptron in model.get_perceptrons():
            if getattr(perceptron, "is_encoding_layer", False):
                continue
            if perceptron.layer.bias is None:
                continue
            clip_off_saturated_effective_bias(perceptron)
            if predicted_weight_zero_fraction(perceptron, bits) >= _STARVATION_ZERO_FRAC:
                armed.append(perceptron)
        if not armed:
            return report
        reference = _decisions(model, batches, device)
        for perceptron in armed:
            v_min, v_max = _channel_preactivation_range(
                perceptron, model, batches, device,
            )
            if v_min is None or v_max is None:
                continue
            lo: torch.Tensor = v_min
            hi: torch.Tensor = v_max

            # The pre-activation range is in the RAW (theta) domain while the
            # transform runs in the effective (normalized) domain: the clamp
            # ceiling is theta, and raw shifts divide by theta on the way back.
            theta = float(
                perceptron.activation_scale.detach().float().mean()
            ) * float(ceiling)
            moved = 0

            def shift(effective_bias, theta=theta, lo=lo, hi=hi):
                nonlocal moved
                n = min(effective_bias.numel(), lo.numel())
                delta_raw = empirical_bias_shift(
                    (effective_bias[:n].detach().float().cpu() * theta),
                    lo[:n], hi[:n],
                    ceiling=theta, margin=margin * theta,
                )
                delta = torch.zeros_like(effective_bias)
                delta[:n] = (delta_raw / theta).to(
                    effective_bias.device, effective_bias.dtype,
                )
                moved = int((delta.abs() > 0).sum())
                return effective_bias + delta

            kept = _apply_decision_verified(
                perceptron, model, batches, device, reference,
                lambda: transformer.apply_effective_bias_transform(perceptron, shift),
            )
            if moved == 0:
                continue
            if kept:
                report["clipped"] += moved
                print(
                    f"[BiasSaturation] {getattr(perceptron, 'name', '<unnamed>')}: "
                    f"guarded-canonicalized {moved} empirically-saturated bias "
                    "outlier(s) (decision agreement verified)"
                )
            else:
                report["restored"] += 1
                print(
                    f"[BiasSaturation] {getattr(perceptron, 'name', '<unnamed>')}: "
                    "guarded canonicalization RESTORED (decision flip on "
                    "calibration data)"
                )
        for perceptron in armed:
            _remove_nuisance_grid_dominators(
                perceptron, model, batches, device, bits,
                reference, transformer, report,
            )
    finally:
        model.train(was_training)
    return report


def _remove_nuisance_grid_dominators(
    perceptron, model, batches, device, bits, reference, transformer, report,
) -> None:
    """Rung 3: while the grid stays starved, zero grid-dominating channels
    whose REMOVAL is decision-invariant on the calibration batches (the
    rep3/ch43 class: wild degenerate-stats channels the QAT cannot represent
    at the shared grid). Each removal is verified; a flip restores exactly."""
    attempts = 0
    while (
        predicted_weight_zero_fraction(perceptron, bits) >= _STARVATION_ZERO_FRAC
        and attempts < 8
    ):
        attempts += 1
        b = transformer.get_effective_bias(perceptron).detach()
        w = transformer.get_effective_weight(perceptron).detach()
        w_max = float(w.abs().max())
        c = int(b.abs().argmax())
        if float(b.abs()[c]) <= w_max:
            return

        def zero_channel(t, c=c):
            out = t.clone()
            out[c] = 0.0
            return out

        def remove_channel(zero_channel=zero_channel):
            transformer.apply_effective_weight_transform(perceptron, zero_channel)
            transformer.apply_effective_bias_transform(perceptron, zero_channel)

        if _apply_decision_verified(
            perceptron, model, batches, device, reference, remove_channel,
        ):
            report["removed"] += 1
            print(
                f"[BiasSaturation] {getattr(perceptron, 'name', '<unnamed>')}: "
                f"removed grid-dominating nuisance channel {c} "
                "(decision agreement verified)"
            )
        else:
            report["removal_restored"] += 1
            print(
                f"[BiasSaturation] {getattr(perceptron, 'name', '<unnamed>')}: "
                f"nuisance-channel removal RESTORED for channel {c} "
                "(decision flip on calibration data)"
            )
            return
