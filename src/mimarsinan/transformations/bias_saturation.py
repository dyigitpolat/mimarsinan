"""Saturation-equivalent effective-bias canonicalization for shared-grid quantization."""

from __future__ import annotations

import torch

from mimarsinan.models.perceptron_mixer.perceptron import activation_channel_axis
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)

_STARVATION_ZERO_FRAC = 0.8
"""Predicted weight-zero fraction at the target grid above which a perceptron
is starved enough to arm the guarded empirical pass."""

_EMPIRICAL_MARGIN = 0.25
"""Saturation slack (in ceiling units) the empirical bias shift must keep."""


def off_saturation_bias_bound(effective_weight: torch.Tensor) -> torch.Tensor:
    """Per-output-channel bound ``-pos_reach``: the largest effective bias at
    which a floor-0 channel stays constant-OFF for every input in ``[0, 1]``."""
    w = (
        effective_weight.flatten(1)
        if effective_weight.dim() > 1
        else effective_weight.unsqueeze(1)
    )
    return -w.clamp(min=0).sum(dim=1)


def clip_off_saturated_effective_bias(perceptron) -> int:
    """Clip constant-OFF channels' effective bias to the saturation bound.

    A channel with ``b + pos_reach <= 0`` outputs the activation floor (0) for
    every normalized input in [0, 1]; bias mass below ``-pos_reach`` is
    functionally unobservable, yet it participates in the shared per-perceptron
    weight/bias quantization scale and can starve the whole weight grid (the
    t01_19/t0_03 WQ-entry crater: one dead-channel b_eff = -12.6 zeroed 100%
    of a layer's 4-bit weights). Function-preserving and idempotent; encoding
    layers are skipped (their input domain is not the normalized [0, 1]).
    Returns the number of channels clipped.
    """
    if getattr(perceptron, "is_encoding_layer", False):
        return 0
    transformer = PerceptronTransformer()
    bound = off_saturation_bias_bound(
        transformer.get_effective_weight(perceptron).detach()
    )
    clipped = 0
    max_delta = 0.0

    def clip(effective_bias):
        nonlocal clipped, max_delta
        limit = bound.to(effective_bias.device, effective_bias.dtype)
        below = effective_bias < limit
        clipped = int(below.sum())
        if clipped:
            max_delta = float((limit - effective_bias)[below].max())
        return torch.where(below, limit, effective_bias)

    transformer.apply_effective_bias_transform(perceptron, clip)
    # QAT drift re-clips dead channels by epsilons every projection; only a
    # MATERIAL clip (a real crater canonicalization) is worth a log line.
    if clipped and max_delta > 1e-3:
        print(
            f"[BiasSaturation] {getattr(perceptron, 'name', '<unnamed>')}: "
            f"clipped {clipped} constant-off channel bias(es) to the "
            f"saturation bound (max delta {max_delta:.3f})"
        )
    return clipped


def empirical_bias_shift(
    effective_bias: torch.Tensor,
    v_min: torch.Tensor,
    v_max: torch.Tensor,
    *,
    ceiling: float,
    margin: float,
) -> torch.Tensor:
    """Per-channel bias shift toward zero for EMPIRICALLY saturated channels.

    A channel whose observed pre-activation never leaves saturation
    (``v_min >= ceiling`` or ``v_max <= 0``) tolerates a uniform shift that
    keeps ``margin`` of saturation slack. The shift only ever shrinks ``|b|``
    and never crosses zero; unsaturated channels get exactly 0.
    """
    zero = torch.zeros_like(effective_bias)
    on = v_min >= ceiling
    off = v_max <= 0.0
    delta = torch.where(
        on, (ceiling - v_min) + margin,
        torch.where(off, (-v_max) - margin, zero),
    )
    shifted = effective_bias + delta
    shifted = torch.where(
        effective_bias > 0, shifted.clamp(min=0.0), shifted.clamp(max=0.0),
    )
    shrinks = shifted.abs() < effective_bias.abs()
    return torch.where((on | off) & shrinks, shifted - effective_bias, zero)


def predicted_weight_zero_fraction(perceptron, bits: int) -> float:
    """Fraction of effective weights that round to zero on the shared grid."""
    transformer = PerceptronTransformer()
    w = transformer.get_effective_weight(perceptron).detach().abs()
    b = transformer.get_effective_bias(perceptron).detach().abs()
    q_max = (2 ** (bits - 1)) - 1
    p_max = max(float(w.max()), float(b.max()), 1e-12)
    step = p_max / q_max
    return float((w < step / 2).float().mean())


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


def canonicalize_starved_bias_outliers(
    model,
    calibration_batches,
    *,
    bits: int,
    ceiling: float = 1.0,
    margin: float = _EMPIRICAL_MARGIN,
) -> dict:
    """Guarded, VERIFIED canonicalization of grid-starving bias outliers the
    provable clip cannot reach (the t01_16/rep3 class: empirically-constant
    channels whose |b_eff| still sets the NAPQ scale).

    Arms only perceptrons whose predicted weight-zero fraction at the target
    grid is >= 0.8 after the provable OFF-clip. Three guarded rungs, each
    VERIFIED (any decision flip on the calibration batches restores the state
    exactly): (2) shrink empirically-saturated channels to their observed
    slack; (3) for a still-starved grid, remove grid-dominating nuisance
    channels whose removal is decision-invariant. Returns
    ``{"clipped", "restored", "removed", "removal_restored"}`` counts.
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
            snapshot = (
                perceptron.layer.bias.data.clone(),
                {k: v.clone() for k, v in perceptron.normalization.state_dict().items()},
            )

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

            transformer.apply_effective_bias_transform(perceptron, shift)
            if moved == 0:
                continue
            verified = _decisions(model, batches, device)
            if all(torch.equal(a, b) for a, b in zip(reference, verified)):
                report["clipped"] += moved
                print(
                    f"[BiasSaturation] {getattr(perceptron, 'name', '<unnamed>')}: "
                    f"guarded-canonicalized {moved} empirically-saturated bias "
                    "outlier(s) (decision agreement verified)"
                )
            else:
                perceptron.layer.bias.data.copy_(snapshot[0])
                perceptron.normalization.load_state_dict(snapshot[1])
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
        snapshot = (
            perceptron.layer.weight.data.clone(),
            perceptron.layer.bias.data.clone(),
            {k: v.clone() for k, v in perceptron.normalization.state_dict().items()},
        )

        def zero_channel(t, c=c):
            out = t.clone()
            out[c] = 0.0
            return out

        transformer.apply_effective_weight_transform(perceptron, zero_channel)
        transformer.apply_effective_bias_transform(perceptron, zero_channel)
        verified = _decisions(model, batches, device)
        if all(torch.equal(a, v) for a, v in zip(reference, verified)):
            report["removed"] += 1
            print(
                f"[BiasSaturation] {getattr(perceptron, 'name', '<unnamed>')}: "
                f"removed grid-dominating nuisance channel {c} "
                "(decision agreement verified)"
            )
        else:
            perceptron.layer.weight.data.copy_(snapshot[0])
            perceptron.layer.bias.data.copy_(snapshot[1])
            perceptron.normalization.load_state_dict(snapshot[2])
            report["removal_restored"] += 1
            print(
                f"[BiasSaturation] {getattr(perceptron, 'name', '<unnamed>')}: "
                f"nuisance-channel removal RESTORED for channel {c} "
                "(decision flip on calibration data)"
            )
            return
