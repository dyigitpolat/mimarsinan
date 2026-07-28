"""Selectable per-layer activation-scale calibration policies (ANN->SNN)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from mimarsinan.mapping.support.tensor_stats import safe_quantile

_CANDIDATES = 160
_MIN_THETA = 1e-3
_FALLBACK_THETA = 1.0


def deployed_distortion(
    activations: torch.Tensor, theta: float, levels: int, *, weight: float = 1.0,
) -> float:
    """``D(θ) = E[(a−θ)²·1(a>θ)] + (θ/L)²/12·P(a≤θ)``, scaled by ``weight``.

    The two terms are the deployed staircase's only error sources: saturation
    above θ, and the θ/L grid below it. ``weight`` is the unit's sensitivity
    (Fisher diagonal); it rescales D uniformly, so it orders units without
    moving any single unit's argmin.
    """
    a = activations.float()
    if a.numel() == 0:
        return 0.0
    over = a[a > theta]
    clipping = float(((over - theta) ** 2).sum()) / a.numel()
    fraction_in = float((a <= theta).float().mean())
    resolution = (theta / max(int(levels), 1)) ** 2 / 12.0 * fraction_in
    return float(weight) * (clipping + resolution)


def optimal_theta(
    activations: torch.Tensor,
    levels: int,
    *,
    weight: float = 1.0,
    candidates: int = _CANDIDATES,
    min_theta: float = _MIN_THETA,
) -> float:
    """θ* minimizing :func:`deployed_distortion` over a candidate grid.

    Deterministic and assumption-free: the grid spans the positive activations'
    median to max, so no distribution family is imposed (the measured shapes
    are heavy-tailed and vary by an order of magnitude across depth). Dead or
    fully pruned units fall back to ``1.0`` — a normal case, not an error.
    """
    a = activations.float()
    positive = a[a > 1e-6]
    if positive.numel() == 0:
        return _FALLBACK_THETA
    low = float(safe_quantile(positive, 0.5))
    high = float(positive.max())
    if not (high > low):
        return max(high, min_theta)
    best_theta, best_distortion = low, None
    for theta in torch.linspace(low, high, int(candidates)).tolist():
        distortion = deployed_distortion(positive, theta, levels, weight=weight)
        if best_distortion is None or distortion < best_distortion:
            best_theta, best_distortion = theta, distortion
    return max(best_theta, min_theta)


PRUNED_THRESHOLD = 1e-9
MIN_SCALE = 1e-6
DEFAULT_SCALE_QUANTILE = 0.99

DEFAULT_ACTIVATION_SCALE_POLICY = "count_quantile"


def _as_float32(flat_acts) -> torch.Tensor:
    return flat_acts.detach().reshape(-1).to(torch.float32)


class ActivationScalePolicy(ABC):
    """Maps a flat activation tensor to a single positive normalization scale."""

    @abstractmethod
    def scale(self, flat_acts) -> float:
        """Return the per-layer activation scale for ``flat_acts``."""
        raise NotImplementedError


class CountQuantilePolicy(ActivationScalePolicy):
    """Framework DEFAULT: count quantile over positive (non-pruned) activations.

    Byte-identical to the legacy ``scale_from_activations`` path.
    """

    def __init__(
        self,
        *,
        quantile: float = DEFAULT_SCALE_QUANTILE,
        pruned_threshold: float = PRUNED_THRESHOLD,
        min_scale: float = MIN_SCALE,
    ):
        self.quantile = float(quantile)
        self.pruned_threshold = float(pruned_threshold)
        self.min_scale = float(min_scale)

    def scale(self, flat_acts) -> float:
        active_mask = flat_acts > self.pruned_threshold
        active_acts = flat_acts[active_mask]

        if active_acts.numel() == 0:
            return (
                max(flat_acts.max().item(), 1.0) if flat_acts.numel() > 0 else 1.0
            )

        q = safe_quantile(
            active_acts.to(torch.float32),
            float(self.quantile),
            interpolation="higher",
        ).item()
        return max(float(q), float(self.min_scale))


class PercentileNormPolicy(ActivationScalePolicy):
    """Rueckauer et al. (2017) robust-norm: p-th percentile of the whole
    activation distribution (``percentile=100`` recovers classic max-norm)."""

    def __init__(
        self,
        *,
        percentile: float = 99.9,
        min_scale: float = MIN_SCALE,
    ):
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(
                f"percentile must be in [0, 100], got {percentile}"
            )
        self.percentile = float(percentile)
        self.min_scale = float(min_scale)

    def scale(self, flat_acts) -> float:
        acts = _as_float32(flat_acts)
        if acts.numel() == 0:
            return 1.0
        q = safe_quantile(
            acts,
            self.percentile / 100.0,
            interpolation="higher",
        ).item()
        return max(float(q), float(self.min_scale))


class MaxNormPolicy(ActivationScalePolicy):
    """Textbook max-norm baseline: scale == max activation (percentile_norm@100)."""

    def __init__(self, *, min_scale: float = MIN_SCALE):
        self.min_scale = float(min_scale)

    def scale(self, flat_acts) -> float:
        acts = _as_float32(flat_acts)
        if acts.numel() == 0:
            return 1.0
        return max(float(acts.max().item()), float(self.min_scale))


class MinDistortionPolicy(ActivationScalePolicy):
    """[calculus §17.13] theta as the argmin of the DEPLOYED distortion
    ``clipping + resolution`` on an ``levels``-step grid — the two error
    sources the staircase actually has, instead of a fixed quantile.

    ``levels`` is required: the optimum depends on the grid the deployment
    will use (T for rate codes), and guessing it silently would reintroduce
    the heuristic this policy replaces."""

    def __init__(self, *, levels: int, min_scale: float = MIN_SCALE):
        self.levels = int(levels)
        self.min_scale = float(min_scale)

    def scale(self, flat_acts) -> float:
        return max(
            float(optimal_theta(_as_float32(flat_acts), self.levels)),
            float(self.min_scale),
        )


_POLICY_FACTORIES = {
    "count_quantile": CountQuantilePolicy,
    "percentile_norm": PercentileNormPolicy,
    "max_norm": MaxNormPolicy,
    "min_distortion": MinDistortionPolicy,
}


def make_activation_scale_policy(
    name: str = DEFAULT_ACTIVATION_SCALE_POLICY, **kwargs
) -> ActivationScalePolicy:
    """Construct a named activation-scale policy; unknown names raise ``ValueError``."""
    try:
        factory = _POLICY_FACTORIES[name]
    except KeyError:
        valid = ", ".join(sorted(_POLICY_FACTORIES))
        raise ValueError(
            f"unknown activation-scale policy {name!r}; valid: {valid}"
        )
    return factory(**kwargs)


def scale_from_activations(
    flat_acts,
    pruned_threshold=PRUNED_THRESHOLD,
    *,
    quantile=DEFAULT_SCALE_QUANTILE,
    min_scale=MIN_SCALE,
):
    """Count-based activation quantile over non-pruned positives, so post-pruning stats stay unskewed."""
    active_mask = flat_acts > pruned_threshold
    active_acts = flat_acts[active_mask]

    if active_acts.numel() == 0:
        return max(flat_acts.max().item(), 1.0) if flat_acts.numel() > 0 else 1.0

    q = safe_quantile(
        active_acts.to(torch.float32),
        float(quantile),
        interpolation="higher",
    ).item()
    return max(float(q), float(min_scale))


def resolve_scale_for_samples(flat_acts, *, policy, quantile, levels):
    """[§17.13] one seam for the scale rule: the legacy count quantile, or the
    deployed-distortion argmin at the grid the deployment will actually use."""
    if policy == "count_quantile":
        return scale_from_activations(
            flat_acts, quantile=quantile, min_scale=MIN_SCALE,
        )
    if policy == "min_distortion":
        if not levels:
            raise ValueError(
                "activation_scale_policy='min_distortion' needs a value grid, "
                "but this mode deploys continuously (value_grid_levels=None): "
                "the distortion argmin is undefined without a resolution term"
            )
        return make_activation_scale_policy(
            "min_distortion", levels=int(levels), min_scale=MIN_SCALE,
        ).scale(flat_acts)
    return make_activation_scale_policy(policy).scale(flat_acts)
