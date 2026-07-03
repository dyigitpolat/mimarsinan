"""Selectable per-layer activation-scale calibration policies (ANN->SNN)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

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

        q = torch.quantile(
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
        q = torch.quantile(
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


_POLICY_FACTORIES = {
    "count_quantile": CountQuantilePolicy,
    "percentile_norm": PercentileNormPolicy,
    "max_norm": MaxNormPolicy,
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
