"""LIF activation helpers (no hybrid / perceptron imports)."""

from __future__ import annotations

from mimarsinan.models.activations import LIFActivation


def unwrap_lif_activation(activation) -> LIFActivation | None:
    """Walk activation wrappers and return inner ``LIFActivation``, or None."""
    from mimarsinan.models.layers import TransformedActivation
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFBlendActivation

    for _ in range(8):
        if activation is None:
            return None
        if isinstance(activation, LIFActivation):
            return activation
        if isinstance(activation, TransformedActivation):
            activation = getattr(activation, "base_activation", None)
            continue
        if isinstance(activation, LIFBlendActivation):
            activation = activation.lif_activation
            continue
        return None
    return None


def apply_cycle_accurate_trains_to_model(model, enabled: bool) -> None:
    """Set ``use_cycle_accurate_trains`` on every reachable ``LIFActivation``."""
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFBlendActivation

    flag = bool(enabled)
    for module in model.modules():
        if isinstance(module, LIFActivation):
            module.use_cycle_accurate_trains = flag
        elif isinstance(module, LIFBlendActivation):
            module.lif_activation.use_cycle_accurate_trains = flag
