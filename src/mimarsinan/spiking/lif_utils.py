"""LIF activation helpers (no hybrid / perceptron imports)."""

from __future__ import annotations

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.layers import TransformedActivation


def unwrap_lif_activation(activation) -> LIFActivation | None:
    """Walk activation wrappers and return inner ``LIFActivation``, or None."""
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


def arm_integer_membrane_lattice(model) -> int:
    """Arm each perceptron's LIF membrane snap with its integral NAPQ grid.

    Idempotent and re-runnable: tuning stages recreate activation objects, so
    the stamp must be derivable from the PERSISTED ``parameter_scale`` at any
    seam (executor build, parity gate). A true threshold tie must be decided
    by the exact value on every twin — float noise fired a strict-'<' tie in
    the NF (t0_45, 2026-08-09) while the exact SCM correctly held it."""
    armed = 0
    for perceptron in model.get_perceptrons():
        lif = unwrap_lif_activation(getattr(perceptron, "activation", None))
        if lif is None:
            continue
        ps = getattr(perceptron, "parameter_scale", None)
        if ps is None:
            continue
        value = float(ps.item() if hasattr(ps, "item") else ps)
        if value > 0 and abs(value - round(value)) <= 1e-6:
            lif.set_membrane_lattice(round(value))
            armed += 1
    return armed
