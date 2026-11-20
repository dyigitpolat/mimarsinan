"""LIF activation helpers (no hybrid imports)."""

from __future__ import annotations

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.layers import TransformedActivation
from mimarsinan.models.perceptron_mixer.membrane_lattice import arm_membrane_lattices


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
    """Arm each perceptron's LIF membrane snap with the lattice its persisted parameters describe.

    Idempotent and re-runnable: tuning stages recreate activation objects and restore states, so the arming DERIVES the
    lattice at every seam (executor build, parity gate, published read) from the perceptron's recorded lattice kind and
    its PERSISTED ``parameter_scale`` (``models.perceptron_mixer.membrane_lattice``), and caches nothing. A true
    threshold tie must be decided by the exact value on every twin -- float noise fired a strict-'<' tie in the NF
    (t0_45, 2026-08-09) while the exact SCM correctly held it. [U-POST G4] A perceptron the projection never reached
    stays unarmed whatever its ``parameter_scale`` (the default 1.0 included): its membrane does not live on a lattice,
    and rounding it each cycle is a value change, not a tie decision. Encoding layers integrate the input-quantized
    pre-activation, whose grid divides by T as well (1/(ps*T)): snapping those on 2*ps rounds true HALF-lattice
    membranes (167.5/164) on float dust (n7 t8 sample-0 catch, 2026-08-10); that holds only where the encoding layer's
    input carries the 1/T entry grid (a host-side encoder reading a float input integrates a float charge)."""
    return arm_membrane_lattices(model)
