"""Structural value-domain guarantees: which nodes absorb a signed input range."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.models.nn.activations import LIFActivation, LeakyGradReLU
from mimarsinan.models.nn.activations.ttfs_cycle import TTFSCycleActivation
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation

__all__ = [
    "node_absorbs_negative_values",
    "op_preserves_wire_ratio",
    "produces_nonnegative_values",
]

# Activations whose output is >= 0 for EVERY input. The spiking activations
# decode spike counts / spike times, which are non-negative by construction.
NONNEGATIVE_ACTIVATIONS: tuple[type, ...] = (
    LeakyGradReLU,
    nn.ReLU,
    nn.ReLU6,
    LIFActivation,
    TTFSActivation,
    TTFSCycleActivation,
)


def _clamp_floor_is_nonnegative(decorator) -> bool:
    """A ClampDecorator with a floor >= 0 forces the range whatever the base does."""
    floor = getattr(decorator, "clamp_min", None)
    if floor is None:
        return False
    if isinstance(floor, torch.Tensor):
        return bool(floor.min() >= 0.0)
    return float(floor) >= 0.0


def produces_nonnegative_values(module) -> bool:
    """Whether ``module``'s output is non-negative for any input, structurally.

    A guarantee, not a calibration observation: only such a node can absorb a
    signed range at a segment boundary. Unknown modules answer ``False`` — the
    conservative direction (subsume further, never encode a signed value).
    """
    if module is None:
        return False

    base = getattr(module, "base_activation", None)
    decorators = getattr(module, "decorators", None)
    if base is not None and decorators is not None:
        if any(_clamp_floor_is_nonnegative(d) for d in decorators):
            return True
        return produces_nonnegative_values(base)

    activation = getattr(module, "activation", None)
    if activation is not None and activation is not module:
        return produces_nonnegative_values(activation)

    return isinstance(module, NONNEGATIVE_ACTIVATIONS)


# Positively homogeneous modules: f(a*x) = a*f(x) for a > 0, so wire rate and
# value produce the same output up to the gauge — no seam transcode needed.
WIRE_TRANSPARENT_MODULES: tuple[type, ...] = (
    nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
    nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
    nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
    nn.Identity, nn.Flatten,
)

# ComputeAdapter payload callables that are positively homogeneous.
_WIRE_TRANSPARENT_ADAPTER_FNS = frozenset({
    "mean", "sum", "amax", "amin", "flatten", "reshape", "permute",
    "transpose", "cat",
})


def op_preserves_wire_ratio(module) -> bool:
    """Whether a host op is positively homogeneous (``f(a*x) = a*f(x)``, a > 0).

    Membership means the op may legally run on the wire rate (the gauge passes
    through); anything unknown answers ``False`` — the conservative direction
    (treat as a value op, transcode at its boundaries).
    """
    if module is None:
        return False
    if isinstance(module, ComputeAdapter):
        fn_name = getattr(module.fn, "__name__", "")
        return fn_name in _WIRE_TRANSPARENT_ADAPTER_FNS
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return module.bias is None
    return isinstance(module, WIRE_TRANSPARENT_MODULES)


def node_absorbs_negative_values(node) -> bool:
    """Whether this mapper-graph node's OUTPUT is structurally non-negative.

    Perceptron nodes answer through their activation, host ComputeOps through
    their module; a structural node (reshape/permute/...) is sign-transparent
    and absorbs nothing.
    """
    perceptron = getattr(node, "perceptron", None)
    if perceptron is not None:
        return produces_nonnegative_values(getattr(perceptron, "activation", None))
    if isinstance(node, ComputeOpMapper):
        return produces_nonnegative_values(getattr(node, "module", None))
    return False
