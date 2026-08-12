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
    "clear_wire_value_ops",
    "heterogeneous_domain_joins",
    "mark_wire_value_ops",
    "node_absorbs_negative_values",
    "op_preserves_wire_ratio",
    "produces_nonnegative_values",
    "value_domain_map",
]


def value_domain_map(exec_order, deps) -> tuple[dict, list]:
    """``(node -> stored value is ABSOLUTE, plain host ops with a MIXED fan-in)``.

    Neural producers and ARMED ComputeOps emit the wire domain; a structural
    chain rooted at the input stays absolute; a plain host op inherits its
    sources'. An op whose sources disagree carries two currencies at once and
    has NO domain until its gauge is established (calculus §11.2) — it is
    reported, and the walk continues as absolute so every such join is listed.
    """
    flags: dict = {}
    joins: list = []
    for node in exec_order:
        node_deps = deps.get(node, [])
        if getattr(node, "perceptron", None) is not None:
            flags[node] = False
        elif isinstance(node, ComputeOpMapper) and node.output_scale is not None:
            flags[node] = False
        elif not node_deps:
            flags[node] = True
        else:
            dep_flags = {flags[dep] for dep in node_deps}
            if len(dep_flags) > 1:
                joins.append(node)
                flags[node] = True
            else:
                flags[node] = dep_flags.pop()
    return flags, joins


def heterogeneous_domain_joins(model_repr) -> list:
    """Plain host ops whose fan-in mixes wire and absolute sources — the graph
    is unclassifiable (and so unexecutable by a wire-currency twin) until every
    such source decodes at its own producer gauge."""
    model_repr._ensure_exec_graph()
    return value_domain_map(model_repr._exec_order, model_repr._deps)[1]


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


def _perceptron_boundaries(node, consumers) -> list:
    """First perceptron-bearing nodes reachable downstream of ``node``."""
    found, frontier, seen = [], list(consumers.get(id(node), [])), set()
    while frontier:
        candidate = frontier.pop()
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        perceptron = getattr(candidate, "perceptron", None)
        if perceptron is not None:
            found.append(perceptron)
        else:
            frontier.extend(consumers.get(id(candidate), []))
    return found


def mark_wire_value_ops(model_repr) -> int:
    """Stamp ``is_wire_value_op`` on every host ComputeOp whose module is NOT
    positively homogeneous — the rate-path analog of the TTFS ``apply_ttfs``
    per-op value transcode.

    An armed value op owns its domain at emission (ScaleNormalizingWrapper):
    inputs lift to values, the output normalizes to the fold currency — the
    boundary-algebra I3/I1 fix. Terminal/host-consumed value ops arm too (the
    trained plain forward feeds them VALUES; a rate-fed terminal head skews
    the bias term and craters argmax parity). Only ops feeding a host-side
    encoder stay unarmed (encoders read raw values). Idempotent; returns the
    number of marked ops.
    """
    consumers = model_repr.consumer_map()
    marked = 0
    for node in model_repr.execution_order():
        if not isinstance(node, ComputeOpMapper):
            continue
        node.is_wire_value_op = False
        if op_preserves_wire_ratio(getattr(node, "module", None)):
            continue
        boundaries = _perceptron_boundaries(node, consumers)
        if all(
            not getattr(p, "is_encoding_layer", False) for p in boundaries
        ):
            node.is_wire_value_op = True
            marked += 1
    return marked


def clear_wire_value_ops(model_repr) -> None:
    """Un-mark every host ComputeOp (the TTFS-family wires own their transcode)."""
    for node in model_repr.execution_order():
        if isinstance(node, ComputeOpMapper):
            node.is_wire_value_op = False


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
