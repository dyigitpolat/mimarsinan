"""On-chip validity gate for a mapped IR graph + shared forward-MAC SSOT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import IRGraph

# Validity threshold SSOT: a mapping whose on-chip params OR ops fraction is below
# DEFAULT_ONCHIP_FLOOR is INVALID (host-majority — an erroneous deployment); the
# MAJORITY threshold separates VALID (>= majority on both) from VALID_FLAGGED.
# Owned here so the pipeline gate, the registry defaults, and the research
# campaign scheduler read ONE source instead of re-hardcoding the literals.
DEFAULT_ONCHIP_FLOOR = 0.20
DEFAULT_ONCHIP_MAJORITY = 0.50


class OnchipMajorityError(ValueError):
    """A deployment maps the parameter MINORITY to chip cores (host-majority)."""


@dataclass(frozen=True)
class OnchipParamBreakdown:
    """Logical host/on-chip split of a deployed model's parameters."""

    onchip_params: int
    host_params: int
    total_params: int

    @property
    def fraction(self) -> float:
        if self.total_params <= 0:
            return 0.0
        return self.onchip_params / self.total_params


@dataclass(frozen=True)
class OnchipOpsBreakdown:
    """Logical host/on-chip split of a deployed model's forward MACs."""

    onchip_ops: int
    host_ops: int
    total_ops: int

    @property
    def fraction(self) -> float:
        if self.total_ops <= 0:
            return 0.0
        return self.onchip_ops / self.total_ops


def _numel(value) -> int:
    if isinstance(value, np.ndarray):
        return int(value.size)
    if torch.is_tensor(value):
        return int(value.numel())
    return 0


def is_scale_wrapper(module) -> bool:
    return type(module).__name__ == "ScaleNormalizingWrapper" and hasattr(
        module, "module"
    )


def unwrap_scale_wrapper(module):
    """Peel a ``ScaleNormalizingWrapper`` to reach the wrapped op module."""
    while is_scale_wrapper(module):
        module = module.module
    return module


def _linear_macs(in_features: int, out_features: int, n_positions: int) -> int:
    return int(in_features) * int(out_features) * int(n_positions)


def _num_positions(in_shape, in_features) -> int:
    if in_shape is None:
        return 1
    numel = 1
    for d in in_shape:
        numel *= int(d)
    if in_features and in_features > 0 and numel % in_features == 0:
        return max(1, numel // int(in_features))
    return 1


def _conv_macs(module, out_shape) -> int:
    if out_shape is None:
        return 0
    out_spatial = 1
    for d in out_shape[2:]:
        out_spatial *= int(d)
    k = 1
    for d in module.kernel_size:
        k *= int(d)
    in_per_group = int(module.in_channels) // int(module.groups)
    return int(module.out_channels) * out_spatial * in_per_group * k


def _attention_macs(module: nn.MultiheadAttention, in_shape) -> int:
    """QKV + output projections (3·E·E + E·E per position) plus the
    scores·values double matmul (2·L·E per position) for self-attention."""
    if in_shape is None:
        return 0
    e = int(module.embed_dim)
    numel = 1
    for d in in_shape:
        numel *= int(d)
    seq = max(1, numel // e) if e > 0 else 1
    proj_macs = 4 * e * e * seq
    score_macs = 2 * seq * seq * e
    return int(proj_macs + score_macs)


def module_macs(module: nn.Module, in_shape, out_shape) -> int:
    """Estimate forward MACs for a host/on-chip unit from its full I/O tensor shapes.

    Covers Linear, Conv1d/2d, MultiheadAttention and a Perceptron's linear layer;
    pure element-wise/shape ops carry no MACs. The shared MAC SSOT for both the
    IR-graph host-op count and the static forward estimator.
    """
    if is_scale_wrapper(module):
        return module_macs(cast(nn.Module, module.module), in_shape, out_shape)

    layer = getattr(module, "layer", None)
    if isinstance(layer, nn.Linear):
        n_positions = _num_positions(in_shape, layer.in_features)
        return _linear_macs(layer.in_features, layer.out_features, n_positions)

    if isinstance(module, nn.Linear):
        n_positions = _num_positions(in_shape, module.in_features)
        return _linear_macs(module.in_features, module.out_features, n_positions)

    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        return _conv_macs(module, out_shape)

    if isinstance(module, nn.MultiheadAttention):
        return _attention_macs(module, in_shape)

    return 0


def count_host_params(ir_graph: IRGraph) -> int:
    """Sum of unique host-side ComputeOp parameters (modules deduped by identity).

    A wrapped module's ``parameters()`` already includes bound constant tensors,
    so bare ``bound_tensors`` are only added when the op carries no module.
    """
    seen_modules: set[int] = set()
    total = 0
    for op in ir_graph.get_compute_ops():
        module = op.params.get("module")
        if module is not None and hasattr(module, "parameters"):
            if id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            total += sum(_numel(p) for p in module.parameters())
            continue
        for tensor in op.params.get("bound_tensors") or []:
            total += _numel(tensor)
    return total


def count_host_ops(ir_graph: IRGraph) -> int:
    """Sum of forward MACs of unique host-side ComputeOps (deduped by identity).

    The ops sibling of :func:`count_host_params`. A subsumed encoding perceptron
    is counted at its wrapped perceptron with a single position (matching the
    static estimator, whose forward hook does not fire on subsumed encoders); a
    plain compute module (readout Linear, pooling) is counted from the
    ComputeOp's declared per-invocation shapes. Reproduces
    ``estimate_onchip_fraction(metric='macs')``'s host decomposition for
    feed-forward host ops; multi-input/attention research-gap ops (whose gather
    triples the sequence) fall outside this structural count and use the
    forward-based estimator directly.
    """
    seen: set[int] = set()
    total = 0
    for op in ir_graph.get_compute_ops():
        module = op.params.get("module")
        if module is None:
            continue
        perceptron = getattr(module, "perceptron", None)
        unit = perceptron if perceptron is not None else module
        if id(unit) in seen:
            continue
        seen.add(id(unit))
        if perceptron is not None:
            in_shape = out_shape = None
        else:
            in_shape = (1, *op.input_shape) if op.input_shape is not None else None
            out_shape = (1, *op.output_shape) if op.output_shape is not None else None
        total += module_macs(unit, in_shape, out_shape)
    return total


def compute_onchip_ops_fraction(
    ir_graph: IRGraph, *, total_ops: int
) -> OnchipOpsBreakdown:
    """Logical on-chip forward-MAC fraction = (total - host) / total.

    On-chip ops are the remainder after the host ComputeOps, never the raw
    crossbar op count: conv weight-bank reuse replicates one logical matmul
    across many cores, so summing per-core MACs over-counts (as tiling does for
    params). ``total_ops`` is the model's forward-MAC total (a model fact, like
    ``sum(model.parameters())`` is for params).
    """
    host_ops = count_host_ops(ir_graph)
    onchip_ops = int(total_ops) - host_ops
    return OnchipOpsBreakdown(
        onchip_ops=onchip_ops,
        host_ops=host_ops,
        total_ops=int(total_ops),
    )


def compute_onchip_fraction(
    ir_graph: IRGraph, *, total_params: int
) -> OnchipParamBreakdown:
    """Logical on-chip fraction = (total - host) / total.

    On-chip is the logical unique-parameter remainder, not the raw crossbar
    footprint, since tiling replicates weights across many cores.
    """
    host_params = count_host_params(ir_graph)
    onchip_params = int(total_params) - host_params
    return OnchipParamBreakdown(
        onchip_params=onchip_params,
        host_params=host_params,
        total_params=int(total_params),
    )


def assert_onchip_majority_or_raise(
    ir_graph: IRGraph, *, total_params: int, min_fraction: float = 0.2
) -> OnchipParamBreakdown:
    """Raise :class:`OnchipMajorityError` when the on-chip fraction is below the floor.

    Raises only below the floor; between floor and majority a mapping is
    VALID_FLAGGED (see :func:`onchip_fraction.classify_validity`).
    """
    breakdown = compute_onchip_fraction(ir_graph, total_params=total_params)
    if breakdown.fraction < min_fraction:
        raise OnchipMajorityError(
            "On-chip parameter majority violated: only "
            f"{breakdown.fraction:.2%} of the {breakdown.total_params} deployed "
            f"parameters are placed on chip cores "
            f"(on-chip={breakdown.onchip_params}, host={breakdown.host_params}), "
            f"below the required {min_fraction:.0%} floor. The host-side "
            "ComputeOps (offloaded encoding Linear/Conv, classifier readout, "
            "attention) hold the parameter majority, so this mapping is not a "
            "genuine on-chip deployment."
        )
    return breakdown
