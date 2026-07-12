"""Channel-axis-aligned consumer discovery over the mapper DAG (the shared walk SSOT)."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.leading_dim import (
    Ensure2DMapper, MergeLeadingDimsMapper, SplitLeadingDimMapper,
)
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import PermuteMapper
from mimarsinan.mapping.support.compute_modules import ComputeAdapter

# Structural mappers that carry a channels-last axis through unchanged.
_LAST_AXIS_PASS_THROUGH = (Ensure2DMapper, MergeLeadingDimsMapper, SplitLeadingDimMapper)


def consumer_columns_unmediated(perceptron) -> bool:
    """Nothing may sit between the consumer's input and its weight columns."""
    return (
        isinstance(perceptron.layer, nn.Linear)
        and isinstance(perceptron.input_activation, nn.Identity)
        and getattr(perceptron, "per_input_scales", None) is None
    )


def _permute_step(node: PermuteMapper, k: int):
    dims = tuple(node.dims)
    rank = len(dims)
    pos_in = rank - int(k)
    if pos_in <= 0 or pos_in >= rank or pos_in not in dims:
        return None
    pos_out = dims.index(pos_in)
    return None if pos_out == 0 else ("pass", rank - pos_out)


def _mean_reduced_dims(adapter: ComputeAdapter):
    fallback = adapter.extra_args[0] if len(adapter.extra_args) == 1 else None
    raw = adapter.kwargs.get("dim", fallback)
    if isinstance(raw, int):
        return (raw,)
    if isinstance(raw, (tuple, list)) and all(isinstance(d, int) for d in raw):
        return tuple(raw)
    return None


def _mean_passthrough(node: ComputeOpMapper, k: int):
    """New from-end channel position through a mean over non-channel axes, or None."""
    adapter = node.module
    if not isinstance(adapter, ComputeAdapter) or adapter.fn is not torch.mean:
        return None
    if getattr(adapter, "_bound_count", 0) != 0 or len(node.get_source_mappers()) != 1:
        return None
    dims = _mean_reduced_dims(adapter)
    shapes = node.input_shapes
    if dims is None or shapes is None or shapes[0] is None:
        return None
    rank = len(shapes[0]) + 1  # input_shapes are batch-stripped
    channel_pos = rank - int(k)
    reduced = {d % rank for d in dims}
    if 0 in reduced or channel_pos in reduced:
        return None
    if bool(adapter.kwargs.get("keepdim", False)):
        return int(k)
    return int(k) - sum(1 for d in reduced if d > channel_pos)


def _step_through(node, k: int):
    """One walk transition: ('pass', new_k) | ('perceptron', p) | ('module', m) | None."""
    if isinstance(node, PerceptronMapper):
        perceptron = node.perceptron
        if k == 1 and consumer_columns_unmediated(perceptron):
            return ("perceptron", perceptron)
        return None
    if isinstance(node, _LAST_AXIS_PASS_THROUGH):
        return ("pass", 1) if k == 1 else None
    if isinstance(node, PermuteMapper):
        return _permute_step(node, k)
    if isinstance(node, ComputeOpMapper):
        passed = _mean_passthrough(node, k)
        if passed is not None:
            return ("pass", passed)
        module = node.module
        if isinstance(module, nn.Linear) and k == 1 and len(node.get_source_mappers()) == 1:
            return ("module", module)
        return None
    return None


def channel_aligned_consumer_targets(producer_node, consumers: dict):
    """All consumers of the producer's channel axis as ``(perceptrons, modules)``,
    or None when any path is not exactly column-aligned (fan-out closure: one bad
    path voids the producer). ``k`` is the channel position counted from the
    tensor's end (1 = last)."""
    frontier: list = [(producer_node, 1)]
    visited: set = set()
    perceptron_targets: dict = {}
    module_targets: dict = {}
    while frontier:
        node, k = frontier.pop()
        downstream = consumers.get(id(node), [])
        if not downstream:
            return None  # the channel axis reaches the model output unmediated
        for consumer in downstream:
            key = (id(consumer), k)
            if key in visited:
                continue
            visited.add(key)
            step = _step_through(consumer, k)
            if step is None:
                return None
            kind, value = step
            if kind == "pass":
                frontier.append((consumer, value))
            elif kind == "perceptron":
                perceptron_targets[id(value)] = value
            else:
                module_targets[id(value)] = value
    return tuple(perceptron_targets.values()), tuple(module_targets.values())
