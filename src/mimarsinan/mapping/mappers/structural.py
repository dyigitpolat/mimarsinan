"""Structural mappers: Input, Reshape, EinopsRearrange, Stack, Concat, Subscript, Permute."""

from __future__ import annotations

import numpy as np
import torch

import einops

from mimarsinan.mapping.ir import IRSource

from mimarsinan.mapping.mappers.base import Mapper
from mimarsinan.mapping.mappers.flowchart import FlowchartNodeEstimate
from mimarsinan.mapping.mappers.scale_propagation import (
    mean_source_scale,
    present_source_scales,
)
from mimarsinan.mapping.support.scale_broadcast import concat_source_scales


def _create_ir_input_source(idx: int):
    return IRSource(node_id=-2, index=idx)


class InputMapper(Mapper):
    def __init__(self, input_shape):
        super(InputMapper, self).__init__()
        if len(input_shape) == 1:
            input_shape = (1, input_shape[0])
        if isinstance(input_shape, int):
            input_shape = (1, input_shape)
        self.input_shape = input_shape

    def _map_to_ir(self, ir_mapping):
        input_length = 1
        for dim in self.input_shape:
            input_length *= dim
        input_sources = []
        for input_idx in range(input_length):
            input_sources.append(_create_ir_input_source(input_idx))
        return np.array(input_sources).reshape(self.input_shape)

    def _forward_impl(self, x):
        return x

    def propagate_source_scale(self, deps, out_scales):
        # Parameterless root: nothing here knows the model's device, so the unit
        # wire scale is emitted on CPU by definition. Consumers anchor it --
        # ``assign_per_input_scales`` stamps it on the perceptron's device, and
        # ``align_scale_devices`` lifts it before any ComputeOp join mixes it.
        return torch.ones(self.input_shape[0])

    def propagate_boundary_scale(self, deps, out_scales, default):
        return float(default)


class ReshapeMapper(Mapper):
    moves_elements_only = True
    def __init__(self, source_mapper, output_shape):
        super(ReshapeMapper, self).__init__(source_mapper)
        self.output_shape = output_shape

    def _map_to_ir(self, ir_mapping):
        return self.require_source_mapper().map_to_ir(ir_mapping).reshape(self.output_shape)

    def _forward_impl(self, x):
        # reshape, not view: this is a LOGICAL reshape and must not also demand
        # a stride layout. A channels-last conv activation (cuDNN's choice,
        # preserved end to end once Normalization Fusion makes the perceptron's
        # normalization an Identity) has the same logical order and crashes
        # ``view`` outright.
        return x.reshape(x.shape[0], *self.output_shape)


class EinopsRearrangeMapper(Mapper):
    moves_elements_only = True
    def __init__(self, source_mapper, einops_str, *einops_args, **einops_kwargs):
        super(EinopsRearrangeMapper, self).__init__(source_mapper)
        self.einops_str = einops_str
        self.einops_args = einops_args
        self.einops_kwargs = einops_kwargs

    def _map_to_ir(self, ir_mapping):
        layer_sources = self.require_source_mapper().map_to_ir(ir_mapping)
        return einops.rearrange(
            layer_sources, self.einops_str, *self.einops_args, **self.einops_kwargs
        )

    def _forward_impl(self, x):
        return einops.rearrange(
            x, self.einops_str, *self.einops_args, **self.einops_kwargs
        )


class StackMapper(Mapper):
    moves_elements_only = True
    def __init__(self, source_mappers):
        super(StackMapper, self).__init__()
        self._source_mappers_list = list(source_mappers)

    @property
    def source_mappers(self):
        return self._source_mappers_list

    def get_source_mappers(self):
        return [m for m in self.source_mappers if m is not None]

    def _map_to_ir(self, ir_mapping):
        layer_sources_list = [mapper.map_to_ir(ir_mapping) for mapper in self.source_mappers]
        return np.stack(layer_sources_list).squeeze()

    def _forward_impl(self, x):
        outputs = list(x)
        return torch.stack(outputs, dim=1).squeeze(1)

    def flowchart_node_estimate(self, out_shape):
        return FlowchartNodeEstimate(sw_text="SW stack (host-side)")


class ConcatMapper(Mapper):
    """``torch.cat`` of activations along ``dim`` of a ``rank``-dimensional batched tensor.

    An IR source array holds one sample, its trailing axes the sample's axes (a vector of F features is (1, F)
    or (F,)), so the IR concatenates along ``dim`` counted from the end, after giving every operand the same rank
    with leading unit axes (which keeps each operand's source order).
    """
    moves_elements_only = True

    # Pickles from before ``rank`` concatenate along IR axis 0 (the leading per-sample axis) for every ``dim``.
    rank: int | None = None

    def __init__(self, source_mappers, dim: int = 1, name: str = "Concat", rank: int | None = None):
        super().__init__()
        self._source_mappers_list = list(source_mappers)
        self.dim = dim
        self._name = name
        if rank is not None:
            self.rank = int(rank)

    def get_source_mappers(self):
        return [m for m in self._source_mappers_list if m is not None]

    def ir_axis(self) -> int:
        """The IR axis of ``dim``: counted from the end, or 0 for a pickle without ``rank`` (whatever ``dim``)."""
        if self.rank is None:
            return 0
        return self.dim if self.dim < 0 else self.dim - self.rank

    def _map_to_ir(self, ir_mapping):
        layer_sources_list = [mapper.map_to_ir(ir_mapping) for mapper in self.get_source_mappers()]
        axis = self.ir_axis()
        if axis >= 0:
            return np.concatenate(layer_sources_list, axis=axis)
        rank = max(arr.ndim for arr in layer_sources_list)
        aligned = [arr.reshape((1,) * (rank - arr.ndim) + arr.shape) for arr in layer_sources_list]
        return np.concatenate(aligned, axis=axis)

    def _forward_impl(self, x):
        return torch.cat(tuple(x), dim=self.dim)

    def propagate_source_scale(self, deps, out_scales):
        # A JOIN: a CPU root scale can meet a device theta here, and ``torch.cat``
        # raises on mixed devices. Anchored in ``concat_source_scales``.
        parts = present_source_scales(deps, out_scales)
        return concat_source_scales(parts) if parts else None

    def propagate_boundary_scale(self, deps, out_scales, default):
        return mean_source_scale(deps, out_scales, float(default))


class SubscriptMapper(Mapper):
    moves_elements_only = True
    def __init__(self, source_mapper, index):
        super(SubscriptMapper, self).__init__(source_mapper)
        self.index = index

    def _map_to_ir(self, ir_mapping):
        return self.require_source_mapper().map_to_ir(ir_mapping)[self.index]

    def _forward_impl(self, x):
        return x.select(1, self.index)


class PermuteMapper(Mapper):
    """tensor.permute/transpose; dims includes batch axis 0. IR mapping uses batch-stripped np.transpose to reorder IRSource arrays for hardware layout."""
    moves_elements_only = True

    def __init__(self, source_mapper, dims):
        super(PermuteMapper, self).__init__(source_mapper)
        self.dims = tuple(dims)
        self._np_dims = tuple(d - 1 for d in self.dims if d != 0)

    def _map_to_ir(self, ir_mapping):
        arr = self.require_source_mapper().map_to_ir(ir_mapping)
        return np.transpose(arr, self._np_dims)

    def _forward_impl(self, x):
        return x.permute(*self.dims)


