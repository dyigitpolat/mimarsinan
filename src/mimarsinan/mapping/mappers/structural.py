"""Structural mappers: Input, Reshape, Delay, EinopsRearrange, Stack, Add, Concat, Subscript."""

from __future__ import annotations

import numpy as np
import torch

import einops

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.soft_core_mapper import map_mm

from mimarsinan.mapping.mappers.base import Mapper


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

    def _map(self, mapping):
        input_length = 1
        for dim in self.input_shape:
            input_length *= dim
        input_sources = []
        for input_idx in range(input_length):
            input_sources.append(SpikeSource(-2, input_idx, True, False))
        return np.array(input_sources).reshape(self.input_shape)

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


class ReshapeMapper(Mapper):
    def __init__(self, source_mapper, output_shape):
        super(ReshapeMapper, self).__init__(source_mapper)
        self.output_shape = output_shape

    def _map(self, mapping):
        return self.source_mapper.map(mapping).reshape(self.output_shape)

    def _map_to_ir(self, ir_mapping):
        return self.source_mapper.map_to_ir(ir_mapping).reshape(self.output_shape)

    def _forward_impl(self, x):
        return x.view(x.shape[0], *self.output_shape)


class DelayMapper(Mapper):
    def __init__(self, source_mapper, delay):
        super(DelayMapper, self).__init__(source_mapper)
        self.delay = delay

    def _map(self, mapping):
        layer_sources = self.source_mapper.map(mapping)
        for _ in range(self.delay):
            layer_sources = map_mm(
                mapping,
                layer_sources,
                np.eye(layer_sources.shape[-2]),
                parameter_scale=torch.tensor(mapping.q_max),
            )
        return layer_sources

    def _forward_impl(self, x):
        return x


class EinopsRearrangeMapper(Mapper):
    def __init__(self, source_mapper, einops_str, *einops_args, **einops_kwargs):
        super(EinopsRearrangeMapper, self).__init__(source_mapper)
        self.einops_str = einops_str
        self.einops_args = einops_args
        self.einops_kwargs = einops_kwargs

    def _map(self, mapping):
        layer_sources = self.source_mapper.map(mapping)
        return einops.rearrange(
            layer_sources, self.einops_str, *self.einops_args, **self.einops_kwargs
        )

    def _map_to_ir(self, ir_mapping):
        layer_sources = self.source_mapper.map_to_ir(ir_mapping)
        return einops.rearrange(
            layer_sources, self.einops_str, *self.einops_args, **self.einops_kwargs
        )

    def _forward_impl(self, x):
        return einops.rearrange(
            x, self.einops_str, *self.einops_args, **self.einops_kwargs
        )


class StackMapper(Mapper):
    def __init__(self, source_mappers):
        super(StackMapper, self).__init__()
        self._source_mappers_list = list(source_mappers)

    @property
    def source_mappers(self):
        return self._source_mappers_list

    def get_source_mappers(self):
        return [m for m in self.source_mappers if m is not None]

    def _map(self, mapping):
        layer_sources_list = [mapper.map(mapping) for mapper in self.source_mappers]
        return np.stack(layer_sources_list).squeeze()

    def _map_to_ir(self, ir_mapping):
        layer_sources_list = [mapper.map_to_ir(ir_mapping) for mapper in self.source_mappers]
        return np.stack(layer_sources_list).squeeze()

    def _forward_impl(self, x):
        outputs = list(x)
        return torch.stack(outputs, dim=1).squeeze(1)


class AddMapper(Mapper):
    def __init__(self, source_mapper_a, source_mapper_b):
        super(AddMapper, self).__init__()
        self._source_mapper_a_container = [source_mapper_a]
        self._source_mapper_b_container = [source_mapper_b]

    @property
    def source_mapper_a(self):
        return self._source_mapper_a_container[0]

    @property
    def source_mapper_b(self):
        return self._source_mapper_b_container[0]

    def get_source_mappers(self):
        return [m for m in [self.source_mapper_a, self.source_mapper_b] if m is not None]

    def _map(self, mapping):
        layer_sources_a = self.source_mapper_a.map(mapping)
        layer_sources_b = self.source_mapper_b.map(mapping)
        assert layer_sources_a.shape == layer_sources_b.shape
        x_rows = layer_sources_a.shape[-2]
        layer_sources = np.concatenate([layer_sources_a, layer_sources_b], axis=0)
        weights = np.concatenate([np.eye(x_rows), np.eye(x_rows)], axis=0).transpose()
        return map_mm(mapping, layer_sources, weights, parameter_scale=torch.tensor(mapping.q_max))

    def _map_to_ir(self, ir_mapping):
        a_sources = self.source_mapper_a.map_to_ir(ir_mapping)
        b_sources = self.source_mapper_b.map_to_ir(ir_mapping)
        assert a_sources.shape == b_sources.shape
        n = a_sources.flatten().shape[0]
        all_sources = np.concatenate([a_sources.flatten(), b_sources.flatten()])
        params = {"half_size": n}
        scale_a = getattr(self, "_ir_add_scale_a", None)
        scale_b = getattr(self, "_ir_add_scale_b", None)
        if scale_a is not None:
            params["scale_a"] = scale_a
        if scale_b is not None:
            params["scale_b"] = scale_b
        return ir_mapping.add_compute_op(
            input_sources=all_sources,
            op_type="add",
            params=params,
            input_shape=(2, *a_sources.shape),
            output_shape=a_sources.shape,
            name="element_add",
        )

    def _forward_impl(self, x):
        a, b = x
        return a + b


class ConcatMapper(Mapper):
    def __init__(self, source_mappers, dim: int = 1, name: str = "Concat"):
        super().__init__()
        self._source_mappers_list = list(source_mappers)
        self.dim = dim
        self._name = name

    def get_source_mappers(self):
        return [m for m in self._source_mappers_list if m is not None]

    def _map(self, mapping):
        layer_sources_list = [mapper.map(mapping) for mapper in self.get_source_mappers()]
        return np.concatenate(layer_sources_list, axis=0)

    def _map_to_ir(self, ir_mapping):
        layer_sources_list = [mapper.map_to_ir(ir_mapping) for mapper in self.get_source_mappers()]
        return np.concatenate(layer_sources_list, axis=0)

    def _forward_impl(self, x):
        return torch.cat(tuple(x), dim=self.dim)


class SubscriptMapper(Mapper):
    def __init__(self, source_mapper, index):
        super(SubscriptMapper, self).__init__(source_mapper)
        self.index = index

    def _map(self, mapping):
        return self.source_mapper.map(mapping)[self.index]

    def _map_to_ir(self, ir_mapping):
        return self.source_mapper.map_to_ir(ir_mapping)[self.index]

    def _forward_impl(self, x):
        return x.select(1, self.index)


class PermuteMapper(Mapper):
    """Mapper for ``tensor.permute(*dims)`` or ``tensor.transpose(d0, d1)``.

    * ``_forward_impl`` applies the true permutation — required for correct
      software validation (ReshapeMapper.view would silently scramble values).
    * ``_map`` / ``_map_to_ir`` use ``np.transpose`` with the batch-stripped
      permutation so that SpikeSource / IRSource arrays are reordered correctly
      for hardware layout.

    ``dims`` is the full permutation tuple including the batch axis 0,
    e.g. ``(0, 2, 1)`` for a 3-D batch-first tensor.
    """

    def __init__(self, source_mapper, dims):
        super(PermuteMapper, self).__init__(source_mapper)
        self.dims = tuple(dims)
        # numpy permutation: drop batch dim 0, shift remaining axes by -1.
        self._np_dims = tuple(d - 1 for d in self.dims if d != 0)

    def _map(self, mapping):
        arr = self.source_mapper.map(mapping)
        return np.transpose(arr, self._np_dims)

    def _map_to_ir(self, ir_mapping):
        arr = self.source_mapper.map_to_ir(ir_mapping)
        return np.transpose(arr, self._np_dims)

    def _forward_impl(self, x):
        return x.permute(*self.dims)


class MeanMapper(Mapper):
    """Mapper for ``tensor.mean(dim=dim)``.

    * ``_forward_impl`` computes the true mean — used for software validation.
    * ``_map`` uses subscript [0] for legacy shape-tracking (SoftCoreMapping).
    * ``_map_to_ir`` creates a ComputeOp with op_type="mean" so the IR graph
      (and spiking simulation) properly averages all groups, not just index 0.
    """

    def __init__(self, source_mapper, dim: int):
        super(MeanMapper, self).__init__(source_mapper)
        self.dim = dim

    def _map(self, mapping):
        # Legacy SoftCoreMapping: shape-tracking only; [0] gives correct output shape.
        return self.source_mapper.map(mapping)[0]

    def _map_to_ir(self, ir_mapping):
        src = self.source_mapper.map_to_ir(ir_mapping)
        # src shape is e.g. (num_patches, features) for mean(dim=1).
        # dim is the batch-relative dim (1 in the original tensor), which maps
        # to axis 0 in the batch-stripped source array.
        reduce_axis = self.dim - 1 if self.dim > 0 else 0
        num_groups = src.shape[reduce_axis]
        # Output shape: remove the reduced axis.
        out_shape = tuple(d for i, d in enumerate(src.shape) if i != reduce_axis)
        group_size = int(np.prod(out_shape)) if out_shape else 1

        # Flatten sources in the order expected by _exec_mean:
        # group 0 features, group 1 features, ..., group N features.
        # Transpose so the reduce axis comes first, then flatten.
        if reduce_axis != 0:
            src = np.moveaxis(src, reduce_axis, 0)
        flat_sources = src.flatten()

        return ir_mapping.add_compute_op(
            input_sources=flat_sources,
            op_type="mean",
            params={"num_groups": num_groups, "group_size": group_size},
            input_shape=(num_groups, group_size),
            output_shape=out_shape if out_shape else (1,),
            name="mean_reduce",
        )

    def _forward_impl(self, x):
        return x.mean(dim=self.dim)
