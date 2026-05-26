"""Structural mappers: Input, Reshape, EinopsRearrange, Stack, Concat, Subscript, Permute.

Multi-input element-wise add and mean-reduction are now expressed as
``ComputeOpMapper(sources, Add())`` / ``ComputeOpMapper(source, Mean(dim))``
— see :mod:`mimarsinan.mapping.compute_modules`.
"""

from __future__ import annotations

import numpy as np
import torch

import einops

from mimarsinan.mapping.ir import IRSource

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

    def _map_to_ir(self, ir_mapping):
        return self.source_mapper.map_to_ir(ir_mapping).reshape(self.output_shape)

    def _forward_impl(self, x):
        return x.view(x.shape[0], *self.output_shape)


class EinopsRearrangeMapper(Mapper):
    def __init__(self, source_mapper, einops_str, *einops_args, **einops_kwargs):
        super(EinopsRearrangeMapper, self).__init__(source_mapper)
        self.einops_str = einops_str
        self.einops_args = einops_args
        self.einops_kwargs = einops_kwargs

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

    def _map_to_ir(self, ir_mapping):
        layer_sources_list = [mapper.map_to_ir(ir_mapping) for mapper in self.source_mappers]
        return np.stack(layer_sources_list).squeeze()

    def _forward_impl(self, x):
        outputs = list(x)
        return torch.stack(outputs, dim=1).squeeze(1)


class ConcatMapper(Mapper):
    def __init__(self, source_mappers, dim: int = 1, name: str = "Concat"):
        super().__init__()
        self._source_mappers_list = list(source_mappers)
        self.dim = dim
        self._name = name

    def get_source_mappers(self):
        return [m for m in self._source_mappers_list if m is not None]

    def _map_to_ir(self, ir_mapping):
        layer_sources_list = [mapper.map_to_ir(ir_mapping) for mapper in self.get_source_mappers()]
        return np.concatenate(layer_sources_list, axis=0)

    def _forward_impl(self, x):
        return torch.cat(tuple(x), dim=self.dim)


class SubscriptMapper(Mapper):
    def __init__(self, source_mapper, index):
        super(SubscriptMapper, self).__init__(source_mapper)
        self.index = index

    def _map_to_ir(self, ir_mapping):
        return self.source_mapper.map_to_ir(ir_mapping)[self.index]

    def _forward_impl(self, x):
        return x.select(1, self.index)


class PermuteMapper(Mapper):
    """Mapper for ``tensor.permute(*dims)`` or ``tensor.transpose(d0, d1)``.

    * ``_forward_impl`` applies the true permutation — required for correct
      software validation (ReshapeMapper.view would silently scramble values).
    * ``_map_to_ir`` uses ``np.transpose`` with the batch-stripped
      permutation so that IRSource arrays are reordered correctly
      for hardware layout.

    ``dims`` is the full permutation tuple including the batch axis 0,
    e.g. ``(0, 2, 1)`` for a 3-D batch-first tensor.
    """

    def __init__(self, source_mapper, dims):
        super(PermuteMapper, self).__init__(source_mapper)
        self.dims = tuple(dims)
        # numpy permutation: drop batch dim 0, shift remaining axes by -1.
        self._np_dims = tuple(d - 1 for d in self.dims if d != 0)

    def _map_to_ir(self, ir_mapping):
        arr = self.source_mapper.map_to_ir(ir_mapping)
        return np.transpose(arr, self._np_dims)

    def _forward_impl(self, x):
        return x.permute(*self.dims)


