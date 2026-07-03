"""FX graph shape and argument utilities for :class:`MapperGraphConverter`."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.torch_mapping.converter_handlers.converter_contract import ConverterContract
from mimarsinan.torch_mapping.fx_shape_utils import (
    node_input_shapes,
    node_output_shape,
    strip_batch,
)


def _normalize_bound_tensor(t: torch.Tensor) -> torch.Tensor:
    """Strip a leading singleton dim so ``ComputeAdapter`` can re-add the batch dim."""
    if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] == 1:
        return t.squeeze(0)
    return t


class MapperGraphFxMixin(ConverterContract):
    """Shape metadata and generic compute-op emission for mapper conversion."""

    @staticmethod
    def _getitem_looks_like_real_slice(node: fx.Node) -> bool:
        if len(node.args) < 2:
            return False
        index = node.args[1]
        if isinstance(index, tuple):
            return any(
                isinstance(part, slice) or part is Ellipsis for part in index
            )
        return False

    def _partition_fx_args(
        self, node: fx.Node,
    ) -> Tuple[list, list, tuple, dict]:
        sources: list = []
        bound: list = []
        extra: list = []
        for arg in node.args:
            if isinstance(arg, fx.Node):
                const = self._get_constant_tensor(arg)
                if const is not None:
                    bound.append(_normalize_bound_tensor(const))
                    continue
                mapper = self._get_mapper(arg)
                if mapper is not None:
                    sources.append(mapper)
                continue
            extra.append(arg)
        kwargs = {
            k: v for k, v in node.kwargs.items() if not isinstance(v, fx.Node)
        }
        return sources, bound, tuple(extra), kwargs

    def _emit_generic_compute_op(self, node: fx.Node, fn) -> None:
        sources, bound, extra, kwargs = self._partition_fx_args(node)
        if not sources:
            self._node_to_mapper[node] = None
            return
        adapter = ComputeAdapter(
            fn,
            bound_tensors=bound,
            extra_args=extra,
            kwargs=kwargs,
        )
        source_shapes = self._source_shapes_for(node)
        output_shape = strip_batch(node_output_shape(node))
        if len(sources) == 1:
            input_shapes_arg: Any = source_shapes[0] if source_shapes else None
        else:
            input_shapes_arg = source_shapes
        self._node_to_mapper[node] = ComputeOpMapper(
            sources,
            adapter,
            input_shapes=input_shapes_arg,
            output_shape=output_shape,
            name=node.name,
        )

    def _source_shapes_for(self, node: fx.Node) -> list:
        """Per-source batch-stripped FX shapes, mirroring ``_partition_fx_args``'s filter."""
        shapes = []
        for arg in node.args:
            if not isinstance(arg, fx.Node):
                continue
            if self._get_constant_tensor(arg) is not None:
                continue
            if self._get_mapper(arg) is None:
                continue
            shapes.append(strip_batch(node_output_shape(arg)))
        return shapes

    def _get_input_shape(self, node: fx.Node) -> Optional[Tuple[int, ...]]:
        shapes = node_input_shapes(node)
        return shapes[0] if shapes else None

    def _get_output_shape(self, node: fx.Node) -> Optional[Tuple[int, ...]]:
        return node_output_shape(node)

    def _get_constant_tensor(self, node: fx.Node):
        if isinstance(node, fx.Node) and node.op == "get_attr":
            value = self._get_attr_value(node)
            if isinstance(value, (nn.Parameter, torch.Tensor)):
                return value
        return None

    def _get_expanded_constant_tensor(self, node: fx.Node):
        if (
            isinstance(node, fx.Node)
            and node.op == "call_method"
            and node.target == "expand"
            and len(node.args) >= 1
            and isinstance(node.args[0], fx.Node)
        ):
            return self._get_constant_tensor(node.args[0])
        return None
