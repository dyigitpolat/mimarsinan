"""FX graph shape and argument utilities for :class:`MapperGraphConverter`."""

from __future__ import annotations

import operator
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.mapping.mapping_utils import ComputeOpMapper


class MapperGraphFxMixin:
    """Shape metadata and generic compute-op emission for mapper conversion."""

    _node_to_mapper: Dict[fx.Node, Any]
    _node_to_attr: Dict[Any, Any]

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
                    bound.append(const)
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
        in_shape = self._get_input_shape(node)
        input_shape = (
            tuple(in_shape[1:]) if in_shape and len(in_shape) >= 2 else None
        )
        out_shape = self._get_output_shape(node)
        output_shape = (
            tuple(out_shape[1:]) if out_shape and len(out_shape) >= 2 else None
        )
        self._node_to_mapper[node] = ComputeOpMapper(
            sources,
            adapter,
            input_shapes=input_shape,
            output_shape=output_shape,
            name=node.name,
        )

    def _get_input_shape(self, node: fx.Node) -> Optional[Tuple[int, ...]]:
        if len(node.args) >= 1 and isinstance(node.args[0], fx.Node):
            input_node = node.args[0]
            meta = input_node.meta.get("tensor_meta")
            if meta is not None and hasattr(meta, "shape"):
                return tuple(meta.shape)
        return None

    def _get_output_shape(self, node: fx.Node) -> Optional[Tuple[int, ...]]:
        meta = node.meta.get("tensor_meta")
        if meta is not None and hasattr(meta, "shape"):
            return tuple(meta.shape)
        return None

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
