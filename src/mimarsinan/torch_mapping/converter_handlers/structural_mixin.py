"""Structural conversion: add, flatten, getitem, cat, and absorption helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.mapping.mapping_utils import (
    AddMapper,
    ConcatMapper,
    ConstantAddMapper,
    ConstantPrependMapper,
    ReshapeMapper,
    SelectMapper,
)

if TYPE_CHECKING:
    from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityReport


class StructuralConvertMixin:
    def _convert_add(self, node: fx.Node) -> None:
        if len(node.args) < 2:
            source = self._get_source_mapper(node)
            self._node_to_mapper[node] = source
            return

        a_node = node.args[0]
        b_node = node.args[1]

        a_mapper = self._get_mapper(a_node) if isinstance(a_node, fx.Node) else None
        b_mapper = self._get_mapper(b_node) if isinstance(b_node, fx.Node) else None
        a_const = self._get_constant_tensor(a_node) if isinstance(a_node, fx.Node) else None
        b_const = self._get_constant_tensor(b_node) if isinstance(b_node, fx.Node) else None

        if a_mapper is not None and b_mapper is not None:
            mapper = AddMapper(a_mapper, b_mapper)
            self._node_to_mapper[node] = mapper
        elif a_mapper is not None and isinstance(b_const, (nn.Parameter, torch.Tensor)):
            self._node_to_mapper[node] = ConstantAddMapper(a_mapper, b_const, name=node.name)
        elif b_mapper is not None and isinstance(a_const, (nn.Parameter, torch.Tensor)):
            self._node_to_mapper[node] = ConstantAddMapper(b_mapper, a_const, name=node.name)
        elif a_mapper is not None:
            self._node_to_mapper[node] = a_mapper
        elif b_mapper is not None:
            self._node_to_mapper[node] = b_mapper

    def _convert_flatten_func(self, node: fx.Node) -> None:
        source = self._get_source_mapper(node)
        out_shape = self._get_output_shape(node)
        if out_shape is not None and len(out_shape) == 2:
            flat_shape = (out_shape[-1],)
            mapper = ReshapeMapper(source, flat_shape)
        else:
            mapper = source
        self._node_to_mapper[node] = mapper

    def _convert_flatten_module(self, node: fx.Node, source) -> None:
        """Convert nn.Flatten (call_module) to ReshapeMapper."""
        out_shape = self._get_output_shape(node)
        if out_shape is not None and len(out_shape) == 2:
            flat_shape = (out_shape[-1],)
            mapper = ReshapeMapper(source, flat_shape)
        else:
            mapper = source
        self._node_to_mapper[node] = mapper

    def _convert_getitem(self, node: fx.Node) -> None:
        source = self._get_source_mapper(node)
        index = node.args[1] if len(node.args) > 1 else None
        if (
            source is not None
            and isinstance(index, tuple)
            and len(index) == 2
            and isinstance(index[0], slice)
            and index[0] == slice(None, None, None)
            and isinstance(index[1], int)
        ):
            self._node_to_mapper[node] = SelectMapper(source, index=index[1], name=node.name)
        else:
            self._node_to_mapper[node] = source

    def _convert_cat(self, node: fx.Node) -> None:
        tensors_arg = node.args[0] if node.args else []
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 1)
        if not isinstance(tensors_arg, (list, tuple)):
            source = self._get_source_mapper(node)
            self._node_to_mapper[node] = source
            return
        if len(tensors_arg) == 2 and int(dim) == 1:
            first, second = tensors_arg
            first_const = self._get_expanded_constant_tensor(first) if isinstance(first, fx.Node) else None
            second_mapper = self._get_mapper(second) if isinstance(second, fx.Node) else None
            if isinstance(first_const, (nn.Parameter, torch.Tensor)) and second_mapper is not None:
                self._node_to_mapper[node] = ConstantPrependMapper(
                    second_mapper, first_const, name=node.name
                )
                return
        source_mappers = []
        for arg in tensors_arg:
            if isinstance(arg, fx.Node):
                m = self._get_mapper(arg)
                if m is not None:
                    source_mappers.append(m)
        if not source_mappers:
            self._node_to_mapper[node] = None
            return
        if len(source_mappers) == 1:
            self._node_to_mapper[node] = source_mappers[0]
            return
        mapper = ConcatMapper(source_mappers, dim=dim, name=node.name)
        self._node_to_mapper[node] = mapper

    def _propagate_absorbed(self, node: fx.Node) -> None:
        """For an absorbed node, point it at its source's mapper."""
        if len(node.args) >= 1 and isinstance(node.args[0], fx.Node):
            self._node_to_mapper[node] = self._get_mapper(node.args[0])

    def _find_absorbed_follower(
        self,
        node: fx.Node,
        target_types: tuple,
        report: RepresentabilityReport,
        skip_bn: bool = False,
    ):
        """Find the first absorbed follower of ``node`` matching ``target_types``.

        Skips through absorbed Identity modules (always) and BatchNorm
        modules (when ``skip_bn=True``) to reach the target.
        """
        for user in node.users:
            if user.name in self._absorbed and user.op == "call_module":
                mod = self._modules.get(user.target)
                if mod is not None and isinstance(mod, target_types):
                    return mod
                # Skip through Identity (always) and BN (when skip_bn)
                if isinstance(mod, nn.Identity):
                    return self._find_absorbed_follower(user, target_types, report, skip_bn)
                if skip_bn and isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    return self._find_absorbed_follower(user, target_types, report, skip_bn)
        return None
