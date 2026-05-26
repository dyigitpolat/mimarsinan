"""Structural shortcuts: cat / flatten + BatchNorm/activation absorption helpers.

Compute ops (``+``, ``getitem``, ``mean``, generic functions, ...) flow
through :meth:`MapperGraphConverter._emit_generic_compute_op` instead of
op-specific handlers — see ``mapper_graph_converter.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.mapping.compute_modules import ComputeAdapter, _cat_along
from mimarsinan.mapping.mapping_utils import (
    ComputeOpMapper,
    ConcatMapper,
    ReshapeMapper,
)

if TYPE_CHECKING:
    from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityReport


class StructuralConvertMixin:
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
                # Normalise to a single batch-stripped token tensor of shape
                # ``(1, D)`` so ``ComputeAdapter`` auto-expands to (B, 1, D)
                # at forward time and ``_cat_along`` produces ``(B, S+1, D)``.
                prefix = first_const.detach().clone()
                if prefix.dim() == 1:
                    prefix = prefix.view(1, -1)
                elif prefix.dim() == 3 and prefix.shape[0] == 1:
                    prefix = prefix.squeeze(0)
                self._node_to_mapper[node] = ComputeOpMapper(
                    second_mapper,
                    ComputeAdapter(
                        _cat_along,
                        bound_tensors=[prefix],
                        kwargs={"dim": int(dim)},
                    ),
                    name=node.name,
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
