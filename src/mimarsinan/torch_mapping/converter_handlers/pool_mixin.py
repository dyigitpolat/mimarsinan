"""Pool conversion: max/avg/adaptive_avg pool (module and function)."""

from __future__ import annotations

import torch.fx as fx
import torch.nn as nn

from mimarsinan.mapping.mapping_utils import (
    AdaptiveAvgPool2DMapper,
    AvgPool2DMapper,
    MaxPool2DMapper,
)


class PoolConvertMixin:
    def _convert_maxpool2d(self, node: fx.Node, mod: nn.MaxPool2d, source) -> None:
        in_shape = self._get_input_shape(node)
        c, h, w = (in_shape[1], in_shape[2], in_shape[3]) if in_shape and len(in_shape) == 4 else (None, None, None)
        mapper = MaxPool2DMapper(
            source,
            kernel_size=mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            input_spatial_shape=(h, w) if h is not None else None,
            input_channels=c,
            name=node.name,
        )
        self._node_to_mapper[node] = mapper

    def _convert_avgpool2d(self, node: fx.Node, mod: nn.AvgPool2d, source) -> None:
        in_shape = self._get_input_shape(node)
        c, h, w = (in_shape[1], in_shape[2], in_shape[3]) if in_shape and len(in_shape) == 4 else (None, None, None)
        mapper = AvgPool2DMapper(
            source,
            kernel_size=mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            input_spatial_shape=(h, w) if h is not None else None,
            input_channels=c,
            name=node.name,
        )
        self._node_to_mapper[node] = mapper

    def _convert_adaptive_avgpool2d(self, node: fx.Node, mod: nn.AdaptiveAvgPool2d, source) -> None:
        in_shape = self._get_input_shape(node)
        c = in_shape[1] if in_shape and len(in_shape) == 4 else None
        mapper = AdaptiveAvgPool2DMapper(
            source,
            output_size=mod.output_size,
            input_channels=c,
            name=node.name,
        )
        self._node_to_mapper[node] = mapper

    def _convert_adaptive_avgpool2d_func(self, node: fx.Node) -> None:
        source = self._get_source_mapper(node)
        output_size = node.args[1] if len(node.args) > 1 else node.kwargs.get("output_size", (1, 1))
        in_shape = self._get_input_shape(node)
        c = in_shape[1] if in_shape and len(in_shape) == 4 else None
        mapper = AdaptiveAvgPool2DMapper(
            source,
            output_size=output_size,
            input_channels=c,
            name=node.name,
        )
        self._node_to_mapper[node] = mapper

    def _convert_maxpool2d_func(self, node: fx.Node) -> None:
        source = self._get_source_mapper(node)
        kernel_size = node.args[1] if len(node.args) > 1 else node.kwargs.get("kernel_size", 2)
        stride = node.args[2] if len(node.args) > 2 else node.kwargs.get("stride", None)
        padding = node.args[3] if len(node.args) > 3 else node.kwargs.get("padding", 0)
        in_shape = self._get_input_shape(node)
        c, h, w = (in_shape[1], in_shape[2], in_shape[3]) if in_shape and len(in_shape) == 4 else (None, None, None)
        mapper = MaxPool2DMapper(
            source,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            input_spatial_shape=(h, w) if h is not None else None,
            input_channels=c,
            name=node.name,
        )
        self._node_to_mapper[node] = mapper

    def _convert_avgpool2d_func(self, node: fx.Node) -> None:
        source = self._get_source_mapper(node)
        kernel_size = node.args[1] if len(node.args) > 1 else node.kwargs.get("kernel_size", 2)
        stride = node.args[2] if len(node.args) > 2 else node.kwargs.get("stride", None)
        padding = node.args[3] if len(node.args) > 3 else node.kwargs.get("padding", 0)
        in_shape = self._get_input_shape(node)
        c, h, w = (in_shape[1], in_shape[2], in_shape[3]) if in_shape and len(in_shape) == 4 else (None, None, None)
        mapper = AvgPool2DMapper(
            source,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            input_spatial_shape=(h, w) if h is not None else None,
            input_channels=c,
            name=node.name,
        )
        self._node_to_mapper[node] = mapper
