"""Pool conversion: F.max_pool2d / F.avg_pool2d / F.adaptive_avg_pool2d function calls.

Module-based pool ops (nn.MaxPool2d, etc.) are handled generically by
ModuleComputeMapper in the converter's else branch. These methods handle
the function-call variants where we need to construct an nn.Module from
the function arguments.
"""

from __future__ import annotations

import torch.fx as fx
import torch.nn as nn

from mimarsinan.mapping.mapping_utils import ModuleComputeMapper


class PoolConvertMixin:
    def _convert_adaptive_avgpool2d_func(self, node: fx.Node) -> None:
        source = self._get_source_mapper(node)
        output_size = node.args[1] if len(node.args) > 1 else node.kwargs.get("output_size", (1, 1))
        mod = nn.AdaptiveAvgPool2d(output_size)
        in_shape = self._get_input_shape(node)
        input_shape = tuple(in_shape[1:]) if in_shape and len(in_shape) >= 2 else None
        out_shape = self._get_output_shape(node)
        output_shape = tuple(out_shape[1:]) if out_shape and len(out_shape) >= 2 else None
        mapper = ModuleComputeMapper(source, mod, input_shape=input_shape,
                                     output_shape=output_shape, name=node.name)
        self._node_to_mapper[node] = mapper

    def _convert_maxpool2d_func(self, node: fx.Node) -> None:
        source = self._get_source_mapper(node)
        kernel_size = node.args[1] if len(node.args) > 1 else node.kwargs.get("kernel_size", 2)
        stride = node.args[2] if len(node.args) > 2 else node.kwargs.get("stride", None)
        padding = node.args[3] if len(node.args) > 3 else node.kwargs.get("padding", 0)
        mod = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        in_shape = self._get_input_shape(node)
        input_shape = tuple(in_shape[1:]) if in_shape and len(in_shape) >= 2 else None
        out_shape = self._get_output_shape(node)
        output_shape = tuple(out_shape[1:]) if out_shape and len(out_shape) >= 2 else None
        mapper = ModuleComputeMapper(source, mod, input_shape=input_shape,
                                     output_shape=output_shape, name=node.name)
        self._node_to_mapper[node] = mapper

    def _convert_avgpool2d_func(self, node: fx.Node) -> None:
        source = self._get_source_mapper(node)
        kernel_size = node.args[1] if len(node.args) > 1 else node.kwargs.get("kernel_size", 2)
        stride = node.args[2] if len(node.args) > 2 else node.kwargs.get("stride", None)
        padding = node.args[3] if len(node.args) > 3 else node.kwargs.get("padding", 0)
        mod = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        in_shape = self._get_input_shape(node)
        input_shape = tuple(in_shape[1:]) if in_shape and len(in_shape) >= 2 else None
        out_shape = self._get_output_shape(node)
        output_shape = tuple(out_shape[1:]) if out_shape and len(out_shape) >= 2 else None
        mapper = ModuleComputeMapper(source, mod, input_shape=input_shape,
                                     output_shape=output_shape, name=node.name)
        self._node_to_mapper[node] = mapper
