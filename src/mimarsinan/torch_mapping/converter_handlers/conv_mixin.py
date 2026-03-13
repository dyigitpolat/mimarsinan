"""Conv conversion: _convert_conv2d, _convert_conv1d, _copy_bn_params."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.mapping.mapping_utils import Conv2DPerceptronMapper

if TYPE_CHECKING:
    from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityReport


class ConvConvertMixin:
    def _convert_conv2d(
        self,
        node: fx.Node,
        mod: nn.Conv2d,
        source,
        report: RepresentabilityReport,
    ) -> None:
        bn_mod = self._find_absorbed_follower(node, (nn.BatchNorm2d,), report)

        conv_mapper = Conv2DPerceptronMapper(
            source,
            in_channels=mod.in_channels,
            out_channels=mod.out_channels,
            kernel_size=mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            bias=mod.bias is not None,
            use_batchnorm=bn_mod is not None,
            name=node.name,
        )

        with torch.no_grad():
            flat_weight = mod.weight.data.reshape(mod.out_channels, -1)
            conv_mapper.perceptron.layer.weight.copy_(flat_weight)
            if mod.bias is not None:
                conv_mapper.perceptron.layer.bias.copy_(mod.bias.data)

            if bn_mod is not None:
                concrete_bn = nn.BatchNorm1d(mod.out_channels)
                self._copy_bn_params(concrete_bn, bn_mod)
                conv_mapper.perceptron.normalization = concrete_bn

        self._node_to_mapper[node] = conv_mapper

    def _convert_conv1d(
        self,
        node: fx.Node,
        mod: nn.Conv1d,
        source,
        report: RepresentabilityReport,
    ) -> None:
        from mimarsinan.mapping.mapping_utils import Conv1DPerceptronMapper

        bn_mod = self._find_absorbed_follower(node, (nn.BatchNorm1d,), report)

        conv_mapper = Conv1DPerceptronMapper(
            source,
            in_channels=mod.in_channels,
            out_channels=mod.out_channels,
            kernel_size=mod.kernel_size[0],
            stride=mod.stride[0],
            padding=mod.padding[0],
            dilation=mod.dilation[0],
            bias=mod.bias is not None,
            use_batchnorm=bn_mod is not None,
            name=node.name,
        )

        with torch.no_grad():
            flat_weight = mod.weight.data.reshape(mod.out_channels, -1)
            conv_mapper.perceptron.layer.weight.copy_(flat_weight)
            if mod.bias is not None:
                conv_mapper.perceptron.layer.bias.copy_(mod.bias.data)

            if bn_mod is not None:
                concrete_bn = nn.BatchNorm1d(mod.out_channels)
                self._copy_bn_params(concrete_bn, bn_mod)
                conv_mapper.perceptron.normalization = concrete_bn

        self._node_to_mapper[node] = conv_mapper

    @staticmethod
    def _copy_bn_params(dst_bn: nn.Module, src_bn: nn.Module) -> None:
        """Copy BatchNorm parameters from source to destination."""
        if isinstance(dst_bn, nn.Identity):
            return

        if hasattr(dst_bn, "weight") and hasattr(src_bn, "weight") and src_bn.weight is not None:
            dst_bn.weight.data.copy_(src_bn.weight.data)
        if hasattr(dst_bn, "bias") and hasattr(src_bn, "bias") and src_bn.bias is not None:
            dst_bn.bias.data.copy_(src_bn.bias.data)
        if hasattr(dst_bn, "running_mean") and hasattr(src_bn, "running_mean") and src_bn.running_mean is not None:
            dst_bn.running_mean.copy_(src_bn.running_mean)
        if hasattr(dst_bn, "running_var") and hasattr(src_bn, "running_var") and src_bn.running_var is not None:
            dst_bn.running_var.copy_(src_bn.running_var)
        if hasattr(dst_bn, "num_batches_tracked") and hasattr(src_bn, "num_batches_tracked"):
            dst_bn.num_batches_tracked.copy_(src_bn.num_batches_tracked)
