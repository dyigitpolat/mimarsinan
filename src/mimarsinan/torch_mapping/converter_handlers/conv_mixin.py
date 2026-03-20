"""Conv conversion: _convert_conv2d, _convert_conv1d, _copy_bn_params."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.mapping.mapping_utils import Conv2DPerceptronMapper, ModuleComputeMapper

if TYPE_CHECKING:
    from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityReport


class ConvConvertMixin:
    @staticmethod
    def _conv_activation_to_name(act_mod) -> str | None:
        """Map a PyTorch activation module to a Perceptron activation name string."""
        if act_mod is None:
            return None
        if isinstance(act_mod, nn.ReLU):
            return "ReLU"
        if isinstance(act_mod, nn.LeakyReLU):
            return "LeakyReLU"
        if isinstance(act_mod, nn.GELU):
            return "GELU"
        return None

    def _convert_conv2d(
        self,
        node: fx.Node,
        mod: nn.Conv2d,
        source,
        report: RepresentabilityReport,
    ) -> None:
        bn_mod = self._find_absorbed_follower(node, (nn.BatchNorm2d,), report)
        act_mod = self._find_absorbed_follower(
            node, (nn.ReLU, nn.LeakyReLU, nn.GELU), report, skip_bn=True
        )
        act_name = self._conv_activation_to_name(act_mod)

        if act_name is None:
            # No activation detected → generic compute op
            conv_copy = copy.deepcopy(mod)
            if bn_mod is not None:
                bn_copy = copy.deepcopy(bn_mod)
                module = nn.Sequential(conv_copy, bn_copy)
            else:
                module = conv_copy
            # Infer output shape from FX metadata (strip batch dim)
            out_shape_with_batch = self._get_output_shape(node)
            output_shape = tuple(out_shape_with_batch[1:]) if out_shape_with_batch else None
            input_shape_with_batch = self._get_input_shape(node)
            input_shape = tuple(input_shape_with_batch[1:]) if input_shape_with_batch else None
            mapper = ModuleComputeMapper(
                source, module, input_shape=input_shape,
                output_shape=output_shape, name=node.name,
            )
        else:
            # Activation detected → perceptron (will be deployed on chip)
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
                base_activation_name=act_name,
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
            mapper = conv_mapper

        self._node_to_mapper[node] = mapper

    def _convert_conv1d(
        self,
        node: fx.Node,
        mod: nn.Conv1d,
        source,
        report: RepresentabilityReport,
    ) -> None:
        from mimarsinan.mapping.mapping_utils import Conv1DPerceptronMapper

        bn_mod = self._find_absorbed_follower(node, (nn.BatchNorm1d,), report)
        act_mod = self._find_absorbed_follower(
            node, (nn.ReLU, nn.LeakyReLU, nn.GELU), report, skip_bn=True
        )
        act_name = self._conv_activation_to_name(act_mod)

        if act_name is None:
            # No activation → generic compute op
            conv_copy = copy.deepcopy(mod)
            if bn_mod is not None:
                bn_copy = copy.deepcopy(bn_mod)
                module = nn.Sequential(conv_copy, bn_copy)
            else:
                module = conv_copy
            mapper = ModuleComputeMapper(source, module, name=node.name)
        else:
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
                base_activation_name=act_name,
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
            mapper = conv_mapper

        self._node_to_mapper[node] = mapper

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
