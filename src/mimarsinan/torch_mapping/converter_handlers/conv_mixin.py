"""Conv1D / Conv2D conversion with BN/activation absorption."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.mapping.mapping_utils import ComputeOpMapper, Conv2DPerceptronMapper
from mimarsinan.torch_mapping.converter_handlers.converter_contract import ConverterContract

if TYPE_CHECKING:
    from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityReport


_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d)


class ConvConvertMixin(ConverterContract):
    @staticmethod
    def _require_int_padding(
        node: fx.Node, padding: str | tuple[int, ...],
    ) -> tuple[int, ...]:
        """Perceptron conv mappers need numeric padding; reject 'same'/'valid' explicitly."""
        if isinstance(padding, str):
            raise NotImplementedError(
                f"String padding {padding!r} is not supported for perceptron conv "
                f"conversion (node {node.name})"
            )
        return padding

    @staticmethod
    def _conv_activation_to_name(act_mod) -> str | None:
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
            conv_copy = copy.deepcopy(mod)
            if bn_mod is not None:
                bn_copy = copy.deepcopy(bn_mod)
                module = nn.Sequential(conv_copy, bn_copy)
            else:
                module = conv_copy
            out_shape_with_batch = self._get_output_shape(node)
            output_shape = tuple(out_shape_with_batch[1:]) if out_shape_with_batch else None
            input_shape_with_batch = self._get_input_shape(node)
            input_shape = tuple(input_shape_with_batch[1:]) if input_shape_with_batch else None
            mapper = ComputeOpMapper(
                source, module, input_shape=input_shape,
                output_shape=output_shape, name=node.name,
            )
        else:
            conv_mapper = Conv2DPerceptronMapper(
                source,
                in_channels=mod.in_channels,
                out_channels=mod.out_channels,
                kernel_size=mod.kernel_size,
                stride=mod.stride,
                padding=self._require_int_padding(node, mod.padding),
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
            conv_copy = copy.deepcopy(mod)
            if bn_mod is not None:
                bn_copy = copy.deepcopy(bn_mod)
                module = nn.Sequential(conv_copy, bn_copy)
            else:
                module = conv_copy
            mapper = ComputeOpMapper(source, module, name=node.name)
        else:
            conv_mapper = Conv1DPerceptronMapper(
                source,
                in_channels=mod.in_channels,
                out_channels=mod.out_channels,
                kernel_size=mod.kernel_size[0],
                stride=mod.stride[0],
                padding=self._require_int_padding(node, mod.padding)[0],
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
        """Copy affine params and running stats between BatchNorms (possibly 2d src → 1d dst)."""
        if not isinstance(dst_bn, _BN_TYPES) or not isinstance(src_bn, _BN_TYPES):
            return

        if src_bn.weight is not None and dst_bn.weight is not None:
            dst_bn.weight.data.copy_(src_bn.weight.data)
        if src_bn.bias is not None and dst_bn.bias is not None:
            dst_bn.bias.data.copy_(src_bn.bias.data)
        if src_bn.running_mean is not None and dst_bn.running_mean is not None:
            dst_bn.running_mean.copy_(src_bn.running_mean)
        if src_bn.running_var is not None and dst_bn.running_var is not None:
            dst_bn.running_var.copy_(src_bn.running_var)
        if src_bn.num_batches_tracked is not None and dst_bn.num_batches_tracked is not None:
            dst_bn.num_batches_tracked.copy_(src_bn.num_batches_tracked)
