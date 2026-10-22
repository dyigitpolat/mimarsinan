"""Conv1D / Conv2D conversion with BN/activation absorption."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mapping_utils import ComputeOpMapper, Conv2DPerceptronMapper
from mimarsinan.models.perceptron_mixer.perceptron import as_affine_package
from mimarsinan.torch_mapping.converter_handlers.converter_contract import ConverterContract
from mimarsinan.mapping.mappers.conv_helpers import perceptron_conv_padding
from mimarsinan.torch_mapping.representability_analyzer import (
    ABSORBABLE_ACTIVATIONS,
    absorbed_activation_name,
    lowers_to_perceptron,
)
from mimarsinan.torch_mapping.fx_shape_utils import strip_batch

if TYPE_CHECKING:
    from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityReport


class UnsupportedConvPaddingError(ToolchainRefusal, NotImplementedError):
    """A conv bound for a perceptron mapper whose padding the patch gather cannot express."""

    def __init__(self, node: fx.Node, mod: nn.Conv1d | nn.Conv2d, reason: str):
        self.node_name = node.name
        self.operator = type(mod).__name__
        self.padding_mode = str(mod.padding_mode)
        super().__init__(
            f"{self.operator} (FX node {node.name!r}): {reason}; the perceptron conv "
            "mapping implements numeric zero padding only"
        )


class ConvConvertMixin(ConverterContract):
    @staticmethod
    def _require_zero_padding(
        node: fx.Node, mod: nn.Conv1d | nn.Conv2d,
    ) -> tuple[int, ...]:
        """Numeric zero padding of a conv bound for a perceptron mapper (representability refused the rest)."""
        padding, reason = perceptron_conv_padding(mod)
        if padding is None:
            raise UnsupportedConvPaddingError(node, mod, str(reason))
        return padding

    def _host_conv_shapes(
        self, node: fx.Node,
    ) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
        """Batch-stripped FX shapes: a host conv op runs on the whole (C, *spatial) grid, never per column."""
        return strip_batch(self._get_input_shape(node)), strip_batch(self._get_output_shape(node))

    def _convert_conv2d(
        self,
        node: fx.Node,
        mod: nn.Conv2d,
        source,
        report: RepresentabilityReport,
    ) -> None:
        bn_mod = self._find_absorbed_follower(node, (nn.BatchNorm2d,), report)
        act_mod = self._find_absorbed_follower(
            node, ABSORBABLE_ACTIVATIONS, report, skip_bn=True
        )
        act_name = absorbed_activation_name(act_mod)

        if not lowers_to_perceptron(act_name is not None, self._packaging):
            conv_copy = copy.deepcopy(mod)
            if bn_mod is not None:
                bn_copy = copy.deepcopy(bn_mod)
                module = nn.Sequential(conv_copy, bn_copy)
            else:
                module = conv_copy
            input_shape, output_shape = self._host_conv_shapes(node)
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
                padding=self._require_zero_padding(node, mod),
                dilation=mod.dilation,
                bias=mod.bias is not None,
                use_batchnorm=bn_mod is not None,
                base_activation_name=act_name,
                name=node.name,
            )
            if act_name is None:
                as_affine_package(conv_mapper.perceptron)
            with torch.no_grad():
                # Follow the source conv's dtype (no silent fp32 downcast).
                conv_mapper.perceptron.layer.to(mod.weight.dtype)
                flat_weight = mod.weight.data.reshape(mod.out_channels, -1)
                conv_mapper.perceptron.layer.weight.copy_(flat_weight)
                if mod.bias is not None:
                    conv_mapper.perceptron.layer.bias.copy_(mod.bias.data)
                if bn_mod is not None:
                    conv_mapper.perceptron.normalization = self._perceptron_normalization(bn_mod)
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
            node, ABSORBABLE_ACTIVATIONS, report, skip_bn=True
        )
        act_name = absorbed_activation_name(act_mod)

        if not lowers_to_perceptron(act_name is not None, self._packaging):
            conv_copy = copy.deepcopy(mod)
            if bn_mod is not None:
                bn_copy = copy.deepcopy(bn_mod)
                module = nn.Sequential(conv_copy, bn_copy)
            else:
                module = conv_copy
            input_shape, output_shape = self._host_conv_shapes(node)
            mapper = ComputeOpMapper(
                source, module, input_shape=input_shape,
                output_shape=output_shape, name=node.name,
            )
        else:
            conv_mapper = Conv1DPerceptronMapper(
                source,
                in_channels=mod.in_channels,
                out_channels=mod.out_channels,
                kernel_size=mod.kernel_size[0],
                stride=mod.stride[0],
                padding=self._require_zero_padding(node, mod)[0],
                dilation=mod.dilation[0],
                bias=mod.bias is not None,
                use_batchnorm=bn_mod is not None,
                base_activation_name=act_name,
                name=node.name,
            )
            if act_name is None:
                as_affine_package(conv_mapper.perceptron)
            with torch.no_grad():
                # Follow the source conv's dtype (no silent fp32 downcast).
                conv_mapper.perceptron.layer.to(mod.weight.dtype)
                flat_weight = mod.weight.data.reshape(mod.out_channels, -1)
                conv_mapper.perceptron.layer.weight.copy_(flat_weight)
                if mod.bias is not None:
                    conv_mapper.perceptron.layer.bias.copy_(mod.bias.data)
                if bn_mod is not None:
                    conv_mapper.perceptron.normalization = self._perceptron_normalization(bn_mod)
            mapper = conv_mapper

        self._node_to_mapper[node] = mapper
