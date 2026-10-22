"""Linear/FC conversion with BN/activation absorption."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.mapping.mapping_utils import ComputeOpMapper, Ensure2DMapper, PerceptronMapper
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron, as_affine_package
from mimarsinan.torch_mapping.converter_handlers.converter_contract import ConverterContract
from mimarsinan.torch_mapping.representability_analyzer import (
    ABSORBABLE_ACTIVATIONS,
    absorbed_activation_name,
    lowers_to_perceptron,
)

if TYPE_CHECKING:
    from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityReport


class LinearConvertMixin(ConverterContract):
    def _convert_linear(
        self,
        node: fx.Node,
        mod: nn.Linear,
        source,
        report: RepresentabilityReport,
    ) -> None:
        bn_mod = self._find_absorbed_follower(node, (nn.BatchNorm1d,), report)
        act_mod = self._find_absorbed_follower(
            node, ABSORBABLE_ACTIVATIONS, report, skip_bn=True
        )
        act_name = absorbed_activation_name(act_mod)

        source = Ensure2DMapper(source)

        if not lowers_to_perceptron(act_name is not None, self._packaging):
            linear = copy.deepcopy(mod)
            if bn_mod is not None:
                bn_copy = copy.deepcopy(bn_mod)
                module = nn.Sequential(linear, bn_copy)
            else:
                module = linear
            mapper = ComputeOpMapper(
                source, module,
                output_shape=(mod.out_features,),
                name=node.name,
            )
        else:
            normalization = self._perceptron_normalization(bn_mod) if bn_mod is not None else nn.Identity()
            perceptron = Perceptron(
                output_channels=mod.out_features,
                input_features=mod.in_features,
                bias=mod.bias is not None,
                normalization=normalization,
                base_activation_name=act_name,
                name=node.name,
            )
            if act_name is None:
                as_affine_package(perceptron)
            with torch.no_grad():
                # Follow the source linear's dtype (no silent fp32 downcast).
                perceptron.layer.to(mod.weight.dtype)
                perceptron.layer.weight.copy_(mod.weight.data)
                if mod.bias is not None:
                    perceptron.layer.bias.copy_(mod.bias.data)
            mapper = PerceptronMapper(source, perceptron)

        self._node_to_mapper[node] = mapper
