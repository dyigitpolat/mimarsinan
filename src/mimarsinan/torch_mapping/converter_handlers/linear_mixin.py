"""Linear/FC conversion: _convert_linear, _activation_to_name."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.mapping.mapping_utils import Ensure2DMapper, PerceptronMapper
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

if TYPE_CHECKING:
    from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityReport


class LinearConvertMixin:
    @staticmethod
    def _activation_to_name(act_mod) -> str | None:
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

    def _convert_linear(
        self,
        node: fx.Node,
        mod: nn.Linear,
        source,
        report: RepresentabilityReport,
    ) -> None:
        bn_mod = self._find_absorbed_follower(node, (nn.BatchNorm1d,), report)
        act_mod = self._find_absorbed_follower(
            node, (nn.ReLU, nn.LeakyReLU, nn.GELU), report, skip_bn=True
        )

        normalization = copy.deepcopy(bn_mod) if bn_mod is not None else nn.Identity()
        act_name = self._activation_to_name(act_mod) or "Identity"

        perceptron = Perceptron(
            output_channels=mod.out_features,
            input_features=mod.in_features,
            bias=mod.bias is not None,
            normalization=normalization,
            base_activation_name=act_name,
            name=node.name,
        )

        with torch.no_grad():
            perceptron.layer.weight.copy_(mod.weight.data)
            if mod.bias is not None:
                perceptron.layer.bias.copy_(mod.bias.data)

            if bn_mod is not None and not isinstance(normalization, nn.Identity):
                self._copy_bn_params(normalization, bn_mod)

        source = Ensure2DMapper(source)
        mapper = PerceptronMapper(source, perceptron)
        self._node_to_mapper[node] = mapper
