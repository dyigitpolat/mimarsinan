"""PerceptronMapper and ModuleMapper: FC/module application mappers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.base import Mapper
from mimarsinan.mapping.soft_core_mapper import map_mm
from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer


class PerceptronMapper(Mapper):
    def __init__(self, source_mapper, perceptron):
        super(PerceptronMapper, self).__init__(source_mapper)
        self.perceptron = perceptron

    def _map(self, mapping):
        layer_weights = PerceptronTransformer().get_effective_weight(self.perceptron)
        layer_biases = PerceptronTransformer().get_effective_bias(self.perceptron)

        layer_weights.detach().cpu().numpy()
        layer_biases.detach().cpu().numpy()

        layer_sources = self.source_mapper.map(mapping)
        layer_sources = layer_sources.transpose()
        layer_sources = map_mm(
            mapping,
            layer_sources,
            layer_weights,
            layer_biases,
            self.perceptron.activation_scale,
            self.perceptron.parameter_scale,
            self.perceptron.input_activation_scale,
        )
        layer_sources = layer_sources.transpose()

        return layer_sources

    def _map_to_ir(self, ir_mapping):
        layer_weights = PerceptronTransformer().get_effective_weight(self.perceptron)
        layer_biases = PerceptronTransformer().get_effective_bias(self.perceptron)

        normalization = getattr(self.perceptron, "normalization", None)
        normalization_type = type(normalization).__name__ if normalization is not None else None

        activation = getattr(self.perceptron, "activation", None)
        activation_type = type(activation).__name__ if activation is not None else None
        if activation is not None and hasattr(activation, "base_activation") and hasattr(activation, "decorators"):
            base = getattr(activation, "base_activation", None)
            base_name = type(base).__name__ if base is not None else "Activation"
            decorators = getattr(activation, "decorators", []) or []
            decorator_names = [type(d).__name__ for d in decorators]
            activation_type = f"{base_name} + {', '.join(decorator_names)}" if decorator_names else base_name

        layer_sources = self.source_mapper.map_to_ir(ir_mapping)
        layer_sources = layer_sources.transpose()

        output_shape = np.array([layer_weights.shape[0], layer_sources.shape[-1]])
        layer_sources = ir_mapping.map_fc(
            layer_sources,
            output_shape,
            layer_weights,
            layer_biases,
            self.perceptron.activation_scale,
            self.perceptron.parameter_scale,
            self.perceptron.input_activation_scale,
            name=getattr(self.perceptron, "name", None),
            normalization_type=normalization_type,
            activation_type=activation_type,
            perceptron_index=getattr(self, "perceptron_index", None),
        )

        return layer_sources.transpose()

    def _forward_impl(self, x):
        return self.perceptron(x)

    def owned_perceptron_groups(self):
        return [[self.perceptron]]


class ModuleMapper(Mapper):
    """
    Forward-only module application mapper.
    For mapping (chip compilation), this acts as identity (passes sources through).
    """

    def __init__(self, source_mapper, module: nn.Module):
        super(ModuleMapper, self).__init__(source_mapper)
        self.module = module

    def _map(self, mapping):
        return self.source_mapper.map(mapping)

    def _map_to_ir(self, ir_mapping):
        return self.source_mapper.map_to_ir(ir_mapping)

    def _forward_impl(self, x):
        return self.module(x)
