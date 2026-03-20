"""PerceptronMapper and ModuleMapper: FC/module application mappers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.base import Mapper, resolve_activation_type, is_perceptron_activation
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

        layer_sources = self.source_mapper.map_to_ir(ir_mapping)
        layer_sources = layer_sources.transpose()

        if not is_perceptron_activation(self.perceptron):
            # No nonlinearity (Identity) → host-side linear ComputeOp.
            w_np = layer_weights.detach().cpu().numpy()
            b_np = layer_biases.detach().cpu().numpy() if layer_biases is not None else None
            name = getattr(self.perceptron, "name", None)
            in_features = w_np.shape[1]

            src_arr = np.array(layer_sources, dtype=object)

            # Handle 2D batch inputs: one ComputeOp per column
            # (e.g., one per token position in mixer architectures).
            if src_arr.ndim == 2:
                if src_arr.shape[0] != in_features and src_arr.shape[1] == in_features:
                    src_arr = src_arr.T
                core_count = int(src_arr.shape[1])
                outputs = []
                for i in range(core_count):
                    col_sources = np.array(src_arr[:, i], dtype=object).flatten()
                    col_out = ir_mapping.add_linear_compute_op(
                        input_sources=col_sources,
                        weights=w_np, biases=b_np,
                        name=(f"{name}_col{i}" if name else None),
                    )
                    outputs.append(col_out.flatten())
                result = np.stack(outputs, axis=1)  # (out_features, core_count)
                return result.transpose()

            output_sources = ir_mapping.add_linear_compute_op(
                input_sources=src_arr, weights=w_np, biases=b_np, name=name,
            )
            return output_sources

        # Has a real activation → NeuralCore.
        normalization = getattr(self.perceptron, "normalization", None)
        normalization_type = type(normalization).__name__ if normalization is not None else None
        activation_type = resolve_activation_type(self.perceptron)

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
        if not is_perceptron_activation(self.perceptron):
            return []
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
