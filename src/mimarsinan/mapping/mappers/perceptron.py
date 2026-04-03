"""PerceptronMapper, ModuleComputeMapper, and ModuleMapper."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.base import Mapper, resolve_activation_type
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

        # Encoding as a single host ComputeOp only when this FC applies to one column of
        # sources (plain MLP). Multi-column (e.g. Mixer token grid) must use map_fc tiling.
        if getattr(self.perceptron, "is_encoding_layer", False):
            if layer_sources.ndim == 1 or (
                layer_sources.ndim == 2 and layer_sources.shape[1] == 1
            ):
                return self._map_to_ir_as_encoding_compute_op(
                    ir_mapping, layer_sources, layer_weights, layer_biases
                )

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

    def _map_to_ir_as_encoding_compute_op(self, ir_mapping, layer_sources, layer_weights, _layer_biases):
        """Host-side full ``Perceptron`` forward as a single ``ComputeOp(module)``."""
        in_features = int(layer_weights.shape[1])
        out_features = int(layer_weights.shape[0])

        flat_in = np.array(layer_sources, dtype=object).flatten()

        out = ir_mapping.add_compute_op(
            input_sources=flat_in,
            op_type="module",
            params={"module": self.perceptron, "input_shape": (in_features,)},
            input_shape=(in_features,),
            output_shape=(out_features,),
            name=getattr(self.perceptron, "name", None),
        )
        return out.transpose()

    def _forward_impl(self, x):
        return self.perceptron(x)

    def owned_perceptron_groups(self):
        return [[self.perceptron]]


class ModuleComputeMapper(Mapper):
    """Generic host-side ComputeOp for any nn.Module that isn't a perceptron.

    Wraps the original PyTorch module. Forward calls the module directly.
    IR mapping creates a ComputeOp with op_type="module" that stores the
    module for host-side execution during spiking simulation.

    Used for bare Linear (no activation), Conv2d (no activation), MaxPool2d,
    LayerNorm, Dropout, or any other module that isn't packed as a perceptron.
    """

    def __init__(self, source_mapper, module: nn.Module,
                 input_shape=None, output_shape=None, name=None):
        super().__init__(source_mapper)
        self.module = module
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.name = name

    def _forward_impl(self, x):
        return self.module(x)

    def _map_to_ir(self, ir_mapping):
        input_sources = self.source_mapper.map_to_ir(ir_mapping)
        src_arr = np.array(input_sources, dtype=object)

        # Handle 2D source arrays: each column is an independent instance
        # (e.g., token positions in mixer architectures). Create one
        # ComputeOp per column so the module runs independently per instance.
        # Transpose to (features, instances) convention used by map_fc.
        if src_arr.ndim == 2:
            src_arr = src_arr.transpose()
            # Determine input feature count from the module
            in_features = getattr(self.module, 'in_features', None)
            if in_features is None and hasattr(self.module, '0'):
                # nn.Sequential: check first sub-module
                in_features = getattr(self.module[0], 'in_features', None)
            # Align first dim with module input features
            if in_features is not None:
                if src_arr.shape[0] != in_features and src_arr.shape[1] == in_features:
                    src_arr = src_arr.T
            col_count = int(src_arr.shape[1])
            outputs = []
            for i in range(col_count):
                col_sources = np.array(src_arr[:, i], dtype=object).flatten()
                col_out = ir_mapping.add_compute_op(
                    input_sources=col_sources,
                    op_type="module",
                    params={"module": self.module, "input_shape": self.input_shape},
                    input_shape=self.input_shape,
                    output_shape=self.output_shape,
                    name=(f"{self.name}_col{i}" if self.name else None),
                )
                outputs.append(np.array(col_out, dtype=object).flatten())
            result = np.stack(outputs, axis=1)
            return result.transpose()

        return ir_mapping.add_compute_op(
            input_sources=input_sources,
            op_type="module",
            params={"module": self.module, "input_shape": self.input_shape},
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            name=self.name,
        )

    def _map(self, mapping):
        raise NotImplementedError(
            f"{self.name}: ModuleComputeMapper does not support legacy SoftCoreMapping. "
            "Use IRMapping."
        )

    # owned_perceptron_groups() → inherited [] from Mapper base


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
