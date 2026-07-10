"""PerceptronMapper: map a Perceptron to IR FC cores (or an encoding ComputeOp)."""

from __future__ import annotations

import numpy as np

from mimarsinan.mapping.layout.layout_source_view_ops import stack_source_views
from mimarsinan.mapping.mappers.base import Mapper, resolve_activation_type
from mimarsinan.mapping.mappers.flowchart import FlowchartFCSpec, FlowchartNodeEstimate
from mimarsinan.mapping.mappers.scale_propagation import (
    perceptron_boundary_scale,
    perceptron_per_source_scale,
)
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer


class PerceptronMapper(Mapper):
    def __init__(self, source_mapper, perceptron):
        super(PerceptronMapper, self).__init__(source_mapper)
        self.perceptron = perceptron

    def _map_to_ir(self, ir_mapping):
        layer_weights = PerceptronTransformer().get_effective_weight(self.perceptron)
        layer_biases = PerceptronTransformer().get_effective_bias(self.perceptron)

        layer_sources = self.require_source_mapper().map_to_ir(ir_mapping)
        layer_sources = layer_sources.transpose()

        if getattr(self.perceptron, "is_encoding_layer", False):
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
            bias_scale=getattr(self.perceptron, "bias_scale", None),
            name=getattr(self.perceptron, "name", None),
            normalization_type=normalization_type,
            activation_type=activation_type,
            perceptron_index=getattr(self, "perceptron_index", None),
        )

        return layer_sources.transpose()

    def _map_to_ir_as_encoding_compute_op(self, ir_mapping, layer_sources, layer_weights, _layer_biases):
        in_features = int(layer_weights.shape[1])
        out_features = int(layer_weights.shape[0])

        src_arr = layer_sources
        if src_arr.ndim == 2 and src_arr.shape[1] > 1:
            num_instances = int(src_arr.shape[1])
            outputs = []
            base_name = getattr(self.perceptron, "name", None)
            for i in range(num_instances):
                col_sources = src_arr[:, i]
                if hasattr(col_sources, "flatten"):
                    col_sources = col_sources.flatten()
                col_out = ir_mapping.add_compute_op(
                    input_sources=col_sources,
                    op_type=type(self.perceptron).__name__,
                    params={"module": self.perceptron, "input_shape": (in_features,)},
                    input_shape=(in_features,),
                    output_shape=(out_features,),
                    name=(f"{base_name}_col{i}" if base_name else None),
                )
                outputs.append(col_out.flatten() if hasattr(col_out, "flatten") else col_out)
            result = stack_source_views(outputs, axis=1)
            return result.transpose()

        flat_in = src_arr.flatten()
        out = ir_mapping.add_compute_op(
            input_sources=flat_in,
            op_type=type(self.perceptron).__name__,
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

    def propagate_source_scale(self, deps, out_scales):
        return perceptron_per_source_scale(self, deps, out_scales)

    def propagate_boundary_scale(self, deps, out_scales, default):
        return perceptron_boundary_scale(self, deps, out_scales, default)

    def flowchart_node_estimate(self, out_shape):
        p = self.perceptron
        in_f = int(p.layer.weight.shape[1])
        out_f = int(p.layer.weight.shape[0])
        sw_text = f"SW perceptrons=1 (in_features={in_f}, out_features={out_f})"
        return FlowchartNodeEstimate(
            sw_text=sw_text,
            fc_spec=FlowchartFCSpec(
                in_features=in_f,
                out_features=out_f,
                instances=1,
                has_bias=(p.layer.bias is not None),
            ),
        )

