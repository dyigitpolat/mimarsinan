"""PerceptronMapper: map a Perceptron to IR FC cores (or an encoding ComputeOp); a rank-k input by its positions."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.layout.layout_source_view_ops import stack_source_views
from mimarsinan.mapping.mappers.base import Mapper, resolve_activation_type
from mimarsinan.mapping.mappers.flowchart import FlowchartFCSpec, FlowchartNodeEstimate
from mimarsinan.mapping.mappers.scale_propagation import (
    perceptron_boundary_scale,
    perceptron_per_source_scale,
)
from mimarsinan.transformations.perceptron.bias_drive import bias_row_values
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer


class PerceptronFanInError(ToolchainRefusal, ValueError):
    """A Linear whose per-sample input neither flattens to its fan-in nor has it as an axis of a rank-2 input or as the
    last axis of a higher-rank one: the FC mapping (``map_fc``) has no IR form for it (review round 13)."""

    def __init__(self, node_name: str, shape: tuple, in_features: int):
        self.node_name, self.shape, self.in_features = node_name, shape, in_features
        super().__init__(
            f"Linear (FX node {node_name!r}): its per-sample input of shape {shape} does not flatten to its fan-in "
            f"{in_features}, and the FC mapping lowers a Linear over a rank-1 per-sample input, an axis of a rank-2 "
            "one or the last axis of a higher-rank one only, so the lowering is refused"
        )


def fan_in_refusal(sources: Any, in_features: int, name: Optional[str]) -> Optional[PerceptronFanInError]:
    """The refusal of a Linear's input sources (its per-sample input transposed) that ``map_fc`` cannot lower: neither
    of ``in_features`` entries nor of rank 2 with an axis of that size."""
    shape = tuple(sources.shape)
    if len(shape) == 2 and in_features in shape or len(shape) != 2 and int(np.prod(shape)) == in_features:
        return None
    return PerceptronFanInError(name or "", tuple(reversed(shape)), in_features)


def leading_axes(sources: Any, in_features: int) -> Optional[tuple]:
    """The leading axes of a per-sample input of rank 3 or more whose last axis is the fan-in and which does not
    flatten to it: the Linear is the rank-2 case over their product (``None`` for every other input)."""
    shape = tuple(int(d) for d in sources.shape)
    if len(shape) >= 3 and shape[-1] == in_features and int(np.prod(shape)) != in_features:
        return shape[:-1]
    return None


class PerceptronMapper(Mapper):
    def __init__(self, source_mapper, perceptron):
        super(PerceptronMapper, self).__init__(source_mapper)
        self.perceptron = perceptron

    def _map_to_ir(self, ir_mapping):
        layer_weights = PerceptronTransformer().get_effective_weight(self.perceptron)
        layer_biases = PerceptronTransformer().get_effective_bias(self.perceptron)

        sources = self.require_source_mapper().map_to_ir(ir_mapping)
        lead = leading_axes(sources, int(layer_weights.shape[1]))
        if lead is None:
            return self._map_positions(ir_mapping, sources, layer_weights, layer_biases)
        flat = sources.reshape(int(np.prod(lead)), int(layer_weights.shape[1]))
        mapped = self._map_positions(ir_mapping, flat, layer_weights, layer_biases)
        return mapped.reshape(*lead, int(mapped.shape[-1]))

    def _map_positions(self, ir_mapping, sources, layer_weights, layer_biases):
        """The Linear over a per-sample input of rank 1, or of rank 2 with the fan-in as an axis (one position per
        column)."""
        layer_sources = sources.transpose()

        if getattr(self.perceptron, "is_encoding_layer", False):
            return self._map_to_ir_as_encoding_compute_op(
                ir_mapping, layer_sources, layer_weights, layer_biases
            )

        normalization = getattr(self.perceptron, "normalization", None)
        normalization_type = type(normalization).__name__ if normalization is not None else None
        activation_type = resolve_activation_type(self.perceptron)

        refusal = fan_in_refusal(layer_sources, int(layer_weights.shape[1]), getattr(self.perceptron, "name", None))
        if refusal is not None:
            raise refusal
        output_shape = np.array([layer_weights.shape[0], layer_sources.shape[-1]])
        layer_sources = ir_mapping.map_fc(
            layer_sources,
            output_shape,
            layer_weights,
            # The always-on row's weight: the bias over the row's drive (``transformations.perceptron.bias_drive``).
            bias_row_values(self.perceptron, layer_biases),
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
        size = int(np.prod(out_shape)) if out_shape is not None else out_f
        positions = size // out_f if size >= out_f and size % out_f == 0 else 1  # its output's leading positions
        return FlowchartNodeEstimate(
            sw_text=sw_text,
            fc_spec=FlowchartFCSpec(
                in_features=in_f,
                out_features=out_f,
                instances=positions,
                has_bias=(p.layer.bias is not None),
            ),
        )

