"""Base Mapper class for the mapper DAG (forward pass and hardware mapping)."""

from __future__ import annotations

import torch.nn as nn

from mimarsinan.mapping.mappers.flowchart import FlowchartNodeEstimate
from mimarsinan.mapping.mappers.scale_propagation import (
    first_source_scale,
    present_source_scales,
)


def resolve_activation_type(perceptron) -> str | None:
    """Extract the activation_type string from a perceptron for IR metadata.

    Handles plain activations and TransformedActivation wrappers, e.g. "Identity + ClampDecorator".
    """
    activation = getattr(perceptron, "activation", None)
    if activation is None:
        return None
    activation_type = type(activation).__name__
    if hasattr(activation, "base_activation") and hasattr(activation, "decorators"):
        base = getattr(activation, "base_activation", None)
        base_name = type(base).__name__ if base is not None else "Activation"
        decorators = getattr(activation, "decorators", []) or []
        decorator_names = [type(d).__name__ for d in decorators]
        activation_type = (
            f"{base_name} + {', '.join(decorator_names)}" if decorator_names else base_name
        )
    return activation_type


class Mapper(nn.Module):
    def __init__(self, source_mapper: "Mapper | None" = None):
        super(Mapper, self).__init__()
        self._source_mapper_container: list[Mapper | None] = [source_mapper]
        self._ir_sources = None
        self._cached_ir_mapping = None

    def get_source_mappers(self):
        """Return the list of source mappers this node depends on."""
        if self.source_mapper is not None:
            return [self.source_mapper]
        return []

    def owned_perceptron_groups(self):
        """Introspection hook for Perceptron-first pipelines. Default: no perceptrons."""
        return []

    def propagate_source_scale(self, deps, out_scales):
        """Per-node out-scale for weight-quant per-source scales; default routes the first source's out-scale through."""
        return first_source_scale(deps, out_scales)

    def propagate_boundary_scale(self, deps, out_scales, default):
        """Per-node out-scale for TTFS theta-out boundary scales; default is the mean of present source out-scales."""
        present = present_source_scales(deps, out_scales)
        if present:
            return sum(present) / len(present)
        return None

    def flowchart_node_estimate(self, out_shape):
        """Software summary + optional FC estimate spec for the softcore flowchart; default is neither."""
        return FlowchartNodeEstimate()

    @property
    def source_mapper(self) -> "Mapper | None":
        return self._source_mapper_container[0]

    def require_source_mapper(self) -> "Mapper":
        """Return the source mapper; raise if this node was built without one."""
        source = self._source_mapper_container[0]
        if source is None:
            raise ValueError(
                f"{type(self).__name__} requires a source mapper but none was set"
            )
        return source

    def clear_cache(self):
        self._ir_sources = None
        self._cached_ir_mapping = None

    def map_to_ir(self, ir_mapping):
        if self._ir_sources is not None and self._cached_ir_mapping is ir_mapping:
            return self._ir_sources
        self._ir_sources = self._map_to_ir(ir_mapping)
        self._cached_ir_mapping = ir_mapping
        return self._ir_sources

    def _map_to_ir(self, ir_mapping):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _map_to_ir. "
            "Override this method to support unified IR mapping."
        )

    def forward(self, x):
        # Do not cache here: tensor ids and CUDA data pointers are recycled across runs, so an id(x)-keyed cache returns stale outputs.
        return self._forward_impl(x)

    def _forward_impl(self, x):
        raise NotImplementedError
