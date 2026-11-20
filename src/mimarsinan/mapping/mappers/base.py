"""Base Mapper class for the mapper DAG (forward pass and hardware mapping)."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from mimarsinan.mapping.mappers.flowchart import FlowchartNodeEstimate
from mimarsinan.mapping.mappers.scale_propagation import (
    first_source_scale,
    present_source_scales,
)
from mimarsinan.mapping.mappers.slot_unfold import (
    SerialSlotUnfold,
    WholeInputSlotUnfold,
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


def dependency_postorder(root, deps_of, after_of=None) -> tuple[list, dict]:
    """Deps-first postorder of the DAG reachable from ``root``, and the deps read per node.

    An explicit stack, so depth is unbounded. It visits exactly as a recursive
    depth-first walk over ``deps_of`` in its order would (each node once, a dep
    re-reached through a later edge skipped), so a walk that maps nodes in this
    order emits them in the order the recursion did. ``after_of(node)``, when
    given, names ordering edges the walk visits after the node's deps without
    recording them as read (``Mapper.run_after``: mappers a node runs after in
    the forward without reading them).
    """
    deps_map: dict = {}
    order: list = []
    visited: set = set()
    stack = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if node is None:
            continue
        if expanded:
            order.append(node)
            continue
        if node in visited:
            continue
        visited.add(node)
        stack.append((node, True))
        deps = deps_of(node)
        deps_map[node] = deps
        for dep in reversed(deps if after_of is None else [*deps, *after_of(node)]):
            if dep is not None and dep not in visited:
                stack.append((dep, False))
    return order, deps_map


class Mapper(nn.Module):
    # Ordering edges (review round 16, targeted pass): mappers this one runs after in the forward without reading them
    # (a write in place runs after every earlier reader of its storage); a class default for pickled mappers.
    run_after: tuple = ()
    # The node only moves elements of its inputs (a layout gather or reshape: no arithmetic), so each output element
    # keeps its producer's currency (``value_domain_map``'s MIXED domain); a class default for pickled mappers.
    moves_elements_only: bool = False
    # [U-TGAUGE3] The node is a C-order reshape of its one source (element ``k`` of its value is its source's element
    # ``k``), so each element's source index is its own (``map_model_to_ir.reshape_gauge``); a class default for pickles.
    keeps_element_order: bool = False
    # [U-TGAUGE3] The node joins its sources' flattened values, all of one shape, in order (``map_model_to_ir``).
    joins_equal_parts: bool = False

    def __init__(self, source_mapper: "Mapper | None" = None):
        super(Mapper, self).__init__()
        self._source_mapper_container: list[Mapper | None] = [source_mapper]
        self._ir_sources: Any = None
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

    def serial_slot_unfold(self, *, input_shape, n_slots: int) -> SerialSlotUnfold:
        """How this node's upstream cells fill its cores' axon slot tables.

        The NF event-serial twin folds through THIS answer, so a mapper that
        tiles one weight bank across positions must override it (see
        ``Conv2DPerceptronMapper``); the default is the whole input in feature
        order, which is what a fully-connected hop maps to.
        """
        del input_shape
        return WholeInputSlotUnfold(n_slots)

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

    def _mapped_for(self, ir_mapping) -> bool:
        return self._ir_sources is not None and self._cached_ir_mapping is ir_mapping

    def map_to_ir(self, ir_mapping) -> Any:
        """This node's IR sources, every unmapped node it depends on mapped first (E276).

        The walk is iterative: the unmapped sub-DAG is mapped deps-first, so each
        ``_map_to_ir``'s own ``source.map_to_ir`` call is a memo hit and no call
        nests, at any depth. Every ``_map_to_ir`` reads its sources before it
        emits, so the IR equals the recursive walk's, node ids and order included.
        A source that is not a ``Mapper`` with this walk (a duck-typed
        ``map_to_ir``) is left to its reader, as the recursive walk left it.
        """
        if not self._mapped_for(ir_mapping):
            order, _ = dependency_postorder(
                self, lambda n: [] if n._mapped_for(ir_mapping) else [s for s in n.get_source_mappers() if _walked(s)],
            )
            for node in order:
                if not node._mapped_for(ir_mapping):
                    node._ir_sources = node._map_to_ir(ir_mapping)
                    node._cached_ir_mapping = ir_mapping
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


def _walked(source) -> bool:
    return isinstance(source, Mapper) and type(source).map_to_ir is Mapper.map_to_ir
