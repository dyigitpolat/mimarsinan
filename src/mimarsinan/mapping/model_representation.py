"""
ModelRepresentation: DAG wrapper for the mapper graph with execution order and perceptron enumeration.

Kept in a separate module so consumers that only need the DAG wrapper do not
load all mapper classes from mapping_utils.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn


class ModelRepresentation:
    def __init__(self, output_layer_mapper):
        self.output_layer_mapper = output_layer_mapper
        self.pytorch_module = nn.Identity()
        self._exec_order = None
        self._deps = None

    def map(self, mapping):
        return self.output_layer_mapper.map(mapping)

    def map_to_ir(self, ir_mapping):
        """
        Map this model representation to a unified IR (IRGraph).

        This produces an IRGraph containing both NeuralCore and ComputeOp nodes.
        """
        return self.output_layer_mapper.map_to_ir(ir_mapping)

    def construct_pytorch_module(self, module, next):
        return self.output_layer_mapper.construct_pytorch_module(self.pytorch_module)

    def _ensure_exec_graph(self):
        """
        Build a reusable topological execution order once (postorder: deps first).
        Also reused for perceptron enumeration to guarantee consistent ordering.
        """
        def deps_of(node):
            if hasattr(node, "get_source_mappers"):
                return node.get_source_mappers()
            if hasattr(node, "source_mapper") and node.source_mapper is not None:
                return [node.source_mapper]
            return []

        if self._exec_order is not None and self._deps is not None:
            return

        deps_map = {}
        order = []
        visited = set()
        stack = [(self.output_layer_mapper, False)]

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
            d = deps_of(node)
            deps_map[node] = d
            for dep in reversed(d):
                if dep is not None and dep not in visited:
                    stack.append((dep, False))

        self._exec_order = order
        self._deps = deps_map

    def get_perceptron_groups(self):
        """
        Return perceptron groups in forward-topological order.
        Groups are used by scale/activation analysis steps to propagate scales.
        """
        self._ensure_exec_graph()

        seen = set()
        groups = []

        for node in self._exec_order:
            if not hasattr(node, "owned_perceptron_groups"):
                continue
            for group in node.owned_perceptron_groups():
                unique = []
                for p in group:
                    if id(p) in seen:
                        continue
                    seen.add(id(p))
                    unique.append(p)
                if len(unique) > 0:
                    groups.append(unique)

        return groups

    def get_perceptrons(self):
        """Flattened list of perceptrons in forward-topological order."""
        perceptrons = []
        for group in self.get_perceptron_groups():
            perceptrons.extend(group)
        return perceptrons

    def assign_perceptron_indices(self):
        """
        Set perceptron_index on each mapper that owns perceptrons, in the same
        order as get_perceptrons(), so that map_to_ir can pass pruning provenance
        to the IR (tiled FC and weight banks).
        """
        self._ensure_exec_graph()
        idx = 0
        for node in self._exec_order:
            if not hasattr(node, "owned_perceptron_groups"):
                continue
            node.perceptron_index = idx
            for group in node.owned_perceptron_groups():
                seen = set()
                for p in group:
                    if id(p) not in seen:
                        seen.add(id(p))
                        idx += 1

    def __call__(self, x):
        """Execute the mapper graph as a single source of truth for forward."""
        self._ensure_exec_graph()

        # When cuda_debug is on, sync after every node's forward so an async
        # CUDA assert is attributed to the mapper that triggered it instead of
        # whichever later op happens to hit the first sync point.
        cuda_debug = (
            os.environ.get("MIMARSINAN_CUDA_DEBUG") == "1" and torch.cuda.is_available()
        )

        values = {}
        for node in self._exec_order:
            d = self._deps.get(node, [])
            try:
                if len(d) == 0:
                    values[node] = node.forward(x)
                elif len(d) == 1:
                    values[node] = node.forward(values[d[0]])
                else:
                    values[node] = node.forward(tuple(values[dep] for dep in d))
                if cuda_debug:
                    torch.cuda.synchronize()
            except Exception as exc:
                node_name = getattr(node, "name", None) or type(node).__name__
                raise RuntimeError(
                    f"[ModelRepresentation] forward failed at node "
                    f"{type(node).__name__}(name={node_name!r})"
                ) from exc

        return values[self.output_layer_mapper]
