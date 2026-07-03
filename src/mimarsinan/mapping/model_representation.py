"""ModelRepresentation: DAG wrapper for the mapper graph with execution order and perceptron enumeration."""

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
        self._consumer_count = None
        self._peak_live_values = 0

    def map_to_ir(self, ir_mapping):
        """Map this model representation to a unified IRGraph (NeuralCore + ComputeOp nodes)."""
        return self.output_layer_mapper.map_to_ir(ir_mapping)

    def construct_pytorch_module(self, module, next):
        return self.output_layer_mapper.construct_pytorch_module(self.pytorch_module)

    def _ensure_exec_graph(self):
        """Build a reusable postorder execution order once (deps first); also drives perceptron enumeration for consistent ordering."""
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

        consumer_count = {n: 0 for n in order}
        for n, deps in deps_map.items():
            for dep in deps:
                if dep is not None and dep in consumer_count:
                    consumer_count[dep] += 1
        self._consumer_count = consumer_count

    def get_perceptron_groups(self):
        """Return perceptron groups in forward-topological order (used by scale/activation analysis)."""
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
        """Set perceptron_index on each owning mapper in get_perceptrons() order so map_to_ir can pass pruning provenance to the IR."""
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
        """Execute the mapper graph; frees intermediates once their consumers run."""
        self._ensure_exec_graph()

        cuda_debug = (
            os.environ.get("MIMARSINAN_CUDA_DEBUG") == "1" and torch.cuda.is_available()
        )

        values = {}
        remaining = dict(self._consumer_count)
        output_node = self.output_layer_mapper
        peak = 0

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

            if len(values) > peak:
                peak = len(values)

            for dep in d:
                if dep is None or dep is output_node:
                    continue
                remaining[dep] -= 1
                if remaining[dep] == 0 and dep in values:
                    del values[dep]

        self._peak_live_values = peak
        return values[output_node]
