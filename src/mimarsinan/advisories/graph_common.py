"""Shared mapper-graph helpers for the graph advisory rules."""

from __future__ import annotations


def exec_and_deps(model_repr) -> tuple[list, dict]:
    """Execution order + per-node dependency lists via the public graph API."""
    order = model_repr.execution_order()
    consumers = model_repr.consumer_map()
    deps: dict = {node: [] for node in order}
    for node in order:
        for consumer in consumers.get(id(node), []):
            deps[consumer].append(node)
    return order, deps


def name_of(perceptron) -> str:
    return str(getattr(perceptron, "name", None) or "<unnamed>")
