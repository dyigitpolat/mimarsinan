"""The recursive mapper walk, kept ONLY as the tests' reference (E276).

``Mapper.map_to_ir`` walks the mapper DAG with an explicit stack. The walk it
replaced recursed once per layer, so the depth a program could map was bounded
by the interpreter's recursion limit. The tests patch the recursive body back
in to show (a) that a deep chain really needs the iterative walk and (b) that
the iterative walk builds the same IR, node ids and order included.
"""

from __future__ import annotations

import contextlib
import pickle
import sys

from mimarsinan.mapping.mappers.base import Mapper

#: CPython's default recursion limit; the deep-chain tests assert it is in force.
DEFAULT_RECURSION_LIMIT = 1000


def _recursive_map_to_ir(self, ir_mapping):
    """The pre-E276 body of ``Mapper.map_to_ir``: memo check, then recurse."""
    if self._ir_sources is not None and self._cached_ir_mapping is ir_mapping:
        return self._ir_sources
    self._ir_sources = self._map_to_ir(ir_mapping)
    self._cached_ir_mapping = ir_mapping
    return self._ir_sources


@contextlib.contextmanager
def recursive_walk(recursion_limit: int | None = None):
    """Map through the recursive reference walk, optionally under a raised limit."""
    iterative = Mapper.map_to_ir
    previous_limit = sys.getrecursionlimit()
    Mapper.map_to_ir = _recursive_map_to_ir  # type: ignore[method-assign]
    if recursion_limit is not None:
        sys.setrecursionlimit(recursion_limit)
    try:
        yield
    finally:
        Mapper.map_to_ir = iterative  # type: ignore[method-assign]
        sys.setrecursionlimit(previous_limit)


def ir_bytes(ir_graph) -> bytes:
    """The IR as bytes: nodes (ids, order, wiring, weights), outputs, banks.

    ``build_token`` is a per-construction uuid, so it is the one field left out.
    """
    return pickle.dumps(
        (ir_graph.nodes, ir_graph.output_sources, ir_graph.weight_banks),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def layout_bytes(softcores) -> bytes:
    """The shape-only layout's softcore specs as bytes."""
    return pickle.dumps(list(softcores), protocol=pickle.HIGHEST_PROTOCOL)
