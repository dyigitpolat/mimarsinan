"""[E6, N2] A bound bank's neuron in ``ir-tiling``: its fan-in is the payload, written per sample, so the IR holds a
placeholder (NaN) and no weight to compare. The neuron reads, per kept row, the payload cell (bank row, bank column) of
its binding times the row's source; it is the same function at both geometries when its rows read the same sources at
the same bank rows of the same binding and its column is the same bank column. Each kept live row becomes one virtual
row under a code naming (binding, row source, bank row) in the key space both IRs share (the binding by its tiling-free
identity: its payload write's host-op name, offset, shape, transpose, batch index, step and width; bank ids and node
ids are cost-only), holding in column ``j`` the bank column ``j`` reads, plus one, so the canonical block compares the
cells exactly."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from mimarsinan.certification.program_check.view import OFF, Canonical, canonical


def binding_of(node: Any, graph: Any) -> Optional[Any]:
    bank = getattr(node, "weight_bank_id", None)
    bindings = getattr(graph, "dynamic_bank_bindings", None) or {}
    return None if bank is None else bindings.get(int(bank))


def _binding_key(binding: Any, keyed: Any) -> tuple:
    source = keyed.by_id.get(int(binding.source_node_id))
    return ("payload", str(getattr(source, "name", binding.source_node_id)), int(binding.payload_offset),
            tuple(int(d) for d in binding.payload_shape), bool(binding.transpose),
            tuple(int(i) for i in binding.batch_index), binding.payload_step, int(binding.payload_bits))


def payload_block(keyed: Any, node: Any, rows: np.ndarray) -> Optional[Canonical]:
    """The canonical block of a bound bank's node over its kept ``rows`` (every column before compaction), None for a
    node of static weights."""
    binding = binding_of(node, keyed.graph)
    if binding is None:
        return None
    sources = keyed.node_sources(node)
    codes = keyed.source_codes([sources[r] for r in rows.tolist()])
    key = _binding_key(binding, keyed)
    first = int((getattr(node, "weight_row_slice", None) or (0, 0))[0])
    width = int(node.get_output_count())
    live = [(r, code) for r, code in zip(rows.tolist(), codes.tolist()) if code != int(OFF)]
    if not live:
        return Canonical(np.zeros(0, dtype=np.int64), np.zeros((0, width)))
    virtual = np.asarray([int(keyed.space.owner((*key, int(code), int(r)))) for r, code in live], dtype=np.int64)
    columns = np.tile(np.arange(first + 1, first + 1 + width, dtype=np.float64), (len(live), 1))
    return canonical(virtual, columns, None)
