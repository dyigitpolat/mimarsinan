"""[U-PCH3] ``ir-tiling``'s reading of a tile's per-neuron values against its unbounded neuron's: each value by the
layout its node declares (``NeuralCore.VALUE_LAYOUTS``, ``mapping.support.value_layouts``), never by the value's size.

- One per output neuron (a threshold register, a hardware bias): at each side's paired columns.
- One per output channel of the perceptron (an activation scale, held whole by every node of the perceptron, and the
  node's entries in the program's boundary maps): at each side's paired columns' channels, the check's own channel
  identity (``keys.perceptron_channels``), derived only when a vector is read.
- One per input row (an entry currency): as each side's paired neurons read it, the set of (source code, entry) over
  the node's kept rows that carry a weight in the paired columns, the fan-in's own row identity.
- Whole: as it is.

A value of a size its declaration does not admit is a violation by name (the refusal's text), never a value read in
another layout."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from mimarsinan.certification.program_check.tiles import _same_value, _values_equal
from mimarsinan.certification.program_check.unbounded.keys import KeyedIR, perceptron_channels
from mimarsinan.mapping.support.value_layouts import (
    ValueLayout, ValueLayoutError, declared_layout, neuron_entries, one_entry, row_entries,
)

Side = Tuple[KeyedIR, Any, np.ndarray, int]   # the keyed IR, the node, its paired columns, its width


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return _values_equal(a, b)
    return _same_value(a, b)


def _entry(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(np.asarray(value.detach().cpu() if hasattr(value, "detach") else value).reshape(-1)[0])


class NeuronPair:
    """A tile's paired columns and its unbounded node's, read value by value."""

    def __init__(self, tile: Side, reference: Side) -> None:
        self.sides = (tile, reference)
        self._channels: Dict[int, Optional[np.ndarray]] = {}

    def channels(self, i: int):
        def derive() -> Optional[np.ndarray]:
            if i not in self._channels:
                _k, node, _cols, width = self.sides[i]
                self._channels[i] = perceptron_channels(node, width)
            return self._channels[i]
        return derive

    def rows_read(self, name: str, values: Tuple[Any, Any]) -> Tuple[Any, ...]:
        if all(v is None or one_entry(v) for v in values):
            return tuple(_entry(v) for v in values)
        out = []
        for (k, node, cols, _width), value in zip(self.sides, values):
            rows = k.rows[int(node.id)]
            entries = row_entries(node, name, rows, value=value)
            live = np.any(np.asarray(node.get_core_matrix(k.graph))[rows][:, cols] != 0, axis=1)
            sources = k.node_sources(node)
            codes = k.source_codes([sources[r] for r in rows[live].tolist()])
            out.append(None if entries is None else tuple(sorted(set(zip(codes.tolist(), entries[live].tolist())))))
        return tuple(out)

    def differs(self, name: str, a: Any, b: Any) -> Optional[str]:
        """None where both sides' paired neurons read value ``name`` alike (``a`` the tile's, ``b`` the unbounded
        node's), else "" (they differ), or the refusal's text where a side holds it in no declared layout."""
        try:
            if declared_layout(self.sides[0][1], name) is ValueLayout.PER_INPUT_ROW:
                read = self.rows_read(name, (a, b))
            else:
                read = tuple(neuron_entries(node, name, cols, width=width, channels=self.channels(i), value=v)
                             for i, ((_k, node, cols, width), v) in enumerate(zip(self.sides, (a, b))))
        except ValueLayoutError as refused:
            return str(refused)
        return None if _same(*read) else ""
