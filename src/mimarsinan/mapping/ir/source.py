"""``IRSource`` and its columnar wire format.

An ``IRSource`` is a pair of ints, so an object array of them pickles at
~27.7 B per axon where two int32 columns cost 8 — and the real vehicle carries
~9.4M of them. Encoding is lossless by construction and refuses anything that
is not a pure ``IRSource`` array, so a mixed array degrades to the object form
instead of silently losing entries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "IRSource", "decode_ir_sources", "encode_ir_sources",
    "is_encoded_ir_sources",
]

_TAG = "ir_sources_v1"


@dataclass
class IRSource:
    """Input source: node output, off (-1), network input (-2), or always-on (-3)."""
    node_id: int
    index: int

    def is_off(self) -> bool:
        return self.node_id == -1

    def is_input(self) -> bool:
        return self.node_id == -2

    def is_always_on(self) -> bool:
        return self.node_id == -3


def encode_ir_sources(array) -> object:
    """``(tag, shape, node_ids, indices)`` — or ``array`` when it is not pure."""
    if array is None:
        return array
    flat = np.asarray(array).flatten()
    if flat.size and not all(isinstance(s, IRSource) for s in flat):
        return array
    node_ids = np.fromiter((int(s.node_id) for s in flat),
                           dtype=np.int32, count=flat.size)
    indices = np.fromiter((int(s.index) for s in flat),
                          dtype=np.int32, count=flat.size)
    return (_TAG, np.asarray(array).shape, node_ids, indices)


def is_encoded_ir_sources(value) -> bool:
    return (
        isinstance(value, tuple) and len(value) == 4 and value[0] == _TAG
    )


def decode_ir_sources(value):
    """Inverse of :func:`encode_ir_sources`; passes through unencoded input."""
    if not is_encoded_ir_sources(value):
        return value
    _tag, shape, node_ids, indices = value
    sources = [
        IRSource(node_id=int(n), index=int(i))
        for n, i in zip(node_ids.tolist(), indices.tolist())
    ]
    return np.array(sources, dtype=object).reshape(shape)
