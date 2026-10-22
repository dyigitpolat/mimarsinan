"""``chunk`` to ``narrow``: a chunk whose every user is an integer ``getitem`` becomes one ``narrow`` per user.

``x.chunk(n, dim)`` and ``torch.chunk(x, n, dim)`` split the extent ``e`` of axis ``dim`` into pieces of
``ceil(e / n)``, the last one shorter; piece ``i`` is ``x.narrow(dim, i * size, length_i)``, the same view of ``x``
(offset, strides and storage), so every user reads the value it read natively. The pieces are literals, so the rewrite
holds only where ``e`` is the same at every batch: the axis's extent must be a static extent the batch domain proves
(``rewrite_support.StaticExtents``: ``BatchLayouts.layout``, a proof of a pure program that every seeded trace agrees
with), never the traced extent alone. Axis 0 of an activation carries the batch, and so may any other axis (``torch.stack(..., 0)``, ``movedim``,
``swapaxes``, ``t``): a chunk there is ``ceil(B / n)`` samples, which the batch-1 trace would freeze at one (exact up
to batch ``n``, wrong above). An impure program proves no extent (``symbolic/purity.py``). Such a chunk stays as
traced and refuses by name (the method rule's ``TupleMethodError``), or keeps the native host op of ``torch.chunk``.
Each ``getitem`` node keeps its name and metadata and becomes the ``narrow`` (a single-output torch function,
converted as a generic host op); the chunk is erased. ShuffleNetV2's channel split converts this way.
"""

from __future__ import annotations

import math
import operator
from typing import Optional, Set, Tuple, cast

import torch
import torch.fx as fx

from mimarsinan.torch_mapping.fx_rewrites.rewrite_support import StaticExtents, rewritable


def _chunk_call(node: fx.Node) -> Optional[Tuple[fx.Node, int, int]]:
    """``(input, chunks, dim)`` of a ``chunk`` call with literal integer arguments; ``None`` otherwise."""
    is_method = node.op == "call_method" and node.target == "chunk"
    if not is_method and not (node.op == "call_function" and node.target is torch.chunk):
        return None
    args = list(node.args)
    source = args[0] if args else node.kwargs.get("input")
    chunks = args[1] if len(args) > 1 else node.kwargs.get("chunks")
    dim = args[2] if len(args) > 2 else node.kwargs.get("dim", 0)
    if not isinstance(source, fx.Node) or type(chunks) is not int or chunks < 1 or type(dim) is not int:
        return None
    return source, chunks, dim


def _piece(user: fx.Node, node: fx.Node) -> Optional[int]:
    if user.op != "call_function" or user.target is not operator.getitem or len(user.args) != 2:
        return None
    return user.args[1] if user.args[0] is node and type(user.args[1]) is int else None


def split_chunks(gm: fx.GraphModule, ordered: Set[str]) -> bool:
    """Rewrite every eligible chunk of ``gm``; True when one was rewritten."""
    changed = False
    extents = StaticExtents(gm)
    for node in list(gm.graph.nodes):
        call = _chunk_call(node)
        shape = getattr(call[0].meta.get("tensor_meta"), "shape", None) if call is not None else None
        users = list(node.users)
        if call is None or shape is None or not users or not rewritable([node, *users], ordered):
            continue
        source, chunks, dim = call
        dim = dim % len(shape)
        pieces = [_piece(user, node) for user in users]
        extent = int(shape[dim])
        if any(piece is None for piece in pieces) or extent == 0:
            continue
        if not extents.holds(source, dim, extent):
            continue
        size = math.ceil(extent / chunks)
        count = math.ceil(extent / size)
        for user, piece in zip(users, pieces):
            index = cast(int, piece) % count
            user.target = torch.narrow
            user.args, user.kwargs = (source, dim, index * size, min(size, extent - index * size)), {}
        gm.graph.erase_node(node)
        changed = True
    return changed
