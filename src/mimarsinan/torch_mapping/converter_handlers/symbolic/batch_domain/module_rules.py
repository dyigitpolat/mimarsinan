"""The batch rules of torch's modules whose output is not their first operand's shape (``kernels.py`` declares which
module has which, and states the lemma): attention, the recurrent modules, ``nn.Transformer`` and ``nn.GLU``.

Every rule reads its call bound to the module's ``forward`` (``binding.py``, review round 16): ``mha(key=kv,
query=q, value=kv)`` reads ``q`` as the query, ``lstm(hx=(h0, c0), input=q)`` reads ``q`` as the input.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.forms import Opaque
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.kernel_rules import Rule, hinted, tensors_of
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.shapes import (
    Call,
    Shape,
    axis_of,
    free,
    quotient,
)

__all__ = ["attention", "glu", "operand", "recurrent"]


def attention(call: Call) -> Optional[list]:
    """``nn.MultiheadAttention``: the output has the query's shape; the weights are (N, L, S), or (N, heads, L, S)
    when not averaged over the heads; unbatched, (L, S) or (heads, L, S)."""
    tensors = tensors_of(call)
    if len(tensors) < 2 or call.out is None:
        return None
    query, key = tensors[0].dims, tensors[1].dims
    found: List[Optional[Tuple[Any, ...]]] = [query]
    for out in call.out[1:]:
        if call.arg(4, "need_weights", True) is False:  # no weights: torch's fast path returns an empty tensor
            found.append(tuple(d for d in out) if all(type(d) is int for d in out) else None)
        elif len(query) == 3:
            first = call.module.batch_first
            n, l, s = (query[0], query[1], key[1]) if first else (query[1], query[0], key[0])
            heads = (hinted(out[1]),) if len(out) == 4 else ()
            found.append((n, *heads, l, s))
        else:
            heads = (hinted(out[0]),) if len(out) == 3 else ()
            found.append((*heads, query[0], key[0]))
    return found


def recurrent(call: Call) -> Optional[list]:
    """``nn.LSTM``, ``nn.GRU``, ``nn.RNN``: the output passes the leading axes through; a state is (layers, N, H)."""
    tensors = tensors_of(call)
    if not tensors or call.out is None or not isinstance(call.arg(0, "input"), Shape) or \
            not free(tensors[0].dims[-1:]):
        return None
    x = tensors[0].dims
    found: List[Optional[Tuple[Any, ...]]] = [x[:-1] + (hinted(call.out[0][-1]),)]
    batch = (x[0] if call.module.batch_first else x[1]) if len(x) == 3 else None
    for out in call.out[1:]:
        found.append((hinted(out[0]), batch, hinted(out[-1])) if batch is not None else tuple(hinted(d) for d in out))
    return found


def operand(index: int, name: str) -> Rule:
    """Rule ``OPERAND``: the shape of the operand at ``index`` (``name``) of the bound call (``nn.Transformer``: its
    output has ``tgt``'s shape, review round 16)."""
    def rule(call: Call) -> Optional[list]:
        found = call.arg(index, name)
        count = len(call.out) if call.out is not None else 1
        return [found.dims] * count if isinstance(found, Shape) else None
    return rule


def glu(call: Call) -> Optional[list]:
    """``nn.GLU`` and ``F.glu``: the input's shape with the extent of axis ``dim`` halved (an exact quotient in the
    domain; torch raises on an odd extent), review round 16."""
    tensors = tensors_of(call)
    dim = call.module.dim if call.module is not None else call.arg(1, "dim", -1)
    if not tensors or type(dim) is not int:
        return None
    dims = tensors[0].dims
    axis = axis_of(dim, len(dims))
    if axis is None:
        return None
    half = quotient(dims[axis], 2)
    halved = half if half is not None else Opaque("half of an extent the domain does not divide by two")
    return [dims[:axis] + (halved,) + dims[axis + 1:]]
