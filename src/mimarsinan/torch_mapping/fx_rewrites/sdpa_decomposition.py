"""``scaled_dot_product_attention`` as its declared operators, under ``decompose_leaf_modules``.

``F.scaled_dot_product_attention(q, k, v)`` with no mask, no dropout, no causal mask and no grouped-query heads is
``softmax((q * s) @ k^T, -1) @ v`` with ``s`` the call's ``scale`` or ``1 / sqrt(E)``, ``E`` the query's last extent
written as a literal, so only where it is a proved static extent (``rewrite_support.StaticExtents``; an impure program
whose call outside the declared set changes ``E`` with the batch keeps the native call): two products of activations
(Contractions at conversion) around a ``softmax``, every extent stated by a declared batch rule (the call itself has
none, so on the base a static mapper reading it refuses by name). A boolean mask computed from activations refuses by
name (``ActivationMaskError``); any other call keeps the host op of its own.
"""

from __future__ import annotations

import inspect
import math
import operator
from typing import Set

import torch
import torch.fx as fx
import torch.nn.functional as F

from mimarsinan.torch_mapping.fx_rewrites.rewrite_support import StaticExtents, rewritable, trace_meta
from mimarsinan.torch_mapping.leaf_decomposition.registry import refuse_activation_mask

_SIGNATURE = inspect.Signature([inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default)
                                for name, default in (("query", inspect.Parameter.empty),
                                                      ("key", inspect.Parameter.empty),
                                                      ("value", inspect.Parameter.empty), ("attn_mask", None),
                                                      ("dropout_p", 0.0), ("is_causal", False), ("scale", None),
                                                      ("enable_gqa", False))])


def _call(gm: fx.GraphModule, at: fx.Node, op: str, target, *args, **kwargs) -> fx.Node:
    with gm.graph.inserting_before(at):
        node = gm.graph.create_node(op, target, args, kwargs)
    trace_meta(node)
    return node


def decompose_sdpa(gm: fx.GraphModule, ordered: Set[str]) -> bool:
    """Rewrite every eligible scaled dot product attention of ``gm``; True when one was rewritten."""
    changed = False
    extents = StaticExtents(gm)
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not F.scaled_dot_product_attention:
            continue
        call = _SIGNATURE.bind(*node.args, **node.kwargs)
        call.apply_defaults()
        found = call.arguments
        refuse_activation_mask("scaled_dot_product_attention", node, "attn_mask", found["attn_mask"])
        q, k, v, scale = found["query"], found["key"], found["value"], found["scale"]
        shape = getattr(q.meta.get("tensor_meta"), "shape", None) if isinstance(q, fx.Node) else None
        plain = found["attn_mask"] is None and found["dropout_p"] == 0.0 and found["is_causal"] is False \
            and found["enable_gqa"] is False and (scale is None or type(scale) is float)
        if not plain or shape is None or not all(isinstance(x, fx.Node) for x in (k, v)) \
                or not rewritable([node], ordered) or (scale is None and not extents.holds(q, -1, int(shape[-1]))):
            continue
        factor = scale if scale is not None else 1.0 / math.sqrt(int(shape[-1]))
        scores = _call(gm, node, "call_function", torch.matmul, _call(gm, node, "call_function", operator.mul, q,
                                                                        factor),
                       _call(gm, node, "call_method", "transpose", k, -2, -1))
        weights = _call(gm, node, "call_function", torch.softmax, scores, -1)
        out = _call(gm, node, "call_function", torch.matmul, weights, v)
        node.replace_all_uses_with(out)
        gm.graph.erase_node(node)
        extents.invalidate()  # the call had no batch rule: what it fed is provable now (a later block's query)
        changed = True
    return changed
