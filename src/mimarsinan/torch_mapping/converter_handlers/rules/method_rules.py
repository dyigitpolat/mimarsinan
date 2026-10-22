"""The tensor-method rule's predicates: which methods it converts (a torch function of the same name) or refuses by name.

A method with one tensor output is the generic host op of ``torch.Tensor.<method>`` (bit for bit the method), a product
method (``matmul``, ``bmm``) the contraction rule; one with a tuple output (``split``, ``unbind``, a ``chunk`` the
rewrite leaves) is refused by name (``TupleMethodError``, N6). A method with no torch function of its name (``float``,
``to``), an in-place method and one whose value is not a tensor (``numel``) have no rule.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.fx as fx

from mimarsinan.common.refusal import ToolchainRefusal


def torch_function_of(method: str) -> Optional[Any]:
    """The torch function a tensor method shares its name with, when both exist (``softmax``, ``sum``, ``narrow``)."""
    function = getattr(torch, method, None)
    if method.startswith("_") or method.endswith("_") or not callable(function) or isinstance(function, type) \
            or not callable(getattr(torch.Tensor, method, None)):
        return None
    return function


def traced_tensor(node: fx.Node) -> bool:
    return hasattr(node.meta.get("tensor_meta"), "shape")


def _traced_tuple(node: fx.Node) -> bool:
    return isinstance(node.meta.get("tensor_meta"), (tuple, list))


def has_method_rule(node: fx.Node) -> bool:
    """A tensor method the method rule converts or refuses by name: a torch function of its name, and a traced tensor
    or tuple value (the analyzer's test for a conversion rule)."""
    return torch_function_of(str(node.target)) is not None and (traced_tensor(node) or _traced_tuple(node))


class TupleMethodError(ToolchainRefusal, RuntimeError):
    """A tensor method whose traced output is a tuple (``split``, ``unbind``, a ``chunk`` with a user other than an
    integer ``getitem``); static-linear lowerings of these are work item N6."""

    def __init__(self, node: fx.Node, reason: str):
        self.node_name, self.reason = node.name, reason
        super().__init__(f"{node.target} (FX node {node.name!r}): {reason}, so the conversion is refused")
