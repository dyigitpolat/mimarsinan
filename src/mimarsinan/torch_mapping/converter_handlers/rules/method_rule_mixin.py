"""The tensor-method rule: a method with a torch function of its name converts as the host op of the method itself
(``method_rules.py``: one tensor output, else ``TupleMethodError``); a product method is a contraction."""

from __future__ import annotations

import torch
import torch.fx as fx

from mimarsinan.mapping.support.contraction import PRODUCT_FUNCTIONS
from mimarsinan.torch_mapping.converter_handlers.rules.contraction_mixin import ContractionMixin
from mimarsinan.torch_mapping.converter_handlers.rules.method_rules import (
    TupleMethodError,
    has_method_rule,
    traced_tensor,
)


class MethodRuleMixin(ContractionMixin):
    """The method rule for every tensor method outside the dedicated layout, shape and arithmetic rules."""

    def _convert_method(self, node: fx.Node) -> bool:
        """Convert ``node`` by the method rule; False when the method has none."""
        method = str(node.target)
        if not has_method_rule(node):
            return False
        native = getattr(torch.Tensor, method)
        if not traced_tensor(node):
            raise TupleMethodError(node, f"the method .{method}(...) returns a tuple, not one tensor")
        if native in PRODUCT_FUNCTIONS:
            self._convert_product(node, native)
        else:
            self._emit_generic_compute_op(node, native)
        return True
