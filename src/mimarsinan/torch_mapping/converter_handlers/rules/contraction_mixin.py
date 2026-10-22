"""The contraction rule: every product of two activations is one host op whose payload is a ``Contraction``."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch
import torch.fx as fx

from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.mapping.support.contraction import Contraction, contraction_macs, matmul_spec
from mimarsinan.torch_mapping.converter_handlers.converter_contract import ConverterContract
from mimarsinan.torch_mapping.fx_shape_utils import node_output_shape, strip_batch


def _equation(node: fx.Node) -> Optional[str]:
    found = node.args[0] if node.args else node.kwargs.get("equation")
    return found if isinstance(found, str) else None


def product_spec(node: fx.Node, fn: Any, shapes: Sequence[Tuple[int, ...]]) -> Optional[str]:
    """The einsum spec of a product node at its traced operand shapes; ``None`` for no product."""
    if fn is torch.einsum:
        return _equation(node)
    if fn in (torch.bmm, torch.Tensor.bmm):
        return "bmk,bkn->bmn"
    return matmul_spec(len(shapes[0]), len(shapes[1]))


class ContractionMixin(ConverterContract):
    """``torch.matmul``, ``@``, ``torch.bmm``, two-operand ``torch.einsum`` and the ``.matmul``/``.bmm`` methods of two
    activations: the host op evaluates the traced call itself and declares its interface (``Contraction``). A product
    with a parameter or buffer operand keeps the generic path (a bound tensor, N6 later)."""

    def _convert_product(self, node: fx.Node, fn: Any) -> None:
        self._native_operands(node)
        partition = self._partition_fx_args(node, fn)
        operands = [arg for _mapper, arg in partition.sources]
        traced = [shape for shape in map(node_output_shape, operands) if shape is not None]
        out = node_output_shape(node)
        two = len(operands) == 2 and len(traced) == 2 and not partition.bound and not partition.unresolved
        spec = product_spec(node, fn, traced) if two and out is not None else None
        if spec is None or out is None:
            self._emit_generic_compute_op(node, fn)
            return
        per_sample = [tuple(shape[1:]) for shape in traced]
        payload = Contraction(fn, spec, per_sample, contraction_macs(spec, traced, out),
                              extra_args=partition.literals, kwargs=partition.literal_kwargs,
                              arg_template=partition.template)
        self._node_to_mapper[node] = ComputeOpMapper(
            [mapper for mapper, _arg in partition.sources], payload, input_shapes=per_sample,
            output_shape=strip_batch(out), name=node.name, elementwise_inputs=False,
        )
