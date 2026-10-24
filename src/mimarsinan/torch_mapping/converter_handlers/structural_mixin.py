"""Structural shortcuts (cat / flatten) and BN / activation absorption helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.mapping.support.call_template import ArgTemplate, BoundSlot, SourceSlot
from mimarsinan.mapping.support.compute_modules import ComputeAdapter, _cat_along, cat_per_sample
from mimarsinan.mapping.mapping_utils import (
    ComputeOpMapper,
    ConcatMapper,
    ReshapeMapper,
)
from mimarsinan.torch_mapping.converter_handlers.batchnorm_lowering import perceptron_normalization
from mimarsinan.torch_mapping.converter_handlers.converter_contract import ConverterContract
from mimarsinan.torch_mapping.converter_handlers.refusals import (
    ConcatenationOperandError,
    GenericOpArgumentError,
)
from mimarsinan.torch_mapping.fx_shape_utils import (
    fx_literal_int,
    node_output_shape,
    node_target_str,
    strip_batch,
)

if TYPE_CHECKING:
    from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityReport


class StructuralConvertMixin(ConverterContract):
    def _convert_flatten_func(self, node: fx.Node) -> None:
        self._native_static_operand(node)
        self._require_batch_leading(node)
        self._require_restored(node)
        if not self._native_layout(node, torch.flatten):
            self._node_to_mapper[node] = self._flat_mapper(node, self._get_source_mapper(node))

    def _convert_flatten_module(self, node: fx.Node, source) -> None:
        self._require_batch_leading(node)
        self._require_restored(node)
        module = self._modules.get(node_target_str(node))
        if module is None or not self._native_layout(node, module):
            self._node_to_mapper[node] = self._flat_mapper(node, source)

    def _flat_mapper(self, node: fx.Node, source):
        out_shape = self._get_output_shape(node)
        return ReshapeMapper(source, (out_shape[-1],)) if out_shape is not None and len(out_shape) == 2 else source

    def _convert_cat(self, node: fx.Node) -> None:
        tensors_arg = node.args[0] if node.args else node.kwargs.get("tensors")
        dim_arg = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if isinstance(dim_arg, fx.Node) and self._native_layout(node, torch.cat):
            return  # an impure program's run-time dim: the host op of torch.cat on its run-time operands
        if not isinstance(tensors_arg, (list, tuple)):
            self._node_to_mapper[node] = self._get_source_mapper(node)
            return
        self._native_operands(node, [arg for arg in tensors_arg if isinstance(arg, fx.Node)])  # review round 15
        dim = self._cat_dim(node, node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0))
        traced = node_output_shape(node)
        if (dim % len(traced) if traced else dim) == 0:
            raise ConcatenationOperandError(
                node, "concatenation along the batch axis (dim 0) mixes samples, which a per-sample program "
                "cannot express",
            )
        operands = [self._cat_operand(node, arg) for arg in tensors_arg]
        sources = [value for is_bound, value in operands if not is_bound]
        bound = [value for is_bound, value in operands if is_bound]
        if not sources:
            raise ConcatenationOperandError(
                node, "no operand is an activation (every one is a parameter or buffer), so the per-sample "
                "program has no source to carry the concatenation",
            )
        if not bound:
            violation = self._layout_violation(node)
            if violation is not None:
                raise ConcatenationOperandError(node, f"along dim {dim}: in its output, {violation}")
            rank = len(traced) if traced else None
            self._node_to_mapper[node] = (
                sources[0] if len(sources) == 1 else ConcatMapper(sources, dim=dim, name=node.name, rank=rank)
            )
            return
        if [is_bound for is_bound, _ in operands] == [True, False]:
            # One bound prefix (a class token): the pre-template ``_cat_along`` form, byte for byte.
            self._node_to_mapper[node] = ComputeOpMapper(
                sources[0], ComputeAdapter(_cat_along, bound_tensors=bound, kwargs={"dim": dim}), name=node.name,
            )
            return
        slots, counts = [], [0, 0]
        for is_bound, _ in operands:
            slots.append(BoundSlot(counts[1]) if is_bound else SourceSlot(counts[0]))
            counts[int(is_bound)] += 1
        adapter = ComputeAdapter(
            cat_per_sample, bound_tensors=bound, kwargs={"dim": dim}, arg_template=ArgTemplate((slots,)),
        )
        shapes = [strip_batch(node_output_shape(arg)) for arg in tensors_arg
                  if isinstance(arg, fx.Node) and self._get_mapper(arg) is not None]
        self._node_to_mapper[node] = ComputeOpMapper(
            sources, adapter, input_shapes=shapes[0] if len(shapes) == 1 else shapes,
            output_shape=strip_batch(traced), name=node.name, elementwise_inputs=False,
        )

    def _cat_dim(self, node: fx.Node, dim) -> int:
        """The concatenation axis: a literal, or a static shape value (``a.dim() - 1``) known at conversion time."""
        return fx_literal_int(self._static_dim(node, dim, ConcatenationOperandError))

    def _cat_operand(self, node: fx.Node, arg) -> tuple[bool, Any]:
        """``(False, mapper)`` for an activation, ``(True, per-sample tensor)`` for a parameter or buffer operand."""
        per_sample = self._per_sample_constant(arg) if isinstance(arg, fx.Node) else None
        if per_sample is not None:
            return True, per_sample
        mapper = self._get_mapper(arg) if isinstance(arg, fx.Node) else None
        if mapper is None:
            if not isinstance(arg, fx.Node):
                raise ConcatenationOperandError(node, f"operand {arg!r} is not a traced value")
            raise GenericOpArgumentError.for_node(torch.cat, node, arg)
        return False, mapper

    def _per_sample_constant(self, arg: fx.Node):
        """A parameter or buffer operand (raw or ``expand``ed per batch) as one sample's slice; a batch-free value
        (``symbolic/batch_free.py``) so as the module computing that slice at every call; None otherwise (an
        ``expand`` converted to a host op of its own, an impure program's, is an activation)."""
        free, traced = self._batch_free_source(arg), node_output_shape(arg)
        if free is not None and traced:
            try:
                self._batch_free.evaluate(free).broadcast_to((1, *traced[1:]))
            except RuntimeError:
                return None
            return self._batch_free.value(free, traced[1:])
        tensor = self._get_constant_tensor(arg)
        traced = tuple(tensor.shape) if tensor is not None else None
        if tensor is None and self._node_to_mapper.get(arg) is None:
            tensor = self._get_expanded_constant_tensor(arg)
            traced = node_output_shape(arg)
        if tensor is None or not traced:
            return None
        try:
            return tensor.detach().broadcast_to((1, *traced[1:]))[0].clone()
        except RuntimeError:
            return None

    def _batch_free_source(self, arg: fx.Node):
        """The batch-free value a concatenation operand is: itself, or the receiver of its ``expand`` (no mapper)."""
        if arg in self._batch_free.nodes:
            return arg
        expanded = arg.op == "call_method" and arg.target == "expand" and self._node_to_mapper.get(arg) is None
        receiver = arg.args[0] if expanded and arg.args else None
        return receiver if isinstance(receiver, fx.Node) and receiver in self._batch_free.nodes else None

    _perceptron_normalization = staticmethod(perceptron_normalization)

    def _propagate_absorbed(self, node: fx.Node) -> None:
        if len(node.args) >= 1 and isinstance(node.args[0], fx.Node):
            self._node_to_mapper[node] = self._get_mapper(node.args[0])

    def _find_absorbed_follower(
        self,
        node: fx.Node,
        target_types: tuple,
        report: RepresentabilityReport,
        skip_bn: bool = False,
    ):
        """Find the first absorbed follower of ``node`` matching ``target_types``.

        Walks through absorbed Identity (and BatchNorm when ``skip_bn=True``)
        to reach the target.
        """
        for user in node.users:
            if user.name in self._absorbed and user.op == "call_module":
                mod = self._modules.get(node_target_str(user))
                if mod is not None and isinstance(mod, target_types):
                    return mod
                if isinstance(mod, nn.Identity):
                    return self._find_absorbed_follower(user, target_types, report, skip_bn)
                if skip_bn and isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    return self._find_absorbed_follower(user, target_types, report, skip_bn)
        return None
