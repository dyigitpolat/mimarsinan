"""FX graph shape and argument utilities for :class:`MapperGraphConverter`."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.support.call_template import (
    BoundArg,
    CallPartition,
    LiteralArg,
    SourceArg,
    partition_call,
)
from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.torch_mapping.converter_handlers.converter_contract import ConverterContract
from mimarsinan.torch_mapping.converter_handlers.refusals import GenericOpArgumentError, op_name
from mimarsinan.torch_mapping.fx_shape_utils import (
    broadcasts_to,
    node_input_shapes,
    node_output_shape,
    strip_batch,
)

# The smallest batch at which broadcasting a size-1 axis cannot hide a misplaced batch axis.
_BATCH_CONTRACT_PROBE_SIZE = 2


class _RefusedGenericCall:
    """Payload of a generic op with a non-tensor unresolvable argument: raising when run, not when built.

    Such an argument is shape arithmetic, which may only feed a view's dimensions and never run.
    """

    def __init__(self, refusal: GenericOpArgumentError):
        self.refusal_args = (
            refusal.op_name, refusal.node_name, refusal.argument_name, refusal.argument_kind,
        )

    def __call__(self, *args, **kwargs):
        raise GenericOpArgumentError(*self.refusal_args)


class GenericOpBatchAxisError(ToolchainRefusal, RuntimeError):
    """A generic host op with bound tensors does not reproduce its traced output with a leading batch axis."""

    def __init__(self, fn, node: fx.Node, expected, observed, bound_shapes):
        self.op_name = op_name(fn)
        self.node_name = node.name
        super().__init__(
            f"generic op '{self.op_name}' (FX node '{node.name}') with bound tensors "
            f"{bound_shapes} does not keep the batch axis leading: the traced per-sample "
            f"output implies {expected} at batch {_BATCH_CONTRACT_PROBE_SIZE}, the op "
            f"gives {observed}; the converted model would mix samples, so the "
            "conversion is refused"
        )


def _normalize_bound_tensor(t: torch.Tensor) -> torch.Tensor:
    """Strip a leading singleton dim: bound tensors are stored per sample (batch-stripped)."""
    if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] == 1:
        return t.squeeze(0)
    return t


def _probe_input(meta, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    shape = (_BATCH_CONTRACT_PROBE_SIZE, *tuple(meta.shape)[1:])
    if meta.dtype.is_floating_point or meta.dtype.is_complex:
        return torch.randn(shape, dtype=meta.dtype, generator=generator).to(device)
    return torch.zeros(shape, dtype=meta.dtype, device=device)


def _batch_contract_inputs(node: fx.Node, source_nodes, device) -> Optional[tuple]:
    """Seeded batch-2 inputs shaped like the traced sources; ``None`` without shape metadata."""
    metas = [n.meta.get("tensor_meta") for n in source_nodes]
    if any(m is None or not hasattr(m, "shape") or len(m.shape) == 0 for m in metas):
        return None
    generator = torch.Generator().manual_seed(0)
    return tuple(_probe_input(m, device, generator) for m in metas)


def _observed_shape(adapter: ComputeAdapter, inputs: tuple) -> Any:
    try:
        with torch.no_grad():
            out = adapter(*inputs)
    except (RuntimeError, TypeError, ValueError, IndexError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return tuple(out.shape) if isinstance(out, torch.Tensor) else type(out).__name__


def _non_floating_dtype(node: fx.Node) -> Optional[torch.dtype]:
    """The traced dtype of a non-floating tensor value (a boolean mask, an index), else ``None``."""
    dtype = getattr(node.meta.get("tensor_meta"), "dtype", None)
    return dtype if isinstance(dtype, torch.dtype) and not (dtype.is_floating_point or dtype.is_complex) else None


def _refusing_adapter(fn, node: fx.Node, unresolved) -> ComputeAdapter:
    """Refuse now when an unresolvable argument is a tensor; otherwise refuse if the op ever runs."""
    tensors = [arg for arg in unresolved if arg.meta.get("tensor_meta") is not None]
    if tensors:
        raise GenericOpArgumentError.for_node(fn, node, tensors[0])
    return ComputeAdapter(_RefusedGenericCall(GenericOpArgumentError.for_node(fn, node, unresolved[0])))


class MapperGraphFxMixin(ConverterContract):
    """Shape metadata and generic compute-op emission for mapper conversion."""

    @staticmethod
    def _getitem_looks_like_real_slice(node: fx.Node) -> bool:
        if len(node.args) < 2:
            return False
        index = node.args[1]
        if isinstance(index, tuple):
            return any(
                isinstance(part, slice) or part is Ellipsis for part in index
            )
        return False

    def _partition_fx_args(self, node: fx.Node, fn=None) -> CallPartition:
        """Sources (mapper, FX node), stripped bound tensors and literals of ``node``, in traced order."""

        def classify(arg: fx.Node):
            static, value = self._static_shape_value(arg)
            if static:
                return LiteralArg(value)
            const = self._get_constant_tensor(arg)
            if const is not None:
                return BoundArg(_normalize_bound_tensor(const), tuple(const.shape))
            if arg in self._batch_free.nodes:
                return self._batch_free_operand(arg)
            mapper = self._get_mapper(arg)
            return None if mapper is None else SourceArg((mapper, arg))

        return partition_call(
            node.target if fn is None else fn, node.args, node.kwargs, classify,
        )

    def _batch_free_operand(self, arg: fx.Node) -> BoundArg:
        """A batch-free value (``symbolic/batch_free.py``) as a bound operand computed at every call, stored as a bound
        tensor is (``_normalize_bound_tensor``)."""
        value = self._batch_free.evaluate(arg)
        stored = _normalize_bound_tensor(value)
        return BoundArg(stored, tuple(value.shape), self._batch_free.value(arg, tuple(stored.shape)))

    def _emit_generic_compute_op(self, node: fx.Node, fn) -> None:
        self._native_operands(node)
        partition = self._partition_fx_args(node, fn)
        if not partition.sources:
            anchor = None if partition.unresolved else self._literal_anchor(node) or self._impure_anchor()
            anchored = None if anchor is None else self._anchored_compute_op(node, fn, partition, anchor)
            self._node_to_mapper[node] = anchored
            return
        source_nodes = [arg for _, arg in partition.sources]
        if partition.unresolved:
            adapter = _refusing_adapter(fn, node, partition.unresolved)
        else:
            adapter = self._batch_contract_adapter(node, fn, partition, source_nodes)
        source_shapes = [strip_batch(node_output_shape(arg)) for arg in source_nodes]
        output_shape = strip_batch(node_output_shape(node))
        if len(source_nodes) == 1:
            input_shapes_arg: Any = source_shapes[0]
        else:
            input_shapes_arg = source_shapes
        dtypes = tuple(_non_floating_dtype(arg) for arg in source_nodes)
        if any(dtype is not None for dtype in dtypes):
            adapter.operand_dtypes = dtypes
        self._node_to_mapper[node] = ComputeOpMapper(
            [mapper for mapper, _ in partition.sources],
            adapter,
            input_shapes=input_shapes_arg,
            output_shape=output_shape,
            name=node.name,
            elementwise_inputs=broadcasts_to(source_nodes, node),
        )

    def _literal_anchor(self, node: fx.Node) -> Optional[fx.Node]:
        """The value a call's static shape arguments were read from: the nearest node with a mapper upstream of them
        through static shape values only; ``None`` when the call reads no shape (a factory of plain literals). Each
        node is visited once (breadth first), so shared shape arithmetic costs its node count, not its path count."""
        queue = [arg for arg in node.all_input_nodes if self._static_shape_value(arg)[0]]
        seen = set(queue)
        for current in queue:
            for arg in current.all_input_nodes:
                if arg in seen:
                    continue
                seen.add(arg)
                if self._static_shape_value(arg)[0]:
                    queue.append(arg)
                elif self._node_to_mapper.get(arg) is not None:
                    return arg
        return None

    def _impure_anchor(self) -> Optional[fx.Node]:
        """An impure program's call that reads no activation and no static shape (a parameter's ``shape``, a factory
        of literals) is anchored on the input: nothing of an impure program is computed at conversion."""
        if self._batch_layouts.impurity is None:
            return None
        return next((n for n in self._node_to_mapper if n.op == "placeholder"), None)

    def _anchored_compute_op(self, node: fx.Node, fn, partition: CallPartition, anchor: fx.Node) -> ComputeOpMapper:
        """A call whose every argument is static (``torch.zeros(y.shape[-1])``, a user's ``torch.fx.wrap`` helper) is
        a host op of the value its shapes were read from, as on w0-base: the anchor is its operand in name only (the
        template reads literals and bound tensors), so the call runs at run time and never at conversion time."""
        template = partition.explicit_template(restore_native_shapes=True)
        adapter = ComputeAdapter(fn, bound_tensors=partition.payloads(), extra_args=partition.literals,
                                 kwargs=partition.literal_kwargs, arg_template=template)
        return ComputeOpMapper(self._get_mapper(anchor), adapter, input_shapes=strip_batch(node_output_shape(anchor)),
                               output_shape=strip_batch(node_output_shape(node)), name=node.name)

    def _batch_contract_adapter(
        self, node: fx.Node, fn, partition: CallPartition, source_nodes,
    ) -> ComputeAdapter:
        """The adapter whose batch-2 output is the traced per-sample output, else a named refusal.

        Stored (batch-stripped) bound tensors are kept when they reproduce the traced
        shape; otherwise their native shapes are restored; otherwise the op is refused.
        In an impure program (``symbolic/purity.py``) every bound tensor takes its
        native shape and the op never runs at conversion: a call outside the declared
        operator set may read its operands' ranks.
        """
        if partition.bound and self._batch_layouts.impurity is not None:
            template = partition.explicit_template(every_bound=True)
            return ComputeAdapter(fn, bound_tensors=partition.payloads(), extra_args=partition.literals,
                                  kwargs=partition.literal_kwargs, arg_template=template)
        adapter = ComputeAdapter.from_partition(fn, partition)
        traced = node_output_shape(node)
        if not partition.bound or traced is None or len(traced) == 0:
            return adapter
        inputs = _batch_contract_inputs(node, source_nodes, partition.bound[0].device)
        if inputs is None:
            return adapter
        expected = (_BATCH_CONTRACT_PROBE_SIZE, *traced[1:])
        observed = _observed_shape(adapter, inputs)
        if observed == expected:
            return adapter
        if partition.strips_bound_shapes:
            restored = ComputeAdapter.from_partition(fn, partition, restore_native_shapes=True)
            observed = _observed_shape(restored, inputs)
            if observed == expected:
                return restored
        raise GenericOpBatchAxisError(
            fn, node, expected, observed, partition.bound_native_shapes,
        )

    def _get_input_shape(self, node: fx.Node) -> Optional[Tuple[int, ...]]:
        shapes = node_input_shapes(node)
        return shapes[0] if shapes else None

    def _get_output_shape(self, node: fx.Node) -> Optional[Tuple[int, ...]]:
        return node_output_shape(node)

    def _get_constant_tensor(self, node: fx.Node):
        if isinstance(node, fx.Node) and node.op == "get_attr":
            value = self._get_attr_value(node)
            if isinstance(value, (nn.Parameter, torch.Tensor)):
                return value
        return None

    def _get_expanded_constant_tensor(self, node: fx.Node):
        if (
            isinstance(node, fx.Node)
            and node.op == "call_method"
            and node.target == "expand"
            and len(node.args) >= 1
            and isinstance(node.args[0], fx.Node)
        ):
            return self._get_constant_tensor(node.args[0])
        return None
