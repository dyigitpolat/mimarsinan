"""Module calls: every operand of a call_module node is carried into the converted call, or refused by name."""

from __future__ import annotations

import copy
from typing import Any, List

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.mapping.support.call_template import CallPartition
from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.torch_mapping.converter_handlers.batchnorm_lowering import batchnorm_host_op
from mimarsinan.torch_mapping.converter_handlers.converter_contract import ConverterContract
from mimarsinan.torch_mapping.fx_shape_utils import node_output_shape, node_target_str, strip_batch
from mimarsinan.torch_mapping.converter_handlers.refusals import (
    GenericOpArgumentError,
    ModuleCallArgumentError,
    TupleOutputIndexError,
)
from mimarsinan.torch_mapping.representability_analyzer import BATCHNORM_TYPES, RepresentabilityReport


def _operand_label(partition: CallPartition, index: int, node: fx.Node) -> str:
    """How source ``index`` enters the call: its keyword, else the FX node that computes it."""
    for key, value in node.kwargs.items():
        if value is partition.sources[index][1]:
            return f"keyword {key!r}"
    return f"operand {partition.sources[index][1].name!r}"


class ModuleCallMixin(ConverterContract):
    """call_module dispatch: dedicated single-input rules, MultiheadAttention, and the generic host-op call."""

    _PASSTHROUGH_MODULES = (nn.Identity, nn.Dropout, nn.Dropout2d)
    _SINGLE_INPUT_MODULES = (
        nn.Linear, nn.Conv1d, nn.Conv2d, nn.LayerNorm, *BATCHNORM_TYPES, *_PASSTHROUGH_MODULES, nn.Flatten,
    )
    _HOST_OP_MODULES = (nn.LayerNorm, *BATCHNORM_TYPES)
    """Single-input modules converted as a host op of the module (review round 15: their operands' native values)."""

    def _handle_call_module(self, node: fx.Node, report: RepresentabilityReport) -> None:
        if node.name in self._absorbed:
            self._propagate_absorbed(node)
            return
        mod = self._modules.get(node_target_str(node))
        if mod is None:
            return
        foreign = node_target_str(node) in self._batch_layouts.purity.foreign_modules
        if foreign or not isinstance(mod, self._SINGLE_INPUT_MODULES) or isinstance(mod, self._HOST_OP_MODULES):
            self._native_operands(node)  # a host op (its own call, a LayerNorm, a BatchNorm) reads native values
        else:
            self._native_static_operand(node)  # a perceptron, a convolution, a Flatten (review round 15)
        partition = self._partition_fx_args(node, mod)
        if foreign:
            # Outside the declared set (a user class, a hook, a parameter of a tensor subclass): no conversion rule
            # computes with its state; the host op is its own call, as natively.
            self._convert_module_call(node, mod, partition)
        elif isinstance(mod, nn.MultiheadAttention):
            self._convert_multihead_attention(node, mod, partition)
        elif isinstance(mod, self._SINGLE_INPUT_MODULES):
            self._convert_single_input_module(node, mod, self._single_activation(node, mod, partition), report)
        else:
            self._convert_module_call(node, mod, partition)

    def _convert_single_input_module(self, node: fx.Node, mod: nn.Module, source, report) -> None:
        if isinstance(mod, nn.Linear):
            self._convert_linear(node, mod, source, report)
        elif isinstance(mod, nn.Conv2d):
            self._convert_conv2d(node, mod, source, report)
        elif isinstance(mod, nn.Conv1d):
            self._convert_conv1d(node, mod, source, report)
        elif isinstance(mod, nn.LayerNorm):
            self._node_to_mapper[node] = ComputeOpMapper(source, copy.deepcopy(mod), name=node.name)
        elif isinstance(mod, BATCHNORM_TYPES):
            self._node_to_mapper[node] = batchnorm_host_op(node, mod, source)
        elif isinstance(mod, self._PASSTHROUGH_MODULES):
            self._node_to_mapper[node] = source
        else:
            self._convert_flatten_module(node, source)

    def _single_activation(self, node: fx.Node, mod: nn.Module, partition: CallPartition):
        """The one activation a dedicated rule converts (positional or keyword); any other operand is refused."""
        self._refuse_unresolved(node, mod, partition)
        extra = []
        if len(partition.sources) != 1:
            extra.append(f"{len(partition.sources)} activation operands")
        if partition.bound:
            extra.append(f"{len(partition.bound)} parameter/buffer operands")
        if partition.literals or partition.literal_kwargs:
            extra.append(f"literal operands {list(partition.literals)} {partition.literal_kwargs}")
        if extra:
            raise ModuleCallArgumentError(
                mod, node, "its conversion rule carries exactly one activation input; the call passes "
                + ", ".join(extra),
            )
        return partition.sources[0][0]

    def _refuse_unresolved(self, node: fx.Node, mod: nn.Module, partition: CallPartition) -> None:
        if partition.unresolved:
            raise GenericOpArgumentError.for_node(mod, node, partition.unresolved[0])

    def _refuse_non_floating_activations(self, node: fx.Node, mod: nn.Module, partition: CallPartition) -> None:
        """Host ops exchange activations in the program's value dtype; a mask would be cast and change meaning."""
        for index, (_, arg) in enumerate(partition.sources):
            meta = arg.meta.get("tensor_meta")
            dtype = getattr(meta, "dtype", None)
            if isinstance(dtype, torch.dtype) and not (dtype.is_floating_point or dtype.is_complex):
                raise ModuleCallArgumentError(
                    mod, node,
                    f"{_operand_label(partition, index, node)} is an activation of dtype {dtype} "
                    f"(computed by FX node {arg.name!r}); host-op activations carry the program's value dtype",
                )

    @staticmethod
    def _is_plain_call(partition: CallPartition) -> bool:
        """``module(activation)``: the form converted with the module itself as the host payload."""
        return (
            len(partition.sources) == 1 and not partition.bound and not partition.literals
            and not partition.literal_kwargs and partition.template is None
        )

    def _source_shapes(self, partition: CallPartition) -> List[Any]:
        return [strip_batch(node_output_shape(arg)) for _, arg in partition.sources]

    def _convert_module_call(self, node: fx.Node, mod: nn.Module, partition: CallPartition) -> None:
        self._refuse_unresolved(node, mod, partition)
        if not partition.sources:
            self._node_to_mapper[node] = None
            return
        if self._is_plain_call(partition):
            self._node_to_mapper[node] = ComputeOpMapper(
                partition.sources[0][0], mod, input_shapes=strip_batch(self._get_input_shape(node)),
                output_shape=strip_batch(node_output_shape(node)), name=node.name,
            )
            return
        self._refuse_non_floating_activations(node, mod, partition)
        self._node_to_mapper[node] = self._templated_call_mapper(
            node, copy.deepcopy(mod), partition, output_shape=strip_batch(node_output_shape(node)),
        )

    def _templated_call_mapper(self, node: fx.Node, fn: nn.Module, partition: CallPartition, *,
                               output_shape, extra_kwargs=None, output_index=None) -> ComputeOpMapper:
        """A host op evaluating ``fn`` on the call's operands in their traced order (bound tensors at native shape)."""
        adapter = ComputeAdapter.from_partition(fn, partition, restore_native_shapes=True)
        adapter.kwargs.update(extra_kwargs or {})
        shapes = self._source_shapes(partition)
        return ComputeOpMapper(
            [mapper for mapper, _ in partition.sources], adapter,
            input_shapes=shapes[0] if len(shapes) == 1 else shapes, output_shape=output_shape,
            name=node.name, output_index=output_index, elementwise_inputs=False,
        )

    def _convert_multihead_attention(
        self, node: fx.Node, mod: nn.MultiheadAttention, partition: CallPartition,
    ) -> None:
        self._refuse_unresolved(node, mod, partition)
        if not partition.sources:
            self._node_to_mapper[node] = None
            return
        query = partition.sources[0][1]
        output_shape = strip_batch(node_output_shape(query))
        if (
            1 <= len(partition.sources) <= 3 and partition.template is None and not partition.bound
            and not partition.literals and set(partition.literal_kwargs) <= {"need_weights"}
        ):
            # q, k, v only: the pre-template form, byte for byte.
            self._node_to_mapper[node] = ComputeOpMapper(
                [mapper for mapper, _ in partition.sources], copy.deepcopy(mod),
                input_shapes=self._source_shapes(partition), output_shape=output_shape, name=node.name,
                module_kwargs={"need_weights": False}, output_index=0,
            )
            return
        self._refuse_non_floating_activations(node, mod, partition)
        self._node_to_mapper[node] = self._templated_call_mapper(
            node, copy.deepcopy(mod), partition, output_shape=output_shape,
            extra_kwargs={"need_weights": False}, output_index=0,
        )

    def _carried_tuple_output(self, node: fx.Node, source_mapper) -> bool:
        """``getitem`` of a module whose conversion carries one tuple element: alias it, or refuse another index."""
        carried = getattr(source_mapper, "output_index", None)
        if carried is None:
            return False
        producer, index = (tuple(node.args) + (None, None))[:2]
        if index != carried:
            if node.users:
                assert isinstance(producer, fx.Node)
                raise TupleOutputIndexError(node, producer, index, carried)
            self._node_to_mapper[node] = None
            return True
        self._node_to_mapper[node] = source_mapper
        return True
