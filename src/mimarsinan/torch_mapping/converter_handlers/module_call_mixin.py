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
from mimarsinan.torch_mapping.fx_shape_utils import (
    broadcasts_to,
    carried_tuple_index,
    node_output_shape,
    node_target_str,
    strip_batch,
    traced_tuple,
    tuple_element_shape,
)
from mimarsinan.torch_mapping.leaf_decomposition.recurrence import Recurrence, input_projection
from mimarsinan.torch_mapping.leaf_decomposition.registry import lowered_at_conversion
from mimarsinan.torch_mapping.converter_handlers.refusals import (
    GenericOpArgumentError,
    ModuleCallArgumentError,
    TupleOutputIndexError,
    node_operator,
)
from mimarsinan.torch_mapping.representability_analyzer import (
    BATCHNORM_TYPES,
    RepresentabilityReport,
    core_lowering_requirement,
)


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
    _decompose_leaves = False
    """``decompose_leaf_modules``: a recurrent leaf is lowered to its input projections and recurrence."""

    def _handle_call_module(self, node: fx.Node, report: RepresentabilityReport) -> None:
        if node.name in self._absorbed:
            self._propagate_absorbed(node)
            return
        mod = self._modules.get(node_target_str(node))
        if mod is None:
            return
        foreign = node_target_str(node) in self._batch_layouts.purity.foreign_modules
        # Outside the declared set (a user class, a hook, a parameter of a tensor subclass), or needing a capability
        # the cores do not declare (``core_lowering_requirement``): the host op is the module's own call, as natively.
        own_call = foreign or core_lowering_requirement(mod) is not None
        if own_call or not isinstance(mod, self._SINGLE_INPUT_MODULES) or isinstance(mod, self._HOST_OP_MODULES):
            self._native_operands(node)  # a host op (its own call, a LayerNorm, a BatchNorm) reads native values
        else:
            self._native_static_operand(node)  # a perceptron, a convolution, a Flatten (review round 15)
        partition = self._partition_fx_args(node, mod)
        if self._decompose_leaves and not own_call and self._lower_recurrent(node, mod, partition, report):
            return
        if own_call:
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
        """The host op of the module's own call; a tuple output carries the one element its users read (the
        attention leaf's ``output_index``, generalized: an LSTM's sequence output)."""
        self._refuse_unresolved(node, mod, partition)
        if not partition.sources:
            self._node_to_mapper[node] = None
            return
        index = carried_tuple_index(node)
        output_shape = strip_batch(node_output_shape(node) if index is None else tuple_element_shape(node, index))
        if self._is_plain_call(partition):
            self._node_to_mapper[node] = ComputeOpMapper(
                partition.sources[0][0], mod, input_shapes=strip_batch(self._get_input_shape(node)),
                output_shape=output_shape, name=node.name, output_index=index,
            )
            return
        self._refuse_non_floating_activations(node, mod, partition)
        self._node_to_mapper[node] = self._templated_call_mapper(
            node, copy.deepcopy(mod), partition, output_shape=output_shape, output_index=index,
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
            # q, k, v only: the pre-template form, byte for byte; a key/value of another length than the query is
            # no elementwise operand (cross-attention, memory tokens).
            self._node_to_mapper[node] = ComputeOpMapper(
                [mapper for mapper, _ in partition.sources], copy.deepcopy(mod),
                input_shapes=self._source_shapes(partition), output_shape=output_shape, name=node.name,
                module_kwargs={"need_weights": False}, output_index=0,
                elementwise_inputs=broadcasts_to([arg for _, arg in partition.sources], query),
            )
            return
        self._refuse_non_floating_activations(node, mod, partition)
        self._node_to_mapper[node] = self._templated_call_mapper(
            node, copy.deepcopy(mod), partition, output_shape=output_shape,
            extra_kwargs={"need_weights": False}, output_index=0,
        )

    def _lower_recurrent(self, node: fx.Node, mod: nn.Module, partition: CallPartition, report) -> bool:
        """A recurrent leaf (``leaf_decomposition/recurrence.py``): per layer, its input projection by the Linear rule,
        then its ``Recurrence`` host op; False for a call the lowering does not take."""
        if not lowered_at_conversion(mod, self._is_plain_call(partition), carried_tuple_index(node)):
            return False
        assert isinstance(mod, nn.RNNBase)  # the one lowering ``CONVERSION_LOWERINGS`` registers
        source, shape = partition.sources[0][0], node_output_shape(partition.sources[0][1])
        steps = int(shape[1]) if shape is not None and len(shape) == 3 else 0
        for layer in range(int(mod.num_layers)):
            self._convert_linear(node, input_projection(mod, layer), source, report)
            projection = self._node_to_mapper[node]
            for named in (projection, getattr(projection, "perceptron", None)):
                if named is not None:
                    named.name = f"{node.name}_ih_l{layer}"
            recurrence = Recurrence(mod, layer, steps)
            last = layer == int(mod.num_layers) - 1
            source = ComputeOpMapper(projection, recurrence, input_shape=(steps, recurrence.input_width),
                                     output_shape=(steps, recurrence.output_width), output_index=0,
                                     name=node.name if last else f"{node.name}_l{layer}")
        self._node_to_mapper[node] = source
        return True

    def _carried_tuple_output(self, node: fx.Node, source_mapper) -> bool:
        """``getitem`` of a module whose conversion carries one tuple element: alias it, or refuse another index. Only
        a read of the tuple itself (its producer's traced value a tuple) is an element read; an index into the carried
        tensor (``h[-1]`` on a GRU's ``h_n``, ``o[0]`` on an attention output) is an ordinary tensor index."""
        carried = getattr(source_mapper, "output_index", None)
        producer, index = (tuple(node.args) + (None, None))[:2]
        if carried is None or not isinstance(producer, fx.Node) or traced_tuple(producer) is None:
            return False
        if index != carried:
            if node.users:
                raise TupleOutputIndexError(node, producer, index, carried)
            self._node_to_mapper[node] = None
            return True
        self._node_to_mapper[node] = source_mapper
        # The host op's value is this element: its batch layout is the element's (``h_n`` carries the batch on axis 1),
        # so stage 2 refuses it by name as it refuses any host value without the batch on axis 0 alone.
        violation = self._layout_violation(node)
        if violation is not None and isinstance(source_mapper, ComputeOpMapper):
            source_mapper.batch_layout_violation = violation
            source_mapper.host_operator = node_operator(producer, self._modules)
        return True
