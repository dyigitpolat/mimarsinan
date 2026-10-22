"""Convert a traced torch.fx graph into a mimarsinan Mapper DAG."""

from __future__ import annotations

import operator
from typing import Any, Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mapping_utils import (
    InputMapper,
    ModelRepresentation,
    PermuteMapper,
    ReshapeMapper,
)
from mimarsinan.mapping.support.compute_modules import read_after
from mimarsinan.mapping.platform.packaging_contract import (
    SPIKING_PACKAGING,
    PackagingContract,
)
from mimarsinan.torch_mapping.representability_analyzer import (
    CONVERTIBLE_CALL_METHODS,
    RepresentabilityReport,
    RepresentabilityError,
)
from mimarsinan.torch_mapping.converter_handlers import (
    LinearConvertMixin,
    ConvConvertMixin,
    StructuralConvertMixin,
)
from mimarsinan.torch_mapping.mapper_graph_fx import MapperGraphFxMixin
from mimarsinan.torch_mapping.converter_handlers.batch_layout import BatchLayouts
from mimarsinan.torch_mapping.converter_handlers.layout_mixin import LayoutMixin
from mimarsinan.torch_mapping.converter_handlers.module_call_mixin import ModuleCallMixin
from mimarsinan.torch_mapping.converter_handlers.refusals import BatchAxisLayoutError
from mimarsinan.torch_mapping.converter_handlers.shape_value_mixin import ShapeValueMixin
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import ProgramPurity
from mimarsinan.torch_mapping.fx_shape_utils import fx_literal_int

# Methods that map one sample's layout to another's (a per-sample ReshapeMapper, PermuteMapper or identity).
LAYOUT_METHODS = frozenset({"view", "reshape", "flatten", "permute", "transpose", "unsqueeze", "squeeze", "expand"})
_ALIAS_METHODS = frozenset({"unsqueeze", "squeeze", "expand"})  # carried as their source's value (``_lower_layout``)


class UnsupportedCallMethodError(ToolchainRefusal, RuntimeError):
    """A tensor call_method with no conversion rule reached the converter.

    Silently mapping it to identity would DROP the op (the ``.softmax(dim=-1)``
    regression); refusing loudly is the only correct behavior.
    """

    def __init__(self, method: str, node: fx.Node):
        self.method = method
        self.node_name = node.name
        super().__init__(
            f"call_method '.{method}(...)' (FX node '{node.name}') has no "
            "conversion rule; refusing to silently map it to identity. "
            f"Allowlisted structural methods: {sorted(CONVERTIBLE_CALL_METHODS)}"
        )


class MapperGraphConverter(
    MapperGraphFxMixin,
    LayoutMixin,
    ModuleCallMixin,
    ShapeValueMixin,
    LinearConvertMixin,
    ConvConvertMixin,
    StructuralConvertMixin,
):
    """Convert a traced ``GraphModule`` into a mimarsinan ``ModelRepresentation``; ``purity``: the traced program's,
    before graph normalization (``trace_and_convert``), else computed from ``graph_module``."""

    def __init__(
        self,
        graph_module: fx.GraphModule,
        input_shape: Tuple[int, ...],
        packaging: "PackagingContract" = SPIKING_PACKAGING,
        purity: Optional[ProgramPurity] = None,
    ):
        self.gm = graph_module
        self.input_shape = input_shape
        self._packaging = packaging
        self._modules: Dict[str, nn.Module] = dict(graph_module.named_modules())
        self._node_to_mapper: Dict[fx.Node, Any] = {}
        self._node_to_attr: Dict[fx.Node, Any] = {}
        self._shape_values: Dict[fx.Node, Any] = {}
        self._batch_layouts = BatchLayouts(graph_module, purity)
        self._absorbed: Set[str] = set()

    def convert(self, report: RepresentabilityReport) -> ModelRepresentation:
        if not report.is_representable:
            raise RepresentabilityError(report)

        self._absorbed = {
            name for name, target in report.absorption_plan.items()
        }

        output_mapper = None
        for node in self.gm.graph.nodes:
            if self._guarded_copy(node):
                pass  # a copy the rule of a write relies on: a host op checking it (review round 16)
            elif node.op == "placeholder":
                self._handle_placeholder(node)
            elif node.op == "get_attr":
                self._handle_get_attr(node)
            elif node.op == "call_module":
                self._handle_call_module(node, report)
            elif node.op == "call_function":
                self._handle_call_function(node)
            elif node.op == "call_method":
                self._handle_call_method(node)
            elif node.op == "output":
                output_mapper = self._handle_output(node)
            self._mark_host_value_layout(node)
            self._mark_kept_inputs(node)

        if output_mapper is None:
            raise RuntimeError("FX graph has no output node")
        self._require_static_reads()
        self._order_runs(output_mapper)

        return ModelRepresentation(output_mapper)

    def _handle_placeholder(self, node: fx.Node) -> None:
        mapper = InputMapper(self.input_shape)
        self._node_to_mapper[node] = mapper

    def _handle_get_attr(self, node: fx.Node) -> None:
        obj = self.gm
        for part in str(node.target).split("."):
            obj = getattr(obj, part)
        self._node_to_attr[node] = obj

    def _handle_output(self, node: fx.Node):
        args = node.args[0]
        returned = args if isinstance(args, fx.Node) else args[0] if (
            isinstance(args, (tuple, list)) and args and isinstance(args[0], fx.Node)) else None
        if returned is None:
            return None
        self._native_operands(node, [returned])
        return self._get_mapper(returned)

    def _handle_call_function(self, node: fx.Node) -> None:
        fn = node.target
        if self._record_shape_value(node):
            return
        if fn is read_after:  # its other operands order it, they are not values it reads (``operand_writes``)
            self._emit_generic_compute_op(node, fn)
            self._node_to_mapper[node].elementwise_inputs = False
        elif fn is torch.cat:
            self._convert_cat(node)
        elif fn is torch.flatten:
            self._convert_flatten_func(node)
        elif fn is operator.getitem and not self._getitem_looks_like_real_slice(node):
            if not self._carried_tuple_output(node, self._get_source_mapper(node)):
                self._emit_generic_compute_op(node, fn)
        else:
            self._emit_generic_compute_op(node, fn)

    def _handle_call_method(self, node: fx.Node) -> None:
        method = node.target
        if method in LAYOUT_METHODS and method not in _ALIAS_METHODS:
            self._native_static_operand(node)  # an alias of another shape read as natively (review round 15)
        source = self._get_source_mapper(node)
        if method in LAYOUT_METHODS:
            self._require_batch_leading(node)
            self._require_restored(node)
            if not self._native_layout(node, getattr(torch.Tensor, str(method))):
                self._lower_layout(node, str(method), source)
            return

        if method == "contiguous":
            self._node_to_mapper[node] = source

        elif method in ("mean",):
            self._emit_generic_compute_op(node, torch.mean)

        elif method in ("size", "dim"):
            if not self._record_shape_value(node):
                self._emit_generic_compute_op(node, getattr(torch.Tensor, str(method)))

        elif method in ("add", "__add__"):
            self._emit_generic_compute_op(node, operator.add)

        else:
            raise UnsupportedCallMethodError(str(method), node)

    def _lower_layout(self, node: fx.Node, method: str, source) -> None:
        """A layout method with the per-sample shape fixed at conversion (the traced one)."""
        if method in ("view", "reshape"):
            shape_args = node.args[1:]
            if len(shape_args) == 1 and isinstance(shape_args[0], (tuple, list)):
                raw_dims = tuple(shape_args[0])
            else:
                raw_dims = shape_args
            # Dynamic (Node) dims act as -1 wildcards; the resolved FX shape wins when present.
            target_shape = tuple(
                a if isinstance(a, int) else -1 for a in raw_dims
            )
            target_shape_no_batch = target_shape[1:] if len(target_shape) > 1 else target_shape
            out_shape = self._get_output_shape(node)
            resolved_shape_no_batch = (
                tuple(out_shape[1:]) if out_shape is not None and len(out_shape) >= 2 else None
            )
            if resolved_shape_no_batch is not None and all(d > 0 for d in resolved_shape_no_batch):
                mapper = ReshapeMapper(source, resolved_shape_no_batch)
            elif all(d > 0 for d in target_shape_no_batch):
                mapper = ReshapeMapper(source, target_shape_no_batch)
            else:
                mapper = source
            self._node_to_mapper[node] = mapper

        elif method == "flatten":
            out_shape = self._get_output_shape(node)
            if out_shape is not None and len(out_shape) >= 2:
                new_shape = out_shape[1:]
                mapper = ReshapeMapper(source, new_shape)
                self._node_to_mapper[node] = mapper
            else:
                self._node_to_mapper[node] = source

        elif method in ("permute", "transpose"):
            in_shape = self._get_input_shape(node)
            raw = [self._static_dim(node, d, BatchAxisLayoutError) for d in node.args[1:]]
            if method == "permute":
                dims = tuple(raw[0]) if len(raw) == 1 and isinstance(raw[0], (tuple, list)) else tuple(raw)
                dims = tuple(fx_literal_int(d) for d in dims)
            else:
                d0 = fx_literal_int(raw[0]) if len(raw) > 0 else 0
                d1 = fx_literal_int(raw[1]) if len(raw) > 1 else 1
                ndim = len(in_shape) if in_shape else 3
                dims = list(range(ndim))
                dims[d0], dims[d1] = dims[d1], dims[d0]
                dims = tuple(dims)
            if dims and dims[0] == 0:
                self._node_to_mapper[node] = PermuteMapper(source, dims)
            else:
                out_shape = self._get_output_shape(node)
                if out_shape is not None and len(out_shape) >= 2:
                    self._node_to_mapper[node] = ReshapeMapper(source, out_shape[1:])
                else:
                    self._node_to_mapper[node] = source

        else:  # unsqueeze, squeeze, expand: the source carried as an alias
            self._node_to_mapper[node] = source

    def _get_mapper(self, node: fx.Node):
        if node in self._node_to_mapper:
            return self._node_to_mapper[node]
        if len(node.args) >= 1 and isinstance(node.args[0], fx.Node):
            return self._get_mapper(node.args[0])
        return None

    def _get_attr_value(self, node: fx.Node):
        return self._node_to_attr.get(node)

    def _get_source_mapper(self, node: fx.Node):
        if len(node.args) >= 1 and isinstance(node.args[0], fx.Node):
            return self._get_mapper(node.args[0])
        return None
