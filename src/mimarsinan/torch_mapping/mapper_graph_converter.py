"""Convert a traced torch.fx graph into a mimarsinan Mapper DAG."""

from __future__ import annotations

import copy
import operator
from typing import Any, Dict, Set, Tuple

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.mapping.mapping_utils import (
    ComputeOpMapper,
    InputMapper,
    ModelRepresentation,
    PermuteMapper,
    ReshapeMapper,
)
from mimarsinan.torch_mapping.representability_analyzer import (
    RepresentabilityReport,
    RepresentabilityError,
)
from mimarsinan.torch_mapping.converter_handlers import (
    LinearConvertMixin,
    ConvConvertMixin,
    StructuralConvertMixin,
)
from mimarsinan.torch_mapping.mapper_graph_fx import MapperGraphFxMixin
from mimarsinan.torch_mapping.fx_shape_utils import (
    node_output_shape,
    strip_batch,
)


class MapperGraphConverter(
    MapperGraphFxMixin,
    LinearConvertMixin,
    ConvConvertMixin,
    StructuralConvertMixin,
):
    """Convert a traced ``GraphModule`` into a mimarsinan ``ModelRepresentation``."""

    def __init__(self, graph_module: fx.GraphModule, input_shape: Tuple[int, ...]):
        self.gm = graph_module
        self.input_shape = input_shape
        self._modules: Dict[str, nn.Module] = dict(graph_module.named_modules())
        self._node_to_mapper: Dict[fx.Node, Any] = {}
        self._node_to_attr: Dict[fx.Node, Any] = {}
        self._absorbed: Set[str] = set()

    def convert(self, report: RepresentabilityReport) -> ModelRepresentation:
        if not report.is_representable:
            raise RepresentabilityError(report)

        self._absorbed = {
            name for name, target in report.absorption_plan.items()
        }

        output_mapper = None
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
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

        if output_mapper is None:
            raise RuntimeError("FX graph has no output node")

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
        if isinstance(args, fx.Node):
            return self._get_mapper(args)
        if isinstance(args, (tuple, list)):
            return self._get_mapper(args[0])
        return None

    _PASSTHROUGH_MODULES = (nn.Identity, nn.Dropout, nn.Dropout2d,
                            nn.BatchNorm1d, nn.BatchNorm2d)

    def _handle_call_module(
        self, node: fx.Node, report: RepresentabilityReport
    ) -> None:
        if node.name in self._absorbed:
            self._propagate_absorbed(node)
            return

        mod = self._modules.get(node.target)
        if mod is None:
            return

        source = self._get_source_mapper(node)

        if isinstance(mod, nn.Linear):
            self._convert_linear(node, mod, source, report)
        elif isinstance(mod, nn.Conv2d):
            self._convert_conv2d(node, mod, source, report)
        elif isinstance(mod, nn.Conv1d):
            self._convert_conv1d(node, mod, source, report)
        elif isinstance(mod, nn.LayerNorm):
            self._node_to_mapper[node] = ComputeOpMapper(
                source, copy.deepcopy(mod), name=node.name,
            )
        elif isinstance(mod, nn.MultiheadAttention):
            self._convert_multihead_attention(node, mod)
        elif isinstance(mod, self._PASSTHROUGH_MODULES):
            self._node_to_mapper[node] = source
        elif isinstance(mod, nn.Flatten):
            self._convert_flatten_module(node, source)
        else:
            input_shape = strip_batch(self._get_input_shape(node))
            output_shape = strip_batch(node_output_shape(node))
            mapper = ComputeOpMapper(
                source, mod, input_shapes=input_shape,
                output_shape=output_shape, name=node.name,
            )
            self._node_to_mapper[node] = mapper

    def _handle_call_function(self, node: fx.Node) -> None:
        fn = node.target
        if fn is torch.cat:
            self._convert_cat(node)
        elif fn is torch.flatten:
            self._convert_flatten_func(node)
        elif fn is operator.getitem and not self._getitem_looks_like_real_slice(node):
            source_mapper = self._get_source_mapper(node)
            if getattr(source_mapper, "output_index", None) is not None:
                self._node_to_mapper[node] = source_mapper
            else:
                self._emit_generic_compute_op(node, fn)
        else:
            self._emit_generic_compute_op(node, fn)

    def _handle_call_method(self, node: fx.Node) -> None:
        method = node.target
        source = self._get_source_mapper(node)

        if method in ("view", "reshape"):
            shape_args = node.args[1:]
            if len(shape_args) == 1 and isinstance(shape_args[0], (tuple, list)):
                target_shape = tuple(shape_args[0])
            else:
                target_shape = tuple(
                    a if isinstance(a, int) else -1 for a in shape_args
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
            start_dim = node.args[1] if len(node.args) > 1 else 1
            out_shape = self._get_output_shape(node)
            if out_shape is not None and len(out_shape) >= 2:
                new_shape = out_shape[1:]
                mapper = ReshapeMapper(source, new_shape)
                self._node_to_mapper[node] = mapper
            else:
                self._node_to_mapper[node] = source

        elif method == "contiguous":
            self._node_to_mapper[node] = source

        elif method in ("permute", "transpose"):
            in_shape = self._get_input_shape(node)
            if method == "permute":
                raw = node.args[1:]
                if len(raw) == 1 and isinstance(raw[0], (tuple, list)):
                    dims = tuple(int(d) for d in raw[0])
                else:
                    dims = tuple(int(d) for d in raw)
            else:
                d0 = int(node.args[1]) if len(node.args) > 1 else 0
                d1 = int(node.args[2]) if len(node.args) > 2 else 1
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

        elif method in ("mean",):
            self._emit_generic_compute_op(node, torch.mean)

        elif method in ("size", "dim"):
            self._node_to_mapper[node] = source

        elif method in ("unsqueeze", "squeeze", "expand"):
            self._node_to_mapper[node] = source

        elif method in ("add", "__add__"):
            self._emit_generic_compute_op(node, operator.add)

        else:
            self._node_to_mapper[node] = source

    def _get_mapper(self, node: fx.Node):
        if node in self._node_to_mapper:
            return self._node_to_mapper[node]
        if len(node.args) >= 1 and isinstance(node.args[0], fx.Node):
            return self._get_mapper(node.args[0])
        return None

    def _get_attr_value(self, node: fx.Node):
        return self._node_to_attr.get(node)

    def _convert_multihead_attention(self, node: fx.Node, mod: nn.MultiheadAttention) -> None:
        source_nodes = [
            arg for arg in node.args[:3] if isinstance(arg, fx.Node)
        ]
        source_mappers = [self._get_mapper(arg) for arg in source_nodes]
        if not source_mappers:
            self._node_to_mapper[node] = None
            return
        input_shapes = [strip_batch(node_output_shape(arg)) for arg in source_nodes]
        output_shape = strip_batch(node_output_shape(source_nodes[0]))
        self._node_to_mapper[node] = ComputeOpMapper(
            source_mappers,
            copy.deepcopy(mod),
            input_shapes=input_shapes,
            output_shape=output_shape,
            name=node.name,
            module_kwargs={"need_weights": False},
            output_index=0,
        )

    def _get_source_mapper(self, node: fx.Node):
        if len(node.args) >= 1 and isinstance(node.args[0], fx.Node):
            return self._get_mapper(node.args[0])
        return None
