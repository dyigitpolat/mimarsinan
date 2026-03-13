"""
Convert a traced torch.fx graph into a mimarsinan Mapper DAG.

Walks the FX graph in topological order, creates Mapper nodes for each
operation, absorbs BatchNorm / activation into preceding Perceptrons,
and transfers trained weights.
"""

from __future__ import annotations

import operator
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx

from mimarsinan.mapping.mapping_utils import (
    InputMapper,
    ModelRepresentation,
    ModuleMapper,
    GELUMapper,
    DropoutMapper,
    ReshapeMapper,
)
from mimarsinan.torch_mapping.representability_analyzer import (
    RepresentabilityAnalyzer,
    RepresentabilityReport,
    RepresentabilityError,
)
from mimarsinan.torch_mapping.converter_handlers import (
    LinearConvertMixin,
    ConvConvertMixin,
    PoolConvertMixin,
    TransformerConvertMixin,
    StructuralConvertMixin,
)


class MapperGraphConverter(
    LinearConvertMixin,
    ConvConvertMixin,
    PoolConvertMixin,
    TransformerConvertMixin,
    StructuralConvertMixin,
):
    """Convert a traced ``GraphModule`` into a mimarsinan ``ModelRepresentation``.

    The converter walks the FX graph topologically.  For each node it
    creates the appropriate Mapper, absorbing BatchNorm / activation layers
    into the preceding Perceptron where possible.  Trained weights are
    transferred from the original modules.
    """

    def __init__(self, graph_module: fx.GraphModule, input_shape: Tuple[int, ...]):
        self.gm = graph_module
        self.input_shape = input_shape
        self._modules: Dict[str, nn.Module] = dict(graph_module.named_modules())
        self._node_to_mapper: Dict[fx.Node, Any] = {}
        self._absorbed: Set[str] = set()

    def convert(self, report: RepresentabilityReport) -> ModelRepresentation:
        """Run the conversion using a pre-computed representability report.

        Args:
            report: A ``RepresentabilityReport`` from ``RepresentabilityAnalyzer``.
                Must have ``is_representable == True``.

        Returns:
            A ``ModelRepresentation`` (root of the Mapper DAG).
        """
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
        pass

    def _handle_output(self, node: fx.Node):
        args = node.args[0]
        if isinstance(args, fx.Node):
            return self._get_mapper(args)
        if isinstance(args, (tuple, list)):
            return self._get_mapper(args[0])
        return None

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
        elif isinstance(mod, nn.MaxPool2d):
            self._convert_maxpool2d(node, mod, source)
        elif isinstance(mod, nn.AvgPool2d):
            self._convert_avgpool2d(node, mod, source)
        elif isinstance(mod, nn.AdaptiveAvgPool2d):
            self._convert_adaptive_avgpool2d(node, mod, source)
        elif isinstance(mod, nn.LayerNorm):
            self._convert_layer_norm(node, mod, source)
        elif isinstance(mod, nn.GELU):
            self._convert_gelu(node, source)
        elif isinstance(mod, (nn.Dropout, nn.Dropout2d)):
            self._convert_dropout(node, mod, source)
        elif isinstance(mod, nn.Identity):
            self._node_to_mapper[node] = source
        elif isinstance(mod, nn.Flatten):
            self._convert_flatten_module(node, source)
        elif isinstance(mod, (nn.ReLU, nn.LeakyReLU)):
            mapper = ModuleMapper(source, mod)
            self._node_to_mapper[node] = mapper
        elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            self._node_to_mapper[node] = source
        else:
            mapper = ModuleMapper(source, mod)
            self._node_to_mapper[node] = mapper

    def _handle_call_function(self, node: fx.Node) -> None:
        fn = node.target

        if fn is operator.add or fn is torch.add:
            self._convert_add(node)
        elif fn is torch.flatten:
            self._convert_flatten_func(node)
        elif fn is torch.relu or fn is F.relu:
            source = self._get_source_mapper(node)
            mapper = ModuleMapper(source, nn.ReLU())
            self._node_to_mapper[node] = mapper
        elif fn is F.gelu:
            source = self._get_source_mapper(node)
            mapper = GELUMapper(source, name=node.name)
            self._node_to_mapper[node] = mapper
        elif fn is F.dropout:
            source = self._get_source_mapper(node)
            mapper = DropoutMapper(source, name=node.name)
            self._node_to_mapper[node] = mapper
        elif fn is F.adaptive_avg_pool2d:
            self._convert_adaptive_avgpool2d_func(node)
        elif fn is F.max_pool2d:
            self._convert_maxpool2d_func(node)
        elif fn is F.avg_pool2d:
            self._convert_avgpool2d_func(node)
        elif fn is F.layer_norm:
            self._convert_layer_norm_func(node)
        elif fn is operator.getitem:
            self._convert_getitem(node)
        elif fn is torch.cat:
            self._convert_cat(node)
        else:
            source = self._get_source_mapper(node)
            self._node_to_mapper[node] = source

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
            if all(d > 0 for d in target_shape_no_batch):
                mapper = ReshapeMapper(source, target_shape_no_batch)
            else:
                mapper = source
            self._node_to_mapper[node] = mapper

        elif method == "flatten":
            start_dim = node.args[1] if len(node.args) > 1 else 1
            if start_dim <= 1:
                out_shape = self._get_output_shape(node)
                if out_shape is not None:
                    flat_shape = (out_shape[-1],) if len(out_shape) == 2 else out_shape[1:]
                    mapper = ReshapeMapper(source, flat_shape)
                    self._node_to_mapper[node] = mapper
                    return
            self._node_to_mapper[node] = source

        elif method == "contiguous":
            self._node_to_mapper[node] = source

        elif method in ("permute", "transpose"):
            self._node_to_mapper[node] = source

        elif method in ("mean",):
            self._node_to_mapper[node] = source

        elif method in ("size", "dim"):
            self._node_to_mapper[node] = source

        elif method in ("unsqueeze", "squeeze", "expand"):
            self._node_to_mapper[node] = source

        elif method in ("add", "__add__"):
            self._convert_add(node)

        else:
            self._node_to_mapper[node] = source

    def _get_mapper(self, node: fx.Node):
        """Resolve the mapper for a node, handling passthrough for absorbed nodes."""
        if node in self._node_to_mapper:
            return self._node_to_mapper[node]
        if len(node.args) >= 1 and isinstance(node.args[0], fx.Node):
            return self._get_mapper(node.args[0])
        return None

    def _get_source_mapper(self, node: fx.Node):
        """Get the mapper for the first argument of a node."""
        if len(node.args) >= 1 and isinstance(node.args[0], fx.Node):
            return self._get_mapper(node.args[0])
        return None

    def _get_input_shape(self, node: fx.Node) -> Optional[Tuple[int, ...]]:
        """Get the input tensor shape (with batch) from ShapeProp metadata."""
        if len(node.args) >= 1 and isinstance(node.args[0], fx.Node):
            input_node = node.args[0]
            meta = input_node.meta.get("tensor_meta")
            if meta is not None:
                return tuple(meta.shape)
        return None

    def _get_output_shape(self, node: fx.Node) -> Optional[Tuple[int, ...]]:
        """Get the output tensor shape (with batch) from ShapeProp metadata."""
        meta = node.meta.get("tensor_meta")
        if meta is not None:
            return tuple(meta.shape)
        return None
