"""
Convert a traced torch.fx graph into a mimarsinan Mapper DAG.

Walks the FX graph in topological order, creates Mapper nodes for each
operation, absorbs BatchNorm / activation into preceding Perceptrons,
and transfers trained weights.
"""

from __future__ import annotations

import copy
import operator
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx

from mimarsinan.mapping.compute_modules import ComputeAdapter
from mimarsinan.mapping.mapping_utils import (
    ComputeOpMapper,
    InputMapper,
    ModelRepresentation,
    PermuteMapper,
    ReshapeMapper,
    SubscriptMapper,
)
from mimarsinan.mapping.mappers.base import Mapper
from mimarsinan.torch_mapping.representability_analyzer import (
    RepresentabilityAnalyzer,
    RepresentabilityReport,
    RepresentabilityError,
)
from mimarsinan.torch_mapping.converter_handlers import (
    LinearConvertMixin,
    ConvConvertMixin,
    StructuralConvertMixin,
)


class MapperGraphConverter(
    LinearConvertMixin,
    ConvConvertMixin,
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
        self._node_to_attr: Dict[fx.Node, Any] = {}
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

    # Modules that are passthrough (no IR node) when not absorbed:
    # - Identity: no-op
    # - Dropout: identity at inference / spiking simulation
    # - BatchNorm: standalone BN (rare; if not absorbed, pass through)
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

        # Perceptron candidates: Linear/Conv + absorbed BN/activation → Perceptron
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
        # Passthrough: no IR node needed
        elif isinstance(mod, self._PASSTHROUGH_MODULES):
            self._node_to_mapper[node] = source
        elif isinstance(mod, nn.Flatten):
            self._convert_flatten_module(node, source)
        else:
            in_shape = self._get_input_shape(node)
            input_shape = tuple(in_shape[1:]) if in_shape and len(in_shape) >= 2 else None
            out_shape = self._get_output_shape(node)
            output_shape = tuple(out_shape[1:]) if out_shape and len(out_shape) >= 2 else None
            mapper = ComputeOpMapper(
                source, mod, input_shapes=input_shape,
                output_shape=output_shape, name=node.name,
            )
            self._node_to_mapper[node] = mapper

    def _handle_call_function(self, node: fx.Node) -> None:
        fn = node.target
        # torch.cat / torch.flatten have non-flat signatures or fold to
        # shape-only mappers; everything else goes through the generic path.
        if fn is torch.cat:
            self._convert_cat(node)
        elif fn is torch.flatten:
            self._convert_flatten_func(node)
        elif fn is operator.getitem and not self._getitem_looks_like_real_slice(node):
            # FX-internal tuple unpacking (e.g. ``mhsa_out[0]``); pass through.
            self._node_to_mapper[node] = self._get_source_mapper(node)
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
                # Non-batch-preserving permutation: fall back to shape-only tracking.
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
        """Resolve the mapper for a node, handling passthrough for absorbed nodes."""
        if node in self._node_to_mapper:
            return self._node_to_mapper[node]
        if len(node.args) >= 1 and isinstance(node.args[0], fx.Node):
            return self._get_mapper(node.args[0])
        return None

    def _get_attr_value(self, node: fx.Node):
        return self._node_to_attr.get(node)

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

    def _convert_multihead_attention(self, node: fx.Node, mod: nn.MultiheadAttention) -> None:
        source_nodes = [
            arg for arg in node.args[:3] if isinstance(arg, fx.Node)
        ]
        source_mappers = [self._get_mapper(arg) for arg in source_nodes]
        if not source_mappers:
            self._node_to_mapper[node] = None
            return
        input_shapes = []
        for arg in source_nodes:
            shape = self._get_output_shape(arg)
            input_shapes.append(tuple(shape[1:]) if shape is not None and len(shape) >= 2 else None)
        query_shape = self._get_output_shape(source_nodes[0])
        output_shape = (
            tuple(query_shape[1:]) if query_shape is not None and len(query_shape) >= 2 else None
        )
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
        """Get the mapper for the first argument of a node."""
        if len(node.args) >= 1 and isinstance(node.args[0], fx.Node):
            return self._get_mapper(node.args[0])
        return None

    @staticmethod
    def _getitem_looks_like_real_slice(node: fx.Node) -> bool:
        """True if ``operator.getitem`` selects from a tensor (vs. tuple unpack).

        Tuple-unpack patterns (``mhsa_out[0]``) have a plain ``int`` index and
        pass through; real slices have a tuple containing a slice / Ellipsis.
        """
        if len(node.args) < 2:
            return False
        index = node.args[1]
        if isinstance(index, tuple):
            return any(
                isinstance(part, slice) or part is Ellipsis for part in index
            )
        return False

    def _partition_fx_args(
        self, node: fx.Node,
    ) -> Tuple[list, list, tuple, dict]:
        """Classify ``node.args`` / ``node.kwargs`` for :class:`ComputeAdapter`.

        Returns ``(sources, bound_tensors, extra_args, kwargs)``: ``fx.Node``
        args resolve to either a mapper (source) or a ``get_attr`` tensor
        (bound); everything else is ``extra_args`` / ``kwargs``.  Lists-of-Nodes
        (e.g. ``torch.cat``'s ``tensors`` arg) need a dedicated handler.
        """
        sources: list = []
        bound: list = []
        extra: list = []
        for arg in node.args:
            if isinstance(arg, fx.Node):
                const = self._get_constant_tensor(arg)
                if const is not None:
                    bound.append(const)
                    continue
                mapper = self._get_mapper(arg)
                if mapper is not None:
                    sources.append(mapper)
                continue
            extra.append(arg)
        kwargs = {
            k: v for k, v in node.kwargs.items() if not isinstance(v, fx.Node)
        }
        return sources, bound, tuple(extra), kwargs

    def _emit_generic_compute_op(self, node: fx.Node, fn) -> None:
        sources, bound, extra, kwargs = self._partition_fx_args(node)
        if not sources:
            self._node_to_mapper[node] = None
            return
        adapter = ComputeAdapter(
            fn,
            bound_tensors=bound,
            extra_args=extra,
            kwargs=kwargs,
        )
        in_shape = self._get_input_shape(node)
        input_shape = (
            tuple(in_shape[1:]) if in_shape and len(in_shape) >= 2 else None
        )
        out_shape = self._get_output_shape(node)
        output_shape = (
            tuple(out_shape[1:]) if out_shape and len(out_shape) >= 2 else None
        )
        self._node_to_mapper[node] = ComputeOpMapper(
            sources,
            adapter,
            input_shapes=input_shape,
            output_shape=output_shape,
            name=node.name,
        )


    def _get_input_shape(self, node: fx.Node) -> Optional[Tuple[int, ...]]:
        """Get the input tensor shape (with batch) from ShapeProp metadata."""
        if len(node.args) >= 1 and isinstance(node.args[0], fx.Node):
            input_node = node.args[0]
            meta = input_node.meta.get("tensor_meta")
            if meta is not None and hasattr(meta, "shape"):
                return tuple(meta.shape)
        return None

    def _get_output_shape(self, node: fx.Node) -> Optional[Tuple[int, ...]]:
        """Get the output tensor shape (with batch) from ShapeProp metadata."""
        meta = node.meta.get("tensor_meta")
        if meta is not None and hasattr(meta, "shape"):
            return tuple(meta.shape)
        return None
