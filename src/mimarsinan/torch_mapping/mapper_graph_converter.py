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

from mimarsinan.mapping.mapping_utils import (
    ConstantAddMapper,
    ConstantPrependMapper,
    InputMapper,
    LayerNormMapper,
    MeanMapper,
    ModelRepresentation,
    ModuleComputeMapper,
    PermuteMapper,
    ReshapeMapper,
    SelectMapper,
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


class _FunctionWrapper(nn.Module):
    """Wrap an ``F.*`` function call as an ``nn.Module`` for ModuleComputeMapper.

    Captures the function and its non-tensor arguments (kwargs and positional
    args beyond the first tensor) from an FX node so the wrapped function can
    be called as ``module(x)`` during forward and spiking simulation.
    """

    def __init__(self, fn, extra_args=(), kwargs=None):
        super().__init__()
        self.fn = fn
        self.extra_args = extra_args
        self.kwargs = kwargs or {}

    def forward(self, x):
        return self.fn(x, *self.extra_args, **self.kwargs)

    @classmethod
    def from_fx_node(cls, node: fx.Node, fn) -> "_FunctionWrapper":
        # args[0] is the input tensor (handled by the mapper source);
        # remaining args are parameters (kernel_size, stride, etc.)
        extra_args = tuple(a for a in node.args[1:] if not isinstance(a, fx.Node))
        # Filter out training= for dropout etc. — keep all kwargs
        kwargs = {k: v for k, v in node.kwargs.items() if not isinstance(v, fx.Node)}
        return cls(fn, extra_args, kwargs)


class _MultiInputModuleComputeMapper(Mapper):
    """Host-side ComputeOp for modules that consume multiple tensor inputs."""

    def __init__(
        self,
        source_mappers,
        module: nn.Module,
        *,
        input_shapes=None,
        output_shape=None,
        name=None,
        module_kwargs=None,
        output_index=None,
    ):
        super().__init__()
        self._source_mappers_list = list(source_mappers)
        self.module = module
        self.input_shapes = input_shapes
        self.output_shape = output_shape
        self.name = name
        self.module_kwargs = module_kwargs or {}
        self.output_index = output_index

    def get_source_mappers(self):
        return [m for m in self._source_mappers_list if m is not None]

    def _forward_impl(self, x):
        inputs = tuple(x) if isinstance(x, tuple) else (x,)
        out = self.module(*inputs, **self.module_kwargs)
        if self.output_index is not None:
            out = out[self.output_index]
        return out

    def _map_to_ir(self, ir_mapping):
        source_arrays = [
            np.array(mapper.map_to_ir(ir_mapping), dtype=object)
            for mapper in self.get_source_mappers()
        ]
        flat_sources = np.concatenate([arr.flatten() for arr in source_arrays])
        input_shapes = self.input_shapes or [tuple(arr.shape) for arr in source_arrays]
        return ir_mapping.add_compute_op(
            input_sources=flat_sources,
            op_type="module",
            params={
                "module": self.module,
                "input_shapes": [tuple(shape) for shape in input_shapes],
                "module_kwargs": self.module_kwargs,
                "output_index": self.output_index,
            },
            input_shape=None,
            output_shape=self.output_shape,
            name=self.name,
        )

    def _map(self, mapping):
        raise NotImplementedError(
            f"{self.name}: multi-input ModuleComputeMapper does not support legacy SoftCoreMapping. "
            "Use IRMapping."
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
            self._node_to_mapper[node] = LayerNormMapper(
                source, copy.deepcopy(mod), name=node.name
            )
        elif isinstance(mod, nn.MultiheadAttention):
            self._convert_multihead_attention(node, mod)
        # Passthrough: no IR node needed
        elif isinstance(mod, self._PASSTHROUGH_MODULES):
            self._node_to_mapper[node] = source
        # Flatten: shape-only (ReshapeMapper, no ComputeOp)
        elif isinstance(mod, nn.Flatten):
            self._convert_flatten_module(node, source)
        else:
            # Generic: any other module → host-side ComputeOp
            in_shape = self._get_input_shape(node)
            input_shape = tuple(in_shape[1:]) if in_shape and len(in_shape) >= 2 else None
            out_shape = self._get_output_shape(node)
            output_shape = tuple(out_shape[1:]) if out_shape and len(out_shape) >= 2 else None
            mapper = ModuleComputeMapper(
                source, mod, input_shape=input_shape,
                output_shape=output_shape, name=node.name,
            )
            self._node_to_mapper[node] = mapper

    def _handle_call_function(self, node: fx.Node) -> None:
        fn = node.target

        # Structural ops: multi-input or shape-only — need dedicated handling
        if fn is operator.add or fn is torch.add:
            self._convert_add(node)
        elif fn is torch.flatten:
            self._convert_flatten_func(node)
        elif fn is operator.getitem:
            self._convert_getitem(node)
        elif fn is torch.cat:
            self._convert_cat(node)
        else:
            # Generic: wrap function call as ModuleComputeMapper
            source = self._get_source_mapper(node)
            wrapper = _FunctionWrapper.from_fx_node(node, fn)
            in_shape = self._get_input_shape(node)
            input_shape = tuple(in_shape[1:]) if in_shape and len(in_shape) >= 2 else None
            out_shape = self._get_output_shape(node)
            output_shape = tuple(out_shape[1:]) if out_shape and len(out_shape) >= 2 else None
            mapper = ModuleComputeMapper(
                source, wrapper, input_shape=input_shape,
                output_shape=output_shape, name=node.name,
            )
            self._node_to_mapper[node] = mapper

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
                # args: (self_node, d0, d1, ...) or single tuple arg
                raw = node.args[1:]
                if len(raw) == 1 and isinstance(raw[0], (tuple, list)):
                    dims = tuple(int(d) for d in raw[0])
                else:
                    dims = tuple(int(d) for d in raw)
            else:  # transpose(dim0, dim1)
                d0 = int(node.args[1]) if len(node.args) > 1 else 0
                d1 = int(node.args[2]) if len(node.args) > 2 else 1
                ndim = len(in_shape) if in_shape else 3
                dims = list(range(ndim))
                dims[d0], dims[d1] = dims[d1], dims[d0]
                dims = tuple(dims)
            # PermuteMapper handles both _forward_impl (true permute) and
            # _map/_map_to_ir (np.transpose for source-array reordering).
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
            # Extract the dim argument from args or kwargs; default to 1.
            if len(node.args) > 1:
                dim = node.args[1]
            else:
                dim = node.kwargs.get("dim", 1)
            if not isinstance(dim, int):
                dim = 1
            self._node_to_mapper[node] = MeanMapper(source, dim=dim)

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
        self._node_to_mapper[node] = _MultiInputModuleComputeMapper(
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
