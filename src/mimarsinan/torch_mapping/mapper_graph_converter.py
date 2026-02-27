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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.layers import LeakyGradReLU
from mimarsinan.mapping.mapping_utils import (
    AddMapper,
    AdaptiveAvgPool2DMapper,
    AvgPool2DMapper,
    ConcatMapper,
    Conv2DPerceptronMapper,
    DropoutMapper,
    EinopsRearrangeMapper,
    Ensure2DMapper,
    GELUMapper,
    InputMapper,
    LayerNormMapper,
    MaxPool2DMapper,
    MergeLeadingDimsMapper,
    ModelRepresentation,
    ModuleMapper,
    PerceptronMapper,
    ReshapeMapper,
)

from mimarsinan.torch_mapping.representability_analyzer import (
    RepresentabilityAnalyzer,
    RepresentabilityReport,
    RepresentabilityError,
)


class MapperGraphConverter:
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

    # ── Node handlers ────────────────────────────────────────────────────

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

    # ── Linear conversion ────────────────────────────────────────────────

    def _convert_linear(
        self,
        node: fx.Node,
        mod: nn.Linear,
        source,
        report: RepresentabilityReport,
    ) -> None:
        bn_mod = self._find_absorbed_follower(node, (nn.BatchNorm1d,), report)
        act_mod = self._find_absorbed_follower(
            node, (nn.ReLU, nn.LeakyReLU), report, skip_bn=True
        )

        normalization = copy.deepcopy(bn_mod) if bn_mod is not None else nn.Identity()

        perceptron = Perceptron(
            output_channels=mod.out_features,
            input_features=mod.in_features,
            bias=mod.bias is not None,
            normalization=normalization,
            name=node.name,
        )

        with torch.no_grad():
            perceptron.layer.weight.copy_(mod.weight.data)
            if mod.bias is not None:
                perceptron.layer.bias.copy_(mod.bias.data)

            if bn_mod is not None and not isinstance(normalization, nn.Identity):
                self._copy_bn_params(normalization, bn_mod)

        source = Ensure2DMapper(source)
        mapper = PerceptronMapper(source, perceptron)
        self._node_to_mapper[node] = mapper

    def _convert_conv2d(
        self,
        node: fx.Node,
        mod: nn.Conv2d,
        source,
        report: RepresentabilityReport,
    ) -> None:
        bn_mod = self._find_absorbed_follower(node, (nn.BatchNorm2d,), report)

        conv_mapper = Conv2DPerceptronMapper(
            source,
            in_channels=mod.in_channels,
            out_channels=mod.out_channels,
            kernel_size=mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            bias=mod.bias is not None,
            use_batchnorm=bn_mod is not None,
            name=node.name,
        )

        with torch.no_grad():
            flat_weight = mod.weight.data.reshape(mod.out_channels, -1)
            conv_mapper.perceptron.layer.weight.copy_(flat_weight)
            if mod.bias is not None:
                conv_mapper.perceptron.layer.bias.copy_(mod.bias.data)

            if bn_mod is not None:
                # Conv2DPerceptronMapper creates a LazyBatchNorm1d that hasn't
                # been initialised. Replace it with a concrete BN1d and copy
                # the trained BN2d params (per-channel, same shape).
                concrete_bn = nn.BatchNorm1d(mod.out_channels)
                self._copy_bn_params(concrete_bn, bn_mod)
                conv_mapper.perceptron.normalization = concrete_bn

        self._node_to_mapper[node] = conv_mapper

    def _convert_conv1d(
        self,
        node: fx.Node,
        mod: nn.Conv1d,
        source,
        report: RepresentabilityReport,
    ) -> None:
        from mimarsinan.mapping.mapping_utils import Conv1DPerceptronMapper

        bn_mod = self._find_absorbed_follower(node, (nn.BatchNorm1d,), report)

        conv_mapper = Conv1DPerceptronMapper(
            source,
            in_channels=mod.in_channels,
            out_channels=mod.out_channels,
            kernel_size=mod.kernel_size[0],
            stride=mod.stride[0],
            padding=mod.padding[0],
            dilation=mod.dilation[0],
            bias=mod.bias is not None,
            use_batchnorm=bn_mod is not None,
            name=node.name,
        )

        with torch.no_grad():
            flat_weight = mod.weight.data.reshape(mod.out_channels, -1)
            conv_mapper.perceptron.layer.weight.copy_(flat_weight)
            if mod.bias is not None:
                conv_mapper.perceptron.layer.bias.copy_(mod.bias.data)

            if bn_mod is not None:
                concrete_bn = nn.BatchNorm1d(mod.out_channels)
                self._copy_bn_params(concrete_bn, bn_mod)
                conv_mapper.perceptron.normalization = concrete_bn

        self._node_to_mapper[node] = conv_mapper

    # ── Pooling conversions ──────────────────────────────────────────────

    def _convert_maxpool2d(self, node: fx.Node, mod: nn.MaxPool2d, source) -> None:
        in_shape = self._get_input_shape(node)
        c, h, w = (in_shape[1], in_shape[2], in_shape[3]) if in_shape and len(in_shape) == 4 else (None, None, None)
        mapper = MaxPool2DMapper(
            source,
            kernel_size=mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            input_spatial_shape=(h, w) if h is not None else None,
            input_channels=c,
            name=node.name,
        )
        self._node_to_mapper[node] = mapper

    def _convert_avgpool2d(self, node: fx.Node, mod: nn.AvgPool2d, source) -> None:
        in_shape = self._get_input_shape(node)
        c, h, w = (in_shape[1], in_shape[2], in_shape[3]) if in_shape and len(in_shape) == 4 else (None, None, None)
        mapper = AvgPool2DMapper(
            source,
            kernel_size=mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            input_spatial_shape=(h, w) if h is not None else None,
            input_channels=c,
            name=node.name,
        )
        self._node_to_mapper[node] = mapper

    def _convert_adaptive_avgpool2d(self, node: fx.Node, mod: nn.AdaptiveAvgPool2d, source) -> None:
        in_shape = self._get_input_shape(node)
        c = in_shape[1] if in_shape and len(in_shape) == 4 else None
        mapper = AdaptiveAvgPool2DMapper(
            source,
            output_size=mod.output_size,
            input_channels=c,
            name=node.name,
        )
        self._node_to_mapper[node] = mapper

    def _convert_adaptive_avgpool2d_func(self, node: fx.Node) -> None:
        source = self._get_source_mapper(node)
        output_size = node.args[1] if len(node.args) > 1 else node.kwargs.get("output_size", (1, 1))
        in_shape = self._get_input_shape(node)
        c = in_shape[1] if in_shape and len(in_shape) == 4 else None
        mapper = AdaptiveAvgPool2DMapper(
            source,
            output_size=output_size,
            input_channels=c,
            name=node.name,
        )
        self._node_to_mapper[node] = mapper

    def _convert_maxpool2d_func(self, node: fx.Node) -> None:
        source = self._get_source_mapper(node)
        kernel_size = node.args[1] if len(node.args) > 1 else node.kwargs.get("kernel_size", 2)
        stride = node.args[2] if len(node.args) > 2 else node.kwargs.get("stride", None)
        padding = node.args[3] if len(node.args) > 3 else node.kwargs.get("padding", 0)
        in_shape = self._get_input_shape(node)
        c, h, w = (in_shape[1], in_shape[2], in_shape[3]) if in_shape and len(in_shape) == 4 else (None, None, None)
        mapper = MaxPool2DMapper(
            source,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            input_spatial_shape=(h, w) if h is not None else None,
            input_channels=c,
            name=node.name,
        )
        self._node_to_mapper[node] = mapper

    def _convert_avgpool2d_func(self, node: fx.Node) -> None:
        source = self._get_source_mapper(node)
        kernel_size = node.args[1] if len(node.args) > 1 else node.kwargs.get("kernel_size", 2)
        stride = node.args[2] if len(node.args) > 2 else node.kwargs.get("stride", None)
        padding = node.args[3] if len(node.args) > 3 else node.kwargs.get("padding", 0)
        in_shape = self._get_input_shape(node)
        c, h, w = (in_shape[1], in_shape[2], in_shape[3]) if in_shape and len(in_shape) == 4 else (None, None, None)
        mapper = AvgPool2DMapper(
            source,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            input_spatial_shape=(h, w) if h is not None else None,
            input_channels=c,
            name=node.name,
        )
        self._node_to_mapper[node] = mapper

    # ── Other conversions ────────────────────────────────────────────────

    def _convert_layer_norm(self, node: fx.Node, mod: nn.LayerNorm, source) -> None:
        ln_copy = copy.deepcopy(mod)
        mapper = LayerNormMapper(source, ln_copy, name=node.name)
        self._node_to_mapper[node] = mapper

    def _convert_layer_norm_func(self, node: fx.Node) -> None:
        source = self._get_source_mapper(node)
        self._node_to_mapper[node] = source

    def _convert_gelu(self, node: fx.Node, source) -> None:
        mapper = GELUMapper(source, name=node.name)
        self._node_to_mapper[node] = mapper

    def _convert_dropout(self, node: fx.Node, mod, source) -> None:
        p = getattr(mod, "p", 0.1)
        mapper = DropoutMapper(source, p=p, name=node.name)
        self._node_to_mapper[node] = mapper

    def _convert_add(self, node: fx.Node) -> None:
        if len(node.args) < 2:
            source = self._get_source_mapper(node)
            self._node_to_mapper[node] = source
            return

        a_node = node.args[0]
        b_node = node.args[1]

        a_mapper = self._get_mapper(a_node) if isinstance(a_node, fx.Node) else None
        b_mapper = self._get_mapper(b_node) if isinstance(b_node, fx.Node) else None

        if a_mapper is not None and b_mapper is not None:
            mapper = AddMapper(a_mapper, b_mapper)
            self._node_to_mapper[node] = mapper
        elif a_mapper is not None:
            self._node_to_mapper[node] = a_mapper
        elif b_mapper is not None:
            self._node_to_mapper[node] = b_mapper

    def _convert_flatten_func(self, node: fx.Node) -> None:
        source = self._get_source_mapper(node)
        out_shape = self._get_output_shape(node)
        if out_shape is not None and len(out_shape) == 2:
            flat_shape = (out_shape[-1],)
            mapper = ReshapeMapper(source, flat_shape)
        else:
            mapper = source
        self._node_to_mapper[node] = mapper

    def _convert_getitem(self, node: fx.Node) -> None:
        source = self._get_source_mapper(node)
        self._node_to_mapper[node] = source

    def _convert_cat(self, node: fx.Node) -> None:
        """Handle torch.cat: args[0] is a list of input nodes, args[1] is dim."""
        tensors_arg = node.args[0] if node.args else []
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 1)
        if not isinstance(tensors_arg, (list, tuple)):
            source = self._get_source_mapper(node)
            self._node_to_mapper[node] = source
            return
        source_mappers = []
        for arg in tensors_arg:
            if isinstance(arg, fx.Node):
                m = self._get_mapper(arg)
                if m is not None:
                    source_mappers.append(m)
        if not source_mappers:
            self._node_to_mapper[node] = None
            return
        if len(source_mappers) == 1:
            self._node_to_mapper[node] = source_mappers[0]
            return
        mapper = ConcatMapper(source_mappers, dim=dim, name=node.name)
        self._node_to_mapper[node] = mapper

    # ── Absorption helpers ───────────────────────────────────────────────

    def _propagate_absorbed(self, node: fx.Node) -> None:
        """For an absorbed node, point it at its source's mapper."""
        if len(node.args) >= 1 and isinstance(node.args[0], fx.Node):
            self._node_to_mapper[node] = self._get_mapper(node.args[0])

    def _find_absorbed_follower(
        self,
        node: fx.Node,
        target_types: Tuple[type, ...],
        report: RepresentabilityReport,
        skip_bn: bool = False,
    ) -> Optional[nn.Module]:
        """Find the first absorbed follower of ``node`` matching ``target_types``."""
        for user in node.users:
            if user.name in self._absorbed and user.op == "call_module":
                mod = self._modules.get(user.target)
                if mod is not None and isinstance(mod, target_types):
                    return mod
                if skip_bn and isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    return self._find_absorbed_follower(user, target_types, report)
        return None

    # ── Weight / parameter copy helpers ──────────────────────────────────

    @staticmethod
    def _copy_bn_params(dst_bn: nn.Module, src_bn: nn.Module) -> None:
        """Copy BatchNorm parameters from source to destination."""
        if isinstance(dst_bn, nn.Identity):
            return

        if hasattr(dst_bn, "weight") and hasattr(src_bn, "weight") and src_bn.weight is not None:
            if hasattr(dst_bn, "num_features"):
                pass
            dst_bn.weight.data.copy_(src_bn.weight.data)
        if hasattr(dst_bn, "bias") and hasattr(src_bn, "bias") and src_bn.bias is not None:
            dst_bn.bias.data.copy_(src_bn.bias.data)
        if hasattr(dst_bn, "running_mean") and hasattr(src_bn, "running_mean") and src_bn.running_mean is not None:
            dst_bn.running_mean.copy_(src_bn.running_mean)
        if hasattr(dst_bn, "running_var") and hasattr(src_bn, "running_var") and src_bn.running_var is not None:
            dst_bn.running_var.copy_(src_bn.running_var)
        if hasattr(dst_bn, "num_batches_tracked") and hasattr(src_bn, "num_batches_tracked"):
            dst_bn.num_batches_tracked.copy_(src_bn.num_batches_tracked)

    # ── Mapper / shape lookup helpers ────────────────────────────────────

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
