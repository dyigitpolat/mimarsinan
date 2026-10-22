"""Typing-only contract of the converter-host surface shared by the conversion mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.mapping.platform.packaging_contract import PackagingContract
from mimarsinan.mapping.support.call_template import CallPartition
from mimarsinan.torch_mapping.converter_handlers.batch_layout import BatchLayouts
from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityReport


class ConverterContract:
    """Runtime no-op base declaring the attributes/methods MapperGraphConverter provides to its mixins."""

    if TYPE_CHECKING:
        gm: fx.GraphModule
        _modules: Dict[str, nn.Module]
        _node_to_mapper: Dict[fx.Node, Any]
        _node_to_attr: Dict[fx.Node, Any]
        _shape_values: Dict[fx.Node, Any]
        _batch_layouts: BatchLayouts
        _absorbed: Set[str]
        _packaging: PackagingContract

        def _get_mapper(self, node: fx.Node) -> Any: ...

        def _get_source_mapper(self, node: fx.Node) -> Any: ...

        def _get_attr_value(self, node: fx.Node) -> Any: ...

        def _get_input_shape(self, node: fx.Node) -> Optional[Tuple[int, ...]]: ...

        def _get_output_shape(self, node: fx.Node) -> Optional[Tuple[int, ...]]: ...

        def _get_constant_tensor(self, node: fx.Node) -> Optional[torch.Tensor]: ...

        def _get_expanded_constant_tensor(self, node: fx.Node) -> Optional[torch.Tensor]: ...

        def _static_shape_value(self, node: fx.Node) -> Tuple[bool, Any]: ...

        def _static_dim(self, node: fx.Node, value: Any, refusal: Callable[[fx.Node, str], Exception]) -> Any: ...

        def _layout_violation(self, node: fx.Node) -> Optional[str]: ...

        def _require_batch_leading(self, node: fx.Node) -> None: ...

        def _native_operands(self, node: fx.Node, operands: Optional[list] = None) -> None: ...

        def _native_static_operand(self, node: fx.Node) -> None: ...

        def _native_layout(self, node: fx.Node, op: Any) -> bool: ...

        def _require_restored(self, node: fx.Node) -> None: ...

        def _require_static_reads(self) -> None: ...

        def _emit_generic_compute_op(self, node: fx.Node, fn: Any) -> None: ...

        def _partition_fx_args(self, node: fx.Node, fn: Any = None) -> CallPartition: ...

        def _propagate_absorbed(self, node: fx.Node) -> None: ...

        def _convert_flatten_module(self, node: fx.Node, source: Any) -> None: ...

        def _convert_linear(self, node: fx.Node, mod: nn.Linear, source: Any, report: RepresentabilityReport) -> None: ...

        def _convert_conv1d(self, node: fx.Node, mod: nn.Conv1d, source: Any, report: RepresentabilityReport) -> None: ...

        def _convert_conv2d(self, node: fx.Node, mod: nn.Conv2d, source: Any, report: RepresentabilityReport) -> None: ...

        def _find_absorbed_follower(
            self,
            node: fx.Node,
            target_types: tuple,
            report: RepresentabilityReport,
            skip_bn: bool = False,
        ) -> Any: ...

        @staticmethod
        def _perceptron_normalization(bn: nn.Module) -> nn.BatchNorm1d: ...
