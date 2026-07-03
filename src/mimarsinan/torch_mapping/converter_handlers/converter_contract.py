"""Typing-only contract of the converter-host surface shared by the conversion mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityReport


class ConverterContract:
    """Runtime no-op base declaring the attributes/methods MapperGraphConverter provides to its mixins."""

    if TYPE_CHECKING:
        _modules: Dict[str, nn.Module]
        _node_to_mapper: Dict[fx.Node, Any]
        _node_to_attr: Dict[fx.Node, Any]
        _absorbed: Set[str]

        def _get_mapper(self, node: fx.Node) -> Any: ...

        def _get_source_mapper(self, node: fx.Node) -> Any: ...

        def _get_attr_value(self, node: fx.Node) -> Any: ...

        def _get_input_shape(self, node: fx.Node) -> Optional[Tuple[int, ...]]: ...

        def _get_output_shape(self, node: fx.Node) -> Optional[Tuple[int, ...]]: ...

        def _get_constant_tensor(self, node: fx.Node) -> Optional[torch.Tensor]: ...

        def _get_expanded_constant_tensor(self, node: fx.Node) -> Optional[torch.Tensor]: ...

        def _find_absorbed_follower(
            self,
            node: fx.Node,
            target_types: tuple,
            report: RepresentabilityReport,
            skip_bn: bool = False,
        ) -> Any: ...

        @staticmethod
        def _copy_bn_params(dst_bn: nn.Module, src_bn: nn.Module) -> None: ...
