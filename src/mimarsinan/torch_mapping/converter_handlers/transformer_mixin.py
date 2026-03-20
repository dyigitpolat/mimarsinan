"""Transformer/ViT conversion: F.layer_norm function handling."""

from __future__ import annotations

import torch.fx as fx


class TransformerConvertMixin:
    def _convert_layer_norm_func(self, node: fx.Node) -> None:
        """F.layer_norm function call → passthrough (shape-only)."""
        source = self._get_source_mapper(node)
        self._node_to_mapper[node] = source
