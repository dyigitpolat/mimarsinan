"""Transformer/ViT conversion: LayerNorm, GELU, Dropout."""

from __future__ import annotations

import copy
import torch.fx as fx
import torch.nn as nn

from mimarsinan.mapping.mapping_utils import DropoutMapper, GELUMapper, LayerNormMapper


class TransformerConvertMixin:
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
