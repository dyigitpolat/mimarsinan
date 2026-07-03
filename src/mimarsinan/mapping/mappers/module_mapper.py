"""ModuleMapper: forward-only module wrapper (identity in IR mapping)."""

from __future__ import annotations

import torch.nn as nn

from mimarsinan.mapping.mappers.base import Mapper


class ModuleMapper(Mapper):
    """Forward-only module; identity in IR mapping."""

    def __init__(self, source_mapper, module: nn.Module):
        super(ModuleMapper, self).__init__(source_mapper)
        self.module = module

    def _map_to_ir(self, ir_mapping):
        return self.source_mapper.map_to_ir(ir_mapping)

    def _forward_impl(self, x):
        return self.module(x)
