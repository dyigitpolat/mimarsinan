"""PerceptronMapper, ComputeOpMapper, ModuleMapper."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.base import Mapper, resolve_activation_type
from mimarsinan.mapping.support.compute_modules import ScaleNormalizingWrapper
from mimarsinan.mapping.support.shape_probe import probe_module_io_shapes
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer


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
