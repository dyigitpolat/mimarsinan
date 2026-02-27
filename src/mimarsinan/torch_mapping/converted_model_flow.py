"""
PerceptronFlow subclass wrapping a Mapper DAG converted from a native
PyTorch model.
"""

from __future__ import annotations

import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow
from mimarsinan.mapping.mapping_utils import ModelRepresentation, ModuleMapper


class ConvertedModelFlow(PerceptronFlow):
    """PerceptronFlow wrapping a Mapper DAG converted from a native PyTorch model.

    This is structurally identical to ``VGG16Mapper`` or ``VisionTransformer``
    but is constructed programmatically by ``MapperGraphConverter`` rather
    than hand-coded.
    """

    def __init__(self, device, mapper_repr: ModelRepresentation):
        super().__init__(device)
        self.input_activation = nn.Identity()
        self._mapper_repr = mapper_repr

        # Register all Perceptron modules so that .to(device), state_dict,
        # parameters(), etc. work correctly.
        self._perceptrons = nn.ModuleList(mapper_repr.get_perceptrons())

        # Also register any Conv/Pool mapper modules that own nn.Modules
        # (e.g. Conv2DPerceptronMapper owns a Perceptron registered above,
        #  but we keep this for completeness).

    def get_perceptrons(self):
        return self._mapper_repr.get_perceptrons()

    def get_perceptron_groups(self):
        return self._mapper_repr.get_perceptron_groups()

    def get_mapper_repr(self):
        return self._mapper_repr

    def get_input_activation(self):
        return self.input_activation

    def set_input_activation(self, activation):
        self.input_activation = activation
        # If the mapper graph has a ModuleMapper for input activation at the
        # root, update it. This mirrors VGG16Mapper/VisionTransformer behavior.

    def _apply(self, fn):
        super()._apply(fn)
        self._mapper_repr._ensure_exec_graph()
        for node in self._mapper_repr._exec_order:
            if isinstance(node, nn.Module):
                node._apply(fn)
        return self

    def forward(self, x):
        return self._mapper_repr(x)
