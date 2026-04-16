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

        # Register ALL Perceptron modules (including non-chip-supported ones) so
        # that .to(device), state_dict, parameters(), etc. work correctly.
        # get_perceptrons() only returns chip-targeted perceptrons for pipeline
        # steps, but ALL perceptrons need PyTorch module registration.
        mapper_repr._ensure_exec_graph()
        seen_perceptrons = set()
        all_perceptrons = []
        for node in mapper_repr._exec_order:
            p = getattr(node, "perceptron", None)
            if p is not None and id(p) not in seen_perceptrons:
                seen_perceptrons.add(id(p))
                all_perceptrons.append(p)
        self._perceptrons = nn.ModuleList(all_perceptrons)

        # Register every nn.Module node in the mapper graph as a proper
        # submodule. Without this, base nn.Module._apply does not reach them,
        # and the warmup forward can initialise a lazy module from a site
        # PyTorch considers "not installed as a submodule". Dedupe by id().
        registered_ids = {id(m) for m in self.modules()}
        idx = 0
        for node in mapper_repr._exec_order:
            if isinstance(node, nn.Module) and id(node) not in registered_ids:
                self.add_module(f"graph_node_{idx}", node)
                registered_ids.add(id(node))
                idx += 1

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

    def forward(self, x):
        return self._mapper_repr(x)
