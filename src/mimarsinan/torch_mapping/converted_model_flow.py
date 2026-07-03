"""PerceptronFlow subclass wrapping a Mapper DAG converted from a native PyTorch model."""

from __future__ import annotations

import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow
from mimarsinan.mapping.mapping_utils import ModelRepresentation


class ConvertedModelFlow(PerceptronFlow):
    """PerceptronFlow wrapping a Mapper DAG converted from a native PyTorch model."""

    def __init__(self, device, mapper_repr: ModelRepresentation):
        super().__init__(device)
        self.input_activation = nn.Identity()
        self._mapper_repr = mapper_repr

        # Register ALL perceptrons (not just chip-targeted get_perceptrons()) so .to(device)/state_dict()/parameters() reach every one.
        mapper_repr._ensure_exec_graph()
        seen_perceptrons = set()
        all_perceptrons = []
        for node in mapper_repr._exec_order:
            p = getattr(node, "perceptron", None)
            if p is not None and id(p) not in seen_perceptrons:
                seen_perceptrons.add(id(p))
                all_perceptrons.append(p)
        self._perceptrons = nn.ModuleList(all_perceptrons)

        # Register every nn.Module graph node as a submodule so _apply reaches them and lazy modules initialise from an installed site.
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

    def forward(self, x):
        return self._mapper_repr(x)
