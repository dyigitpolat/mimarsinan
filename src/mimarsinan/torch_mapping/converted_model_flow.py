"""PerceptronFlow subclass wrapping a Mapper DAG converted from a native PyTorch model."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn

from mimarsinan.models.nn.adaptive_chunks import forward_adaptive_chunks
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow
from mimarsinan.mapping.mapping_utils import ModelRepresentation
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import Impurity
from mimarsinan.torch_mapping.value_parity.certificate import ConversionCertificate, certificate_problem


class ConvertedModelFlow(PerceptronFlow):
    """PerceptronFlow wrapping a Mapper DAG converted from a native PyTorch model."""

    # Stamped by convert_torch_model at flow birth; flows pickled before the stamp existed read None.
    conversion_certificate: ConversionCertificate | None = None
    # The shape values folded into conversion-time literals (FX node -> operator, literal), set at conversion.
    shape_literals: Mapping[str, Tuple[str, str]] = MappingProxyType({})
    # What those literals are proved by (``ShapeValueMixin.shape_proof``), set at conversion; None before.
    shape_proof: Optional[str] = None
    # The first node that makes the traced program impure (``symbolic/purity.py``), set at conversion; None for a pure
    # program (and before the field): stage 2 refuses an impure program's value program by name.
    shape_impurity: Optional[Impurity] = None
    # The first opaque batch value of a pure program (FX node, operator, why; review round 14), set at conversion; None
    # when every batch-dependent value is in the affine batch domain: stage 2 refuses the value program by name.
    shape_opaque: Optional[Tuple[str, str, str]] = None

    def __init__(self, device, mapper_repr: ModelRepresentation):
        super().__init__(device)
        self.input_activation = nn.Identity()
        self._mapper_repr = mapper_repr

        # Register ALL perceptrons (not just chip-targeted get_perceptrons()) so .to(device)/state_dict()/parameters() reach every one.
        mapper_repr._ensure_exec_graph()
        exec_order = mapper_repr._exec_order
        assert exec_order is not None  # populated by _ensure_exec_graph
        seen_perceptrons = set()
        all_perceptrons = []
        for node in exec_order:
            p = getattr(node, "perceptron", None)
            if p is not None and id(p) not in seen_perceptrons:
                seen_perceptrons.add(id(p))
                all_perceptrons.append(p)
        self._perceptrons = nn.ModuleList(all_perceptrons)

        # Register every nn.Module graph node as a submodule so _apply reaches them and lazy modules initialise from an installed site.
        registered_ids = {id(m) for m in self.modules()}
        idx = 0
        for node in exec_order:
            if isinstance(node, nn.Module) and id(node) not in registered_ids:
                self.add_module(f"graph_node_{idx}", node)
                registered_ids.add(id(node))
                idx += 1

    def conversion_certificate_problem(self) -> str | None:
        """Why this flow may not be used as a certified conversion (``None``: current and certified)."""
        return certificate_problem(self.conversion_certificate)

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
        # [C4'] grad-free forwards chunk adaptively on CUDA OOM: an installed
        # cycle-accurate activation materializes S x batch x features, so a
        # val batch sized for the fp32 loaders can exceed VRAM at eval time.
        if not torch.is_grad_enabled():
            return forward_adaptive_chunks(self._mapper_repr, x)
        return self._mapper_repr(x)
