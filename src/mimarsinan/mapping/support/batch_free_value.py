"""Batch-free values (coverage W6): a host operand computed from the model's state and literals at every call."""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn


class BatchFreeValue(nn.Module):
    """A bound operand computed at every call by ``program`` (the traced operators that compute it natively, with no
    argument: it reads no input) from its own state, so gradients reach that state; returned in the stored form of a
    bound tensor, ``shape`` (one sample's slice: a leading singleton axis stripped, a broadcast class token's slice).
    The program runs in eval mode, the converter's semantics, whatever the mode of the model holding it."""

    def __init__(self, program: nn.Module, shape: Sequence[int]) -> None:
        super().__init__()
        self.program = program.eval()
        self.shape = tuple(int(d) for d in shape)

    def train(self, mode: bool = True) -> "BatchFreeValue":
        super().train(mode)
        self.program.eval()
        return self

    def forward(self) -> torch.Tensor:
        return self.program().broadcast_to((1, *self.shape))[0]


def deployed_payload(module: nn.Module) -> nn.Module:
    """``module`` as the deployed program holds it (IR lowering): a ``ComputeAdapter`` with a ``BatchFreeValue`` slot
    gets a copy whose every such slot is its value now, a frozen bound tensor (a constant of the program, as every
    bound tensor is); any other module is itself."""
    slots = [name for name, child in module.named_children()
             if isinstance(child, BatchFreeValue) and name.startswith("bound_")]
    if not slots or not hasattr(module, "_bound_count"):
        return module
    frozen = copy.copy(module)
    frozen._parameters = OrderedDict(module._parameters)
    frozen._modules = OrderedDict(module._modules)
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        for name in slots:
            value = frozen._modules.pop(name)
            assert isinstance(value, BatchFreeValue)
            frozen.register_parameter(name, nn.Parameter(value().detach().clone(), requires_grad=False))
    return frozen
