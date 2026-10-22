"""The probe's native reference of a decomposed program: the user's own model, on the witness program's tensors.

The witness program is the decomposed model's traced graph, whose projection parameters view rows of the leaf's
(``Decomposition.stacked``); the reference is the user's model itself (``NativeFp64``), each leaf parameter the
stacking of the witness tensors that view it, so the certificate compares the conversion with the native leaves.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn as nn

from mimarsinan.torch_mapping.value_parity.witness import NativeFp64, native_reference


def _named_tensors(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {**dict(module.named_parameters(remove_duplicate=False)),
            **dict(module.named_buffers(remove_duplicate=False))}


def stacked_state(state: Mapping[str, torch.Tensor], stacked: Mapping[str, List[str]]) -> Dict[str, torch.Tensor]:
    """``state`` with every stacked leaf parameter rebuilt from the tensors that view its rows."""
    found = dict(state)
    for name, parts in stacked.items():
        found[name] = torch.cat([state[part] for part in parts], dim=0)
    return found


def decomposition_reference(model: nn.Module, stacked: Mapping[str, List[str]]):
    """``native_reference(model)`` where nothing is stacked, else the user's model on the stacked witness state."""
    if not stacked:
        return native_reference(model)

    def build(program: nn.Module, inputs: Sequence[torch.Tensor]) -> nn.Module:
        return NativeFp64(model, stacked_state(_named_tensors(program), stacked))
    return build
