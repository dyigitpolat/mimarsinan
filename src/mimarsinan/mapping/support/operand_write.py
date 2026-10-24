"""OperandWrite: the host op that writes the payloads of one product's dynamic operand banks (N2)."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.models.nn.activations.value_quantizer import quantize_to_value_grid


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


class OperandWrite(ComputeAdapter):
    """Gathers the stationary operand of every bank of one product (its input arrives in payload layout: bank,
    axon, neuron) and applies the payload law: the identity on codes when each bank reads whole producer groups,
    else the re-snap of each bank onto its own payload grid (``payload_quantizers``, the operand perceptrons'
    boundary-rule grids, shared by reference so the model and the program read one grid)."""

    @property
    def display_name(self) -> str:
        return "OperandWrite"

    def __init__(self, *, banks: int, payload_shape: Tuple[int, int],
                 payload_quantizers: Optional[Sequence[nn.Module]] = None) -> None:
        super().__init__(_identity)
        self.banks = int(banks)
        self.payload_shape = (int(payload_shape[0]), int(payload_shape[1]))
        self._quantizers: List[Any] = list(payload_quantizers or ())
        if self._quantizers and len(self._quantizers) != self.banks:
            raise ValueError(f"OperandWrite: {len(self._quantizers)} payload grids for {self.banks} banks")

    def payload_cells(self) -> int:
        return self.banks * self.payload_shape[0] * self.payload_shape[1]

    def forward_macs(self) -> int:
        return 0

    def forward(self, *inputs, **_kwargs) -> torch.Tensor:
        (x,) = inputs
        if not self._quantizers:
            return x
        flat = x.reshape(x.shape[0], self.banks, -1)
        banks = [quantize_to_value_grid(flat[:, b], q.scale.to(flat.dtype), int(q.activation_bits))
                 for b, q in enumerate(self._quantizers)]
        return torch.stack(banks, dim=1).reshape(x.shape)
