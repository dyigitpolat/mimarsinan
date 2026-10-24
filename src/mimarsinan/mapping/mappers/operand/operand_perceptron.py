"""The model side of one operand bank: its readout, its entry grid and its payload grid (N2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.models.nn.activations.value_quantizer import ValueGridQuantizer


@dataclass(eq=False)
class OperandStamp:
    """A bank's solved law per output column (the ``CoreLaw`` fields), its streamed grid step and its payload step."""

    law: Dict[str, np.ndarray]
    input_step: float
    payload_step: float

    def on(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return {k: torch.tensor(np.array(v), dtype=torch.float64, device=device) for k, v in self.law.items()}


class OperandPerceptron(nn.Module):
    """Duck-types a perceptron's two quantizer slots for the steps that install and calibrate them: ``scaler``
    holds the bank's ``ReadoutQuantizer``, ``input_activation`` the entry grid of a host-fed streamed operand;
    ``payload_quantizer`` re-snaps the stationary operand when the payload is not on its producer's grid."""

    #: The batched channel axis of the bank's output, read by the readout calibration (the neuron axis).
    output_channel_axis = -1
    name: str
    batch_index: Tuple[int, ...]
    payload_bits: int
    streamed_host_fed: bool
    input_activation: nn.Module
    scaler: nn.Module
    payload_quantizer: Optional[ValueGridQuantizer]
    perceptron_index: Optional[int]
    stamp: Optional[OperandStamp]

    def __init__(self, name: str, *, batch_index: Tuple[int, ...], payload_bits: int, resnap: bool,
                 streamed_host_fed: bool) -> None:
        super().__init__()
        self.name = str(name)
        self.batch_index = tuple(int(i) for i in batch_index)
        self.payload_bits = int(payload_bits)
        self.streamed_host_fed = bool(streamed_host_fed)
        self.input_activation = nn.Identity()
        self.scaler = nn.Identity()
        self.payload_quantizer = ValueGridQuantizer(payload_bits) if resnap else None
        self.perceptron_index = None
        self.stamp = None

    def append_input_wire_op(self, module: nn.Module) -> None:
        if isinstance(self.input_activation, nn.Identity):
            self.input_activation = module
        else:
            self.input_activation = nn.Sequential(self.input_activation, module)

    def set_scaler(self, scaler: nn.Module) -> None:
        self.scaler = scaler

    def streamed(self, x: torch.Tensor) -> torch.Tensor:
        return x if isinstance(self.input_activation, nn.Identity) else self.input_activation(x)

    def payload(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.payload_quantizer is None else self.payload_quantizer(x)

    def readout(self, y: torch.Tensor) -> torch.Tensor:
        return y if isinstance(self.scaler, nn.Identity) else self.scaler(y)
