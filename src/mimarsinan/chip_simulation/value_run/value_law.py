"""The integer readout law on a packed core: codes on its rows' DAC grids, an exact accumulate, one law."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from mimarsinan.models.nn.activations.integer_law import IntegerLawRangeError, apply_law, grid_codes


def core_row_steps(hard_core) -> np.ndarray:
    """Per axon row, the DAC step of the block it feeds; inf on rows with no data (off, bias, unused)."""
    law = hard_core.readout_law
    steps = np.full(int(hard_core.axons_per_core), np.inf)
    if hard_core.core_matrix is not None or not hard_core.matrix_placements:
        raise ValueError("a law-carrying hard core needs matrix placements to place its input grids")
    for placement in hard_core.matrix_placements:
        block = law.input_step[placement.neuron_offset: placement.neuron_offset + placement.neurons]
        if block.size and not np.all(block == block[0]):
            raise ValueError("one placed soft core reads one input grid")
        steps[placement.axon_offset: placement.axon_offset + placement.axons] = block[0] if block.size else np.inf
    for row, source in enumerate(hard_core.axon_sources):
        if source.is_off_ or source.is_always_on_:
            steps[row] = np.inf
    return steps


@dataclass
class PreparedLaw:
    """A core's law as fp64 tensors on the execution device."""

    name: str
    row_steps: torch.Tensor
    weight: torch.Tensor
    columns: Dict[str, torch.Tensor]


def prepare_core_law(hard_core, weight: torch.Tensor, index: int) -> Optional[PreparedLaw]:
    law = getattr(hard_core, "readout_law", None)
    if law is None:
        return None
    device = weight.device

    def t(values):
        return torch.as_tensor(np.array(values, dtype=np.float64), device=device)

    names = ("m0", "shift", "bias", "lo", "hi", "output_step", "adc_shift", "adc_levels")
    return PreparedLaw(
        name=f"hard core {index}", row_steps=t(core_row_steps(hard_core)),
        weight=weight.to(torch.float64), columns={n: t(getattr(law, n)).reshape(1, -1) for n in names},
    )


def run_core_law(signal: torch.Tensor, law: PreparedLaw) -> torch.Tensor:
    """``y Δout`` in fp64: the integer codes of ``signal``, their exact accumulator, the stamped law."""
    codes = grid_codes(signal, law.row_steps, where=law.name)
    acc = codes @ law.weight
    if not bool(torch.equal(acc, torch.round(acc))):
        raise IntegerLawRangeError(f"{law.name}: the fp64 accumulator is not integral")
    c = law.columns
    y = apply_law(acc, m0=c["m0"], bias=c["bias"], shift=c["shift"], lo=c["lo"], hi=c["hi"],
                  adc_shift=c["adc_shift"], adc_levels=c["adc_levels"])
    return y * c["output_step"]
