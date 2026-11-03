"""The integer readout law on a packed core: codes on its rows' DAC grids, an exact accumulate, one law."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from mimarsinan.mapping.ir.deployment_dtype import VALUE_ARITHMETIC
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


def core_row_offsets(hard_core) -> Optional[np.ndarray]:
    """Per axon row, the offset of the declared entry grid its host value was snapped on (``None``: no offset)."""
    grid = getattr(hard_core, "boundary_grid", None)
    offset = 0.0 if grid is None or not grid.armed else grid.entry_offset()
    if offset == 0.0:
        return None
    rows = np.zeros(int(hard_core.axons_per_core))
    for row, source in enumerate(hard_core.axon_sources):
        if source.is_input_ and not (source.is_off_ or source.is_always_on_):
            rows[row] = offset
    return rows


def lane_rows(hard_core) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A lane law's per-row decode (WI-5e): ``(steps, offsets, operand row per column)``. Every placed soft core's
    rows read its accumulator grid (step and offset), its operand rows their operand's grid; off and always-on rows
    carry no code (step inf)."""
    law = hard_core.readout_law
    steps = core_row_steps(hard_core)
    offsets = np.zeros(int(hard_core.axons_per_core))
    for placement in hard_core.matrix_placements:
        block = law.input_offset[placement.neuron_offset: placement.neuron_offset + placement.neurons]
        offsets[placement.axon_offset: placement.axon_offset + placement.axons] = block[0] if block.size else 0.0
    rows = np.asarray(law.operand_row)
    present = rows >= 0
    steps[rows[present]] = law.operand_step[present]
    offsets[rows[present]] = law.operand_offset[present]
    for row, source in enumerate(hard_core.axon_sources):
        if source.is_off_ or source.is_always_on_:
            steps[row], offsets[row] = np.inf, 0.0
    return steps, offsets, rows


@dataclass
class PreparedLaw:
    """A core's law as fp64 tensors on the execution device."""

    name: str
    row_steps: torch.Tensor
    weight: torch.Tensor
    columns: Dict[str, torch.Tensor]
    row_offsets: Optional[torch.Tensor] = None
    #: A lane law's operand: the row per column (``-1``: none) and the columns that have one.
    operand_rows: Optional[torch.Tensor] = None


def prepare_core_law(hard_core, weight: torch.Tensor, index: int) -> Optional[PreparedLaw]:
    law = getattr(hard_core, "readout_law", None)
    if law is None:
        return None
    device = weight.device

    def t(values):
        return torch.as_tensor(np.array(values, dtype=VALUE_ARITHMETIC.core_np), device=device)

    names = ("m0", "shift", "bias", "lo", "hi", "output_step", "adc_shift", "adc_levels")
    if hasattr(law, "operand_row"):
        steps, offsets, rows = lane_rows(hard_core)
        lane = ("operand_m", "output_offset")
        return PreparedLaw(
            name=f"hard core {index}", row_steps=t(steps), weight=weight,
            columns={n: t(getattr(law, n)).reshape(1, -1) for n in names + lane},
            row_offsets=t(offsets) if bool((offsets != 0).any()) else None,
            operand_rows=torch.as_tensor(np.array(rows, dtype=np.int64), device=device),
        )
    offsets = core_row_offsets(hard_core)
    return PreparedLaw(
        name=f"hard core {index}", row_steps=t(core_row_steps(hard_core)),
        weight=weight.to(VALUE_ARITHMETIC.core_dtype), columns={n: t(getattr(law, n)).reshape(1, -1) for n in names},
        row_offsets=None if offsets is None else t(offsets),
    )


def run_core_law(signal: torch.Tensor, law: PreparedLaw, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``y Δout`` in fp64: the integer codes of ``signal``, their exact accumulator, the stamped law (on a lane, the
    operand's code and the output offset too: ``y Δout - o_out``). ``weight`` (batch, axons, neurons) replaces the
    stamped grid for a core holding a per-sample operand payload (N2)."""
    codes = grid_codes(signal, law.row_steps, where=law.name, offset=law.row_offsets)
    if weight is None:
        acc = codes @ law.weight.to(codes.dtype)  # integer codes: exact in fp64 below 2^53
    else:
        acc = torch.bmm(codes.unsqueeze(1), weight.to(VALUE_ARITHMETIC.core_dtype)).squeeze(1)
    if not bool(torch.equal(acc, torch.round(acc))):
        raise IntegerLawRangeError(f"{law.name}: the fp64 accumulator is not integral")
    c = law.columns
    operands = ()
    if law.operand_rows is not None:
        rows = law.operand_rows
        q = torch.where(rows >= 0, codes[:, rows.clamp(min=0)], torch.zeros((), dtype=codes.dtype))
        operands = ((q, c["operand_m"]),)
    y = apply_law(acc, m0=c["m0"], bias=c["bias"], shift=c["shift"], lo=c["lo"], hi=c["hi"],
                  adc_shift=c["adc_shift"], adc_levels=c["adc_levels"], operands=operands)
    return y * c["output_step"] - c["output_offset"] if "output_offset" in c else y * c["output_step"]
