"""Operand blocks (N2): where a bound bank sits on a hard core, and its payload per sample."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence, Tuple

import numpy as np
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir.dynamic_bank import stage_bound_banks
from mimarsinan.models.nn.activations.integer_law import grid_codes, grid_levels


class OperandPlacementError(ToolchainRefusal, ValueError):
    """An operand bank placed where its per-sample matrix cannot be materialized (a fused core)."""


@dataclass(frozen=True)
class OperandBlock:
    """One placed region of a bound bank on a hard core: the core's block and the bank's rows and columns."""

    bank: int
    axon_offset: int
    neuron_offset: int
    axons: int
    neurons: int
    bank_axons: Tuple[int, int]
    bank_neurons: Tuple[int, int]

    @property
    def cells(self) -> int:
        return self.axons * self.neurons


def operand_blocks(hcm, bindings: Mapping) -> List[List[OperandBlock]]:
    """Per hard core of ``hcm``, its bound-bank regions (empty for a static core)."""
    out: List[List[OperandBlock]] = []
    for core, placements in zip(hcm.cores, hcm.soft_core_placements_per_hard_core):
        blocks = [OperandBlock(int(p["weight_bank_id"]), int(p["axon_offset"]), int(p["neuron_offset"]),
                               int(p["axons"]), int(p["neurons"]), tuple(p["bank_axon_range"]),
                               tuple(p["bank_neuron_range"]))
                  for p in placements if p.get("weight_bank_id") is not None and int(p["weight_bank_id"]) in bindings]
        if blocks and getattr(core, "fused_component_axons", None) is not None:
            raise OperandPlacementError("an operand bank landed on a fused hard core; operand banks are placed on "
                                        "whole cores")
        out.append(blocks)
    return out


def stage_banks(stage, bindings: Mapping) -> List[int]:
    """The bound banks a neural stage holds, sorted (``dynamic_bank.stage_bound_banks``, the one rule)."""
    return stage_bound_banks(stage, bindings)


def bank_payload(values: torch.Tensor, binding) -> torch.Tensor:
    """(batch, axons, neurons) of one bank from its write's flat output, in units of its payload step."""
    axons, neurons = binding.payload_shape
    flat = values.reshape(values.shape[0], -1)
    block = flat[:, binding.payload_offset: binding.payload_offset + axons * neurons].reshape(-1, axons, neurons)
    if binding.payload_step is None:
        return block
    step = torch.tensor(float(binding.payload_step), dtype=torch.float64, device=block.device)
    return grid_codes(block, step, where=f"operand bank {binding.bank_id}").to(block.dtype)


def cells_beyond_width(codes, payload_bits: int) -> int:
    """The cells of a payload whose |code| exceeds the positive levels of its ``payload_bits`` grid: a code a
    payload cell cannot hold, which the law's accumulator bound and ADC do not cover."""
    return int(np.count_nonzero(np.abs(np.asarray(codes, dtype=np.float64)) > grid_levels(payload_bits)))


def fill_operand_blocks(static: torch.Tensor, blocks: Sequence[OperandBlock],
                        payloads: Mapping[int, torch.Tensor]) -> torch.Tensor:
    """(batch, axons, neurons): the core's static grid with every operand block written per sample."""
    batch = next(iter(payloads.values())).shape[0] if payloads else 1
    weight = static.unsqueeze(0).expand(batch, -1, -1).clone()
    for block in blocks:
        rows, cols = slice(*block.bank_axons), slice(*block.bank_neurons)
        weight[:, block.axon_offset: block.axon_offset + block.axons,
               block.neuron_offset: block.neuron_offset + block.neurons] = payloads[block.bank][:, rows, cols]
    return weight


def static_part(weight: torch.Tensor, blocks: Sequence[OperandBlock]) -> torch.Tensor:
    """The core's grid with its operand blocks zeroed (a bank's placeholder never reaches an accumulator)."""
    static = weight.clone()
    for block in blocks:
        static[block.axon_offset: block.axon_offset + block.axons,
               block.neuron_offset: block.neuron_offset + block.neurons] = 0.0
    return static
