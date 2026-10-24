"""Per-sample operand payloads (N2): a bound bank's matrix, materialized from its payload write per sample."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np
import torch

from mimarsinan.chip_simulation.value_run.int64_golden import GoldenTrace, run_golden
from mimarsinan.chip_simulation.value_run.operand_blocks import bank_payload, stage_banks


def materialize(stage, state_buffer: Mapping[int, torch.Tensor], bindings: Mapping,
                sample_slice: slice = slice(None)) -> Dict[int, torch.Tensor]:
    """{bank: (batch, axons, neurons)} for every bound bank of ``stage``, from the state buffer's payload writes."""
    return {bank: bank_payload(state_buffer[int(bindings[bank].source_node_id)][sample_slice], bindings[bank])
            for bank in stage_banks(stage, bindings)}


def payload_codes(program, x) -> Dict[Tuple[int, int], np.ndarray]:
    """{(sample, bank): int64 payload codes} the int64 golden writes into every bound bank of ``program``."""
    trace = GoldenTrace()
    run_golden(program, x, trace=trace)
    return {(sample, bank): codes[sample] for bank, codes in trace.payloads.items() for sample in range(codes.shape[0])}
