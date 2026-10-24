"""N2 design §2.3 and §3.1: every payload lies on a grid of its operand core type's ``payload_bits``, so the law's
accumulator bound (axons x payload levels x input levels) and the ADC it sizes hold by a checked statement.

A product whose stationary operand is another product's output (linear attention) takes the declared width by a
re-snap in its write; a payload on a producer grid of another width refuses by name where its step is bound; a
payload code beyond its width refuses by name in the int64 golden and fails the payload edge."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.certification.operand_certificate import certify_operand_program, certify_payload_cells
from mimarsinan.chip_simulation.value_run import int64_golden
from mimarsinan.chip_simulation.value_run.int64_golden import GoldenGridError, GoldenTrace, run_golden
from mimarsinan.mapping.export import operand_laws
from mimarsinan.mapping.ir.dynamic_bank import OperandBindingError
from mimarsinan.mapping.mappers.operand import OperandContractionMapper
from mimarsinan.models.nn.activations.integer_law import grid_levels
from mimarsinan.pipelining.pipeline_steps.quantization import readout_law_step
from unit.chip_simulation.value_run.operand_programs import (
    attention_program,
    build_operand_program,
    image_inputs,
    operand_platform,
)

T, F_IN = 6, 8
GEOMETRY = dict(axons=64, neurons=64, count=3, passes=128)


class LinearAttention(nn.Module):
    """phi(Q) (phi(K)^T V): the outer product's stationary operand is the inner product's output.

    phi is ReLU + 1 (the +1 a host op), not the review's ELU + 1: torch's fp64 ELU kernel returns other last bits
    for a sample alone than inside a batch (1 ulp on a few elements), which the certificate's batch-of-one edge
    reports as a mismatch, so an ELU program is not certified per sample (fix round report)."""

    def __init__(self):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(F_IN, 4), nn.Linear(F_IN, 4), nn.Linear(F_IN, 4)
        self.head = nn.Linear(T * 4, 10)

    def forward(self, x):
        q, k = F.relu(self.q(x)) + 1.0, F.relu(self.k(x)) + 1.0
        return self.head(torch.flatten(q @ (k.transpose(1, 2) @ self.v(x)), 1))


def _linear_attention(payload_bits: int):
    torch.manual_seed(0)
    return build_operand_program(LinearAttention(), (T, F_IN), num_classes=10,
                                 platform=operand_platform(payload_bits=payload_bits))


def _inputs(batch: int = 16) -> torch.Tensor:
    return torch.randn((batch, T, F_IN), generator=torch.Generator().manual_seed(4), dtype=torch.float64)


def _payload_extremes(program, x) -> dict:
    trace = GoldenTrace()
    run_golden(program.identity_mapping(), x.numpy(), trace=trace)
    return {bank: int(np.abs(codes).max()) for bank, codes in trace.payloads.items()}


def test_a_product_chain_writes_every_payload_at_the_declared_width():
    program = _linear_attention(payload_bits=4)
    x = _inputs()
    extremes = _payload_extremes(program, x)
    assert len(extremes) == 2 and max(extremes.values()) <= grid_levels(4), extremes
    outer = [m for m in program.model.get_mapper_repr().execution_order() if isinstance(m, OperandContractionMapper)]
    assert any(op.payload_quantizer is not None for m in outer for op in m.banks)  # the chained payload re-snaps
    certificate = certify_operand_program(program.nf(), program.identity_mapping(),
                                          program.packed_mapping(axons=64, neurons=64, count=2, passes=128), x)
    assert certificate.passed, certificate.summary()
    assert certificate.payload.beyond_width == 0


def test_a_product_chain_at_equal_widths_copies_its_producer_codes():
    program = _linear_attention(payload_bits=8)
    mappers = [m for m in program.model.get_mapper_repr().execution_order() if isinstance(m, OperandContractionMapper)]
    assert all(op.payload_quantizer is None for m in mappers for op in m.banks)
    assert max(_payload_extremes(program, _inputs()).values()) <= grid_levels(8)


def test_a_payload_on_a_producer_grid_of_another_width_refuses_by_name(monkeypatch):
    """Mutant of the partition: no group is marked as a payload, so K and V leave qkv on the 8-bit grid while the
    operand core type declares 4 payload bits."""
    original = readout_law_step.consumer_group_partition

    def unmarked(model_repr):
        return {key: type(groups)(channels=groups.channels, payload=tuple(False for _ in groups.payload))
                for key, groups in original(model_repr).items()}

    monkeypatch.setattr(readout_law_step, "consumer_group_partition", unmarked)
    with pytest.raises(OperandBindingError, match="payload_bits 4"):
        attention_program("softmax", platform=operand_platform(payload_bits=4))


def _halved_step_program(monkeypatch):
    """Mutant of the step binding: every payload step halved, so the payload codes double past their width."""
    original = operand_laws._payload_step
    monkeypatch.setattr(operand_laws, "_payload_step", lambda *a, **k: original(*a, **k) / 2.0)
    program = attention_program("softmax")
    monkeypatch.setattr(operand_laws, "_payload_step", original)
    return program


def test_a_payload_beyond_its_width_refuses_by_name_in_the_golden(monkeypatch):
    program = _halved_step_program(monkeypatch)
    with pytest.raises(GoldenGridError, match="beyond its payload_bits 8"):
        run_golden(program.identity_mapping(), image_inputs(8).numpy())


def test_the_payload_edge_fails_a_payload_beyond_its_width_without_the_goldens_refusal(monkeypatch):
    program = _halved_step_program(monkeypatch)
    monkeypatch.setattr(int64_golden, "cells_beyond_width", lambda codes, bits: 0)
    certificate = certify_payload_cells(program.packed(**GEOMETRY), program.identity_mapping(), image_inputs(8))
    assert certificate.beyond_width > 0 and certificate.mismatched >= certificate.beyond_width
    assert not certificate.passed
