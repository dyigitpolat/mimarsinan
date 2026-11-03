"""N2 design §2.4 and §3.1: an operand bank's matrix is materialized per sample from its payload write; the bound
core's output is the integer law of the exact integer product; a payload cached across samples is caught."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.certification.operand_certificate import certify_payload_cells
from mimarsinan.chip_simulation.value_run import operand_payloads
from mimarsinan.chip_simulation.value_run.int64_golden import GoldenTrace, run_golden
from mimarsinan.chip_simulation.value_run.operand_payloads import materialize, payload_codes
from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.models.nn.activations.integer_law import apply_law_int64
from unit.chip_simulation.value_run.integer_programs import assert_exact
from unit.chip_simulation.value_run.operand_programs import build_operand_program

T, F_IN, K = 6, 8, 5


class Scores(nn.Module):
    """One product: A = x W_a (T x K) streamed, B = (x W_b)^T (K x T) written as the payload."""

    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(F_IN, K), nn.Linear(F_IN, K)
        self.head = nn.Linear(T * T, 4)

    def forward(self, x):
        return self.head(torch.flatten(torch.bmm(self.a(x), self.b(x).transpose(1, 2)), 1))


@pytest.fixture(scope="module")
def program():
    torch.manual_seed(2)
    return build_operand_program(Scores(), (T, F_IN), num_classes=4)


def _inputs(batch=8, seed=4):
    return torch.randn((batch, T, F_IN), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _neural_stage_with_bank(mapping):
    bank = next(iter(mapping.dynamic_bank_bindings))
    for index, stage in enumerate(mapping.stages):
        if stage.kind == "neural" and any(
                p.get("weight_bank_id") == bank
                for placements in stage.hard_core_mapping.soft_core_placements_per_hard_core for p in placements):
            return index, stage, bank
    raise AssertionError("no neural stage holds the operand bank")


def test_the_payload_is_the_written_operand_on_its_grid_per_sample(program):
    mapping = program.identity_mapping()
    flow = program.identity()
    _out, state, _flat = flow.run_retaining_state(_inputs())
    _index, stage, bank = _neural_stage_with_bank(mapping)
    payloads = materialize(stage, state, mapping.dynamic_bank_bindings)
    binding = mapping.dynamic_bank_bindings[bank]
    write = state[binding.source_node_id].reshape(8, -1)
    codes = payloads[bank]
    assert codes.shape == (8, *binding.payload_shape)
    expected = write[:, binding.payload_offset: binding.payload_offset + K * T].reshape(8, K, T) / binding.payload_step
    assert torch.equal(codes, torch.round(expected)) and float((expected - codes).abs().max()) < 1e-9
    assert np.count_nonzero(codes.numpy())


def test_the_bound_core_output_is_the_law_of_the_exact_integer_product(program):
    """Each instance's codes (the golden's, which the executor equals at atol 0) are the stamped law of the
    int64 product of its streamed codes and the payload codes, recomputed here from the executor's state."""
    x = _inputs()
    mapping = program.identity_mapping()
    trace = GoldenTrace()
    run_golden(mapping, x.numpy(), trace=trace)
    _out, state, _flat = program.identity().run_retaining_state(x)
    bank = next(iter(mapping.dynamic_bank_bindings))
    hard = {int(placements[0]["ir_node_id"]): core for s in mapping.stages if s.kind == "neural"
            for core, placements in zip(s.hard_core_mapping.cores, s.hard_core_mapping.soft_core_placements_per_hard_core)}
    instances = [n for n in program.ir.get_neural_cores() if n.weight_bank_id == bank]
    assert len(instances) == T
    for core in instances:
        step = float(core.readout_law.input_step[0])  # every column of an instance reads one input grid
        sources = core.input_sources.flatten()
        streamed = np.stack([state[int(s.node_id)][:, int(s.index)].numpy() for s in sources], axis=1)
        q = np.round(streamed / step).astype(np.int64)
        assert float(np.abs(streamed / step - q).max()) < 1e-9
        acc = np.einsum("bk,bkn->bn", q, trace.payloads[bank])
        law = core.readout_law
        codes = apply_law_int64(acc, m0=law.m0, bias=law.bias, shift=law.shift, lo=law.lo, hi=law.hi,
                                adc_shift=law.adc_shift, adc_levels=law.adc_levels)
        assert np.array_equal(trace.core_codes[id(hard[int(core.id)])], codes) and np.count_nonzero(codes)


def test_two_samples_with_different_operands_program_different_payloads_and_outputs(program):
    x = _inputs(batch=2)
    mapping = program.identity_mapping()
    codes = payload_codes(mapping, x.numpy())
    bank = next(iter(mapping.dynamic_bank_bindings))
    assert not np.array_equal(codes[(0, bank)], codes[(1, bank)])
    out = program.identity()(x)
    assert not torch.equal(out[0], out[1])


def test_a_mixed_batch_materializes_per_sample(program):
    x = _inputs(batch=6, seed=9)
    flow = program.packed(axons=64, neurons=64)
    together = flow(x)
    alone = torch.cat([flow(x[i:i + 1]) for i in range(x.shape[0])])
    assert_exact(together, alone)


def test_the_payload_edge_passes_and_counts_every_written_cell(program):
    """Each streamed instance is placed as its own region holding the whole bank, so the cells written per
    sample are the bank's cells times its placed regions (T here), every one of them compared."""
    x = _inputs()
    mapping = program.packed_mapping(axons=64, neurons=64)
    regions = sum(1 for s in mapping.stages if s.kind == "neural"
                  for placements in s.hard_core_mapping.soft_core_placements_per_hard_core for p in placements
                  if p.get("weight_bank_id") in mapping.dynamic_bank_bindings)
    assert regions == T
    from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
    certificate = certify_payload_cells(ValueHybridCoreFlow(mapping, dtype=torch.float64, device="cpu"), program.identity_mapping(), x)
    assert certificate.passed and certificate.mismatched == 0
    assert certificate.cells_compared == x.shape[0] * regions * K * T and certificate.nonzero_payload_codes > 0


def test_a_payload_cached_across_samples_fails_the_payload_edge(program, monkeypatch):
    original = operand_payloads.materialize

    def cached(stage, state_buffer, bindings, sample_slice=slice(None)):
        fresh = original(stage, state_buffer, bindings, sample_slice)
        return {bank: codes[:1].expand_as(codes).clone() for bank, codes in fresh.items()}

    monkeypatch.setattr(operand_payloads, "materialize", cached)
    x = _inputs()
    certificate = certify_payload_cells(program.packed(axons=64, neurons=64), program.identity_mapping(), x)
    assert not certificate.passed and certificate.mismatched > 0


def test_the_golden_reads_the_payload_write_and_refuses_an_off_grid_payload(program, monkeypatch):
    x = _inputs()
    mapping = program.identity_mapping()
    codes, _steps = run_golden(mapping, x.numpy())
    assert np.count_nonzero(codes)
    write = next(s.compute_op for s in mapping.stages if s.kind == "compute"
                 and isinstance(s.compute_op, ComputeOp) and s.compute_op.op_type == "OperandWrite")
    module = write.params["module"]
    monkeypatch.setattr(type(module), "forward", lambda self, v: v + 1e-3)
    from mimarsinan.models.nn.activations.integer_law import OffGridInputError
    with pytest.raises(OffGridInputError):
        run_golden(mapping, x.numpy())
