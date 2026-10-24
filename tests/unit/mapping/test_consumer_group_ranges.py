"""N2 design §2.3: ``readout_ranges: per_consumer_group`` gives a perceptron one output range per group of the
columns its consumers read whole; operand-payload groups take the operand core type's ``payload_bits``; the
streamed Q keeps its group's grid across the payload write's host segment (no boundary quantizer on it)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.mappers.operand import OperandContractionMapper, operand_perceptrons
from mimarsinan.mapping.support.tensor_stats import safe_quantile
from mimarsinan.models.nn.activations.integer_law import grid_levels
from mimarsinan.models.nn.activations.readout_quantizer import ReadoutQuantizer
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    CALIBRATION_QUANTILE,
    MIN_BOUNDARY_SCALE,
)
from unit.chip_simulation.value_run.integer_programs import law_digests_by_core
from unit.chip_simulation.value_run.operand_programs import attention_program, operand_platform

DIM, HEADS = 8, 2
HEAD_DIM = DIM // HEADS


@pytest.fixture(scope="module")
def grouped():
    return attention_program(platform=operand_platform(payload_bits=4))


@pytest.fixture(scope="module")
def whole():
    return attention_program(platform=operand_platform(payload_bits=4, grouped=False))


def _perceptron(program, name):
    return next(p for p in program.model.get_perceptrons() if p.name == name)


def _cores(program, perceptron_name):
    index = next(getattr(m, "perceptron_index") for m in program.model.get_mapper_repr().execution_order()
                 if getattr(getattr(m, "perceptron", None), "name", None) == perceptron_name)
    return [n for n in program.ir.nodes if isinstance(n, NeuralCore) and n.perceptron_index == index]


def test_the_qkv_perceptron_of_a_two_head_block_gets_six_groups(grouped):
    quantizer = _perceptron(grouped, "blocks_0_self_attn_qkv").scaler
    assert isinstance(quantizer, ReadoutQuantizer)
    groups = np.asarray(quantizer.channel_groups)
    assert groups.shape == (3 * DIM,) and len(set(groups.tolist())) == 3 * HEADS
    for role in range(3):
        for head in range(HEADS):
            start = role * DIM + head * HEAD_DIM
            assert len(set(groups[start:start + HEAD_DIM].tolist())) == 1


def test_payload_groups_take_payload_bits_and_the_streamed_group_activation_bits(grouped):
    law = _cores(grouped, "blocks_0_self_attn_qkv")[0].readout_law
    hi = np.asarray(law.hi)
    assert set(hi[:DIM].tolist()) == {grid_levels(8)}          # Q, streamed
    assert set(hi[DIM:].tolist()) == {grid_levels(4)}          # K and V, written as payloads


def test_a_perceptron_read_whole_gets_one_group_and_the_per_perceptron_law(grouped, whole):
    quantizer = _perceptron(grouped, "blocks_0_linear1").scaler
    assert quantizer.channel_groups is None  # one group: G2's per-perceptron quantizer, its state untouched
    digest = [c.readout_law.digest() for c in _cores(grouped, "blocks_0_linear1")]
    assert digest == [c.readout_law.digest() for c in _cores(whole, "blocks_0_linear1")]


def test_group_ranges_are_the_boundary_rule_on_the_group_columns():
    """The per-group range is the 0.999 quantile of the group's outputs, the largest over the calibration
    batches, floored at MIN_BOUNDARY_SCALE: measured beside the calibration on the very tensors it reads."""
    import contextlib
    import io

    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.mapping.support.operand_lowering import lower_contractions
    from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import (
        calibrate_readout_quantizers,
        install_readout_quantizers,
    )
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from unit.chip_simulation.value_run.operand_programs import attention_block

    platform = operand_platform(payload_bits=4)
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(attention_block(), (3, 8, 8), 5, device="cpu", packaging=MVM_PACKAGING).eval()
    lower_contractions(flow.get_mapper_repr(), platform)
    perceptrons = list(flow.get_perceptrons()) + list(operand_perceptrons(flow.get_mapper_repr()))
    quantizers = install_readout_quantizers(perceptrons, 8)
    qkv = next(q for p, q in zip(perceptrons, quantizers) if p.name == "blocks_0_self_attn_qkv")
    seen = []
    handle = qkv.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    batches = [torch.randn(16, 3, 8, 8, generator=torch.Generator().manual_seed(s)) for s in (5, 6)]
    try:
        calibrate_readout_quantizers(flow, perceptrons, quantizers, batches, platform=platform)
    finally:
        handle.remove()
    groups = np.asarray(qkv.channel_groups)
    assert len(seen) >= 2 and len(set(groups.tolist())) == 3 * HEADS
    for group in sorted(set(groups.tolist())):
        columns = torch.as_tensor(np.flatnonzero(groups == group))
        measured = max(float(safe_quantile(y[..., columns].abs(), CALIBRATION_QUANTILE).item()) for y in seen[:2])
        assert float(qkv.group_ranges[group]) == max(measured, MIN_BOUNDARY_SCALE)


def test_the_s_core_reads_its_q_group_grid_with_no_boundary_quantizer(grouped):
    qkv_law = _cores(grouped, "blocks_0_self_attn_qkv")[0].readout_law
    mappers = [m for m in grouped.model.get_mapper_repr().execution_order() if isinstance(m, OperandContractionMapper)]
    scores = mappers[0]
    for head, operand in enumerate(scores.operand_perceptrons):
        assert isinstance(operand.input_activation, torch.nn.Identity)
        cores = [n for n in grouped.ir.nodes if isinstance(n, NeuralCore) and n.perceptron_index == operand.perceptron_index]
        assert cores and all(getattr(c, "boundary_grid", None) is None for c in cores)
        q_step = float(qkv_law.output_step[head * HEAD_DIM])
        assert {float(s) for c in cores for s in c.readout_law.input_step} == {q_step}


def test_every_operand_perceptron_carries_a_readout_quantizer(grouped):
    operands = list(operand_perceptrons(grouped.model.get_mapper_repr()))
    assert len(operands) == 2 * 2 * HEADS  # two products per block, two blocks, one perceptron per head
    assert all(isinstance(p.scaler, ReadoutQuantizer) for p in operands)


def test_consumer_groups_without_an_operand_core_type_group_nothing():
    """Groups are made only for the perceptrons an operand product reads (registry doc, integration review): with no
    operand core type declared (a projections-only twin) the declaration groups nothing, and every readout law
    equals per_perceptron's."""
    static_only = [{"max_axons": 256, "max_neurons": 256, "count": 64}]
    grouped = attention_program(platform=operand_platform(cores=static_only))
    plain = attention_program(platform=operand_platform(cores=static_only, grouped=False))
    assert not grouped.ir.dynamic_bank_bindings
    quantizers = [m for m in grouped.model.modules() if isinstance(m, ReadoutQuantizer)]
    assert quantizers and all(q.channel_groups is None for q in quantizers)
    assert law_digests_by_core(grouped.ir) == law_digests_by_core(plain.ir)
