"""[U-SC S-2] The count-grid rule itself: which host ops it governs, the exact decode, the comb re-encode, the deployed
seam that applies it to a plain op reading a raw-gauge buffer, and its refusals by name.

A plain host op that reads a raw-gauge buffer (a host-run spiking perceptron's raw value, or a plain op inheriting it)
is evaluated by ``CountGridModule``: each input decodes once to its exact window counts (``value / divisor * T``, on the
integer grid by construction), the op runs in fp64 on those counts, and the result re-encodes through the one comb rule
(``comb_round``, the chip's tie rule ``comb_spike_count`` uses) before it returns at the op's own gauge. Both twins use
the one module: the NF with unit divisors (it holds the wire rate), the deployed executors with the divisor table.
"""

from __future__ import annotations

import operator

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.recording.spike_modes import comb_spike_count
from mimarsinan.common.refusal.toolchain_refusal import ToolchainRefusal
from mimarsinan.mapping.ir import ComputeOp, IRSource
from mimarsinan.mapping.support.compute_modules import ComputeAdapter, ScaleNormalizingWrapper
from mimarsinan.spiking.count_grid import (
    CountGridModule,
    CountGridOperandError,
    comb_round,
    count_grid_applies,
    decode_window_counts,
    host_op_for_evaluation,
    op_commutes_exactly,
)

T = 15

NON_COMMUTING = [
    nn.AvgPool1d(2), nn.AvgPool2d(2), nn.AvgPool3d(2),
    nn.AdaptiveAvgPool1d(1), nn.AdaptiveAvgPool2d(1), nn.AdaptiveAvgPool3d(1),
    nn.Linear(4, 3, bias=False), nn.Conv1d(2, 2, 3, bias=False), nn.Conv2d(2, 2, 3, bias=False),
    nn.Conv3d(2, 2, 1, bias=False),
    ComputeAdapter(torch.mean), ComputeAdapter(torch.sum),
]
EXACTLY_COMMUTING = [
    nn.MaxPool1d(2), nn.MaxPool2d(2), nn.MaxPool3d(2),
    nn.AdaptiveMaxPool1d(1), nn.AdaptiveMaxPool2d(1), nn.AdaptiveMaxPool3d(1),
    nn.Identity(), nn.Flatten(),
    ComputeAdapter(torch.amax), ComputeAdapter(torch.amin), ComputeAdapter(torch.flatten),
    ComputeAdapter(torch.reshape), ComputeAdapter(torch.permute), ComputeAdapter(torch.transpose),
    ComputeAdapter(torch.cat),
]
#: Ops a raw-gauge read never reaches as a plain op: the scale policy arms them (they own their domain).
VALUE_OPS = [nn.Linear(4, 3), nn.Conv2d(2, 2, 3), nn.LayerNorm(4), ComputeAdapter(operator.add), nn.Upsample(scale_factor=2)]


@pytest.mark.parametrize("module", NON_COMMUTING, ids=lambda m: type(m).__name__ + getattr(getattr(m, "fn", None), "__name__", ""))
def test_every_non_commuting_wire_transparent_op_takes_the_rule(module):
    assert not op_commutes_exactly(module)
    assert count_grid_applies(module)


@pytest.mark.parametrize("module", EXACTLY_COMMUTING, ids=lambda m: type(m).__name__ + getattr(getattr(m, "fn", None), "__name__", ""))
def test_exactly_commuting_ops_stay_as_they_are(module):
    assert op_commutes_exactly(module)
    assert not count_grid_applies(module)


@pytest.mark.parametrize("module", VALUE_OPS, ids=lambda m: type(m).__name__)
def test_value_ops_are_left_to_their_domain_owning_wrapper(module):
    assert not count_grid_applies(module)


def test_the_comb_rule_is_shared_with_the_chip_comb():
    x = torch.tensor([0.0, 0.49999, 0.5, 1.5, 2.5, 5.5, 6.0, 14.5, -0.5, -1.5], dtype=torch.float64)
    assert torch.equal(comb_round(x), torch.tensor([0, 0, 1, 2, 3, 6, 6, 15, 0, -1], dtype=torch.float64))
    rates = torch.tensor([0.0, 1 / 30, 3 / 30, 11 / 30, 1.0], dtype=torch.float64)
    assert torch.equal(comb_spike_count(rates, T).to(torch.float64), comb_round(rates * T))


@pytest.mark.parametrize("divisor", (1.0, 2.185, 0.731))
def test_the_decode_recovers_exact_window_counts_from_either_gauge(divisor):
    counts = torch.arange(T + 1, dtype=torch.float32)
    buffer = (counts / T * divisor).to(torch.float32)
    decoded = decode_window_counts(buffer, divisor, T)
    assert decoded.dtype == torch.float64
    assert torch.equal(decoded, counts.to(torch.float64))


def test_a_buffer_off_the_count_grid_is_refused_by_name():
    with pytest.raises(CountGridOperandError, match="window-count grid"):
        decode_window_counts(torch.tensor([0.123]), 1.0, T)
    assert issubclass(CountGridOperandError, ToolchainRefusal)


@pytest.mark.parametrize("theta", (2.185, 0.731))
def test_a_half_count_average_decides_alike_in_both_gauges(theta):
    """The SD-10c tie: the average of the counts 5 and 6 is exactly 5.5, and the comb takes it up to 6 in both twins."""
    counts = torch.tensor([[[5.0, 6.0, 4.0, 5.0, 8.0, 9.0]]])
    wire = CountGridModule(nn.AvgPool1d(2), in_divisor=1.0, out_divisor=1.0, simulation_length=T)
    raw = CountGridModule(nn.AvgPool1d(2), in_divisor=theta, out_divisor=theta, simulation_length=T)
    out_wire = wire((counts / T).to(torch.float32))
    out_raw = raw((counts / T * theta).to(torch.float32))
    assert out_wire.dtype == torch.float32 and out_raw.dtype == torch.float32
    expected = torch.tensor([[[6.0, 5.0, 9.0]]], dtype=torch.float64)
    assert torch.equal(torch.round(out_wire.double() * T), expected)
    assert torch.equal(torch.round(out_raw.double() / theta * T), expected)


def test_a_linear_op_runs_in_fp64_on_the_counts():
    torch.manual_seed(0)
    linear = nn.Linear(6, 4, bias=False)
    x = torch.randint(0, T + 1, (3, 6)).to(torch.float32)
    module = CountGridModule(linear, in_divisor=1.0, out_divisor=1.0, simulation_length=T)
    out = module(x / T)
    ref = comb_round(x.double() @ linear.weight.detach().double().t())
    assert torch.equal(torch.round(out.double() * T), ref)
    assert linear.weight.dtype == torch.float32, "the wrapped module keeps its own dtype"


def test_gradients_pass_straight_through_the_decode_and_the_re_encode():
    linear = nn.Linear(6, 4, bias=False)
    module = CountGridModule(linear, in_divisor=1.0, out_divisor=1.0, simulation_length=T)
    x = (torch.randint(0, T + 1, (3, 6)).to(torch.float32) / T).requires_grad_(True)
    module(x).sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0
    assert linear.weight.grad is not None and float(linear.weight.grad.abs().sum()) > 0


# ---------------------------------------------------------------------------
# The deployed seam: the op the executors evaluate
# ---------------------------------------------------------------------------


def _op(module, sources, op_id=9):
    return ComputeOp(id=op_id, name="op", input_sources=np.array([IRSource(node_id=s, index=i) for i, s in enumerate(sources)]),
                     op_type=type(module).__name__, params={"module": module, "input_shape": (1, len(sources))})


def test_a_plain_average_pool_on_a_raw_buffer_is_evaluated_on_the_count_grid():
    op = _op(nn.AvgPool1d(2), [3, 3])
    seen = host_op_for_evaluation(op, {3: 2.185, 9: 2.185}, simulation_length=T, spiking_mode="lif")
    assert seen is not op and seen.id == op.id
    module = seen.params["module"]
    assert isinstance(module, CountGridModule)
    assert module.in_divisor == pytest.approx(2.185) and module.out_divisor == pytest.approx(2.185)
    assert op.params["module"].__class__ is nn.AvgPool1d, "the program's own op is untouched"


@pytest.mark.parametrize("module", [nn.MaxPool1d(2), nn.Identity()], ids=lambda m: type(m).__name__)
def test_an_exactly_commuting_op_is_evaluated_as_it_is(module):
    op = _op(module, [3, 3])
    assert host_op_for_evaluation(op, {3: 2.185, 9: 2.185}, simulation_length=T, spiking_mode="lif") is op


def test_an_op_on_wire_buffers_is_evaluated_as_it_is():
    op = _op(nn.AvgPool1d(2), [3, 3])
    assert host_op_for_evaluation(op, {}, simulation_length=T, spiking_mode="lif") is op


def test_a_domain_owning_op_keeps_its_wire_read():
    owner = _op(ScaleNormalizingWrapper(nn.AvgPool1d(2), [torch.ones(1)], torch.ones(1)), [3, 3])
    assert host_op_for_evaluation(owner, {3: 2.185, 9: 2.185}, simulation_length=T, spiking_mode="lif") is owner


def test_the_rule_is_scoped_to_the_lif_executors():
    op = _op(nn.AvgPool1d(2), [3, 3])
    assert host_op_for_evaluation(op, {3: 2.185, 9: 2.185}, simulation_length=T, spiking_mode="rate") is op


def test_a_raw_buffer_beside_a_wire_buffer_is_refused_by_name():
    """No one count grid holds both gauges: a non-commuting plain op over them has no exact operand domain."""
    op = _op(nn.AvgPool1d(2), [3, 4])
    with pytest.raises(CountGridOperandError, match="raw-gauge"):
        host_op_for_evaluation(op, {3: 2.185, 9: 1.59}, simulation_length=T, spiking_mode="lif")
