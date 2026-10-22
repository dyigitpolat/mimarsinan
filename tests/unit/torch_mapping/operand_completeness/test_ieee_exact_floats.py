"""Float operations on the batch (review rounds 12 and 13; restated in review round 14).

The round-12 reviewer's views (``r12_float.py``) compute a per-sample width from a float of the batch:
``2 + math.floor(0.0 * B)`` (FloorZeroView), the length of ``torch.arange(B, B + 2, 0.5)`` (ArangeHalfView) and
``math.floor(3 * (B / B))`` (HalfBatchPerSampleView). Each was exact on w0-base with a static view; the round-12 tree
refused them by name (a float decision), and the round-13 tree kept their static views by IEEE-exactness proofs
(``symbolic/arithmetic/ieee_exactness.py``).

[restated in review round 14: the owner's shape-proof domain folds a batch-dependent scalar only through affine
integer forms ``a*B + c``; float arithmetic on a batch-dependent value is opaque, whatever IEEE double computes, and the
IEEE proofs are deleted with the rest of the special-case arithmetic. The value is computed on the host at run time,
and so is every node depending on it in a shape position: each view is the host op of its native method, exact at
batches 1 to 64 (no static per-sample layout of it), as are OverflowLateView (``B * 1e307``) and ArangeLateView
(``B * 2**50``); stage 2, whose value program fixes every per-sample shape at conversion, is refused by name, naming
the first opaque node.]
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.mapping.mapping_utils import ReshapeMapper

from .harness import CONTRACTS, PACKAGINGS, assert_converted_exact, native, quietly
from .test_scalar_operator_semantics import Floats, Kernels

ALL_BATCHES = tuple(range(1, 65))

WIDTHS = {
    "floor_zero": lambda b: 2 + math.floor(0.0 * b),
    "arange_half": lambda b: torch.arange(b, b + 2, 0.5).shape[0],
    "half_batch_per_sample": lambda b: math.floor(3 * (b / b)),
}
FIRST_OPAQUE = {"floor_zero": "mul", "arange_half": "arange", "half_batch_per_sample": "truediv"}


class ExactFloats(nn.Module):
    """The round-12 reviewer's views, and the same widths as a host op's operand (a tile count)."""

    def __init__(self, how: str, tile: bool = False):
        super().__init__()
        self.how, self.tile = how, tile
        self.fc = nn.Linear(8, 12)

    def forward(self, x):
        y, b = self.fc(x), x.size(0)
        width = WIDTHS[self.how](b)
        if self.tile:
            return y * torch.tile(y, (1, width)).shape[1]
        return F.softmax(y.view(b, width, -1), -1).flatten(1)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(WIDTHS))
def test_a_per_sample_view_from_a_float_of_the_batch_runs_on_the_host(how, contract):
    """Exact on w0-base and on the round-13 tree with static views; refused by name on the round-12 tree. The width is
    opaque now: the view is the host op of its native method, exact at batches 1 to 64."""
    flow = assert_converted_exact(native(lambda: ExactFloats(how)), (8,), contract, batches=ALL_BATCHES,
                                  kinds=("normal", "uniform"))
    assert not any(isinstance(m, ReshapeMapper) for m in flow.get_mapper_repr().execution_order())
    assert flow.shape_opaque[:2] == (FIRST_OPAQUE[how], FIRST_OPAQUE[how]) and flow.shape_impurity is None


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(WIDTHS))
def test_a_width_from_a_float_of_the_batch_is_a_run_time_host_value(how, contract):
    flow = assert_converted_exact(native(lambda: ExactFloats(how, tile=True)), (8,), contract, batches=ALL_BATCHES,
                                  kinds=("normal", "uniform"))
    assert flow.shape_literals == {} and flow.shape_opaque[:2] == (FIRST_OPAQUE[how], FIRST_OPAQUE[how])


@pytest.mark.parametrize("how", sorted(WIDTHS))
def test_stage_2_of_a_program_with_a_float_of_the_batch_is_refused_by_name(how):
    with pytest.raises(ToolchainRefusal, match="identity value program") as excinfo:
        quietly(convert_torch_model, native(lambda: ExactFloats(how, tile=True)), (8,), 10,
                packaging=PACKAGINGS["mvm"], value_program_stage=True)
    name = FIRST_OPAQUE[how]
    assert "OpaqueBatchValueError" in str(excinfo.value) and f"{name} (FX node {name!r})" in str(excinfo.value)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_an_overflowing_float_view_runs_on_the_host(contract):
    """OverflowLateView (``B * 1e307``) and ArangeLateView (``B * 2**50``): certified and wrong on the round-11 tree,
    refused by name on the round-12 and round-13 trees; exact at batches 1 to 64 now, 17, 18 and 20 included."""
    for factory, name in ((lambda: Floats("view", "overflow"), "mul"), (lambda: Kernels("arange_step_view"), "arange")):
        flow = assert_converted_exact(native(factory), (8,), contract, batches=ALL_BATCHES, kinds=("normal", "uniform"))
        assert flow.shape_opaque[:2] == (name, name)
