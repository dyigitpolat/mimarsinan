"""A generic host op reading a non-floating activation (a boolean mask, an index) reads it in its traced dtype.

A value program carries every value in a floating buffer, so on the base ``torch.where(y > 0, ...)`` and
``torch.masked_fill(y, y > 0, 0)`` failed loudly in stage 2 (``RuntimeError``: a mask of dtype double), and their
method forms, refused by the analyzer on the base, would fail the same way through the method rule. The host op now
reads such an operand back in its traced dtype (``ComputeAdapter.operand_dtypes``; exact, the buffer holds its 0/1
or integer values; a no-op in the converted model), so stage 2 certifies them.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from operand_completeness.harness import (
    CONTRACTS,
    assert_converted_exact,
    assert_value_program_exact,
    native,
    quietly,
)
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model

FORMS = {
    "where_fn": lambda y: torch.where(y > 0, y, y * 0.5),
    "masked_fill_fn": lambda y: torch.masked_fill(y, y > 0, 0.0),
    "where_method": lambda y: y.where(y > 0, y * 0.5),
    "masked_fill_method": lambda y: y.masked_fill(y > 0, 0.0),
    "mask_times_value": lambda y: y * (y > 0),
}


class Masked(nn.Module):
    def __init__(self, form):
        super().__init__()
        self.a, self.head = nn.Linear(8, 8), nn.Linear(8, 10)
        self.form = form

    def forward(self, x):
        return self.head(self.form(self.a(x)))


@pytest.mark.parametrize("key", sorted(FORMS))
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_host_op_with_a_mask_operand_converts_exactly(key, contract):
    assert_converted_exact(native(lambda: Masked(FORMS[key])), (8,), contract)


@pytest.mark.parametrize("key", sorted(FORMS))
def test_a_host_op_with_a_mask_operand_passes_stage_2(key):
    model = native(lambda: Masked(FORMS[key]))
    assert_value_program_exact(model, (8,))
    flow = quietly(convert_torch_model, model, (8,), 10, packaging=MVM_PACKAGING, value_program_stage=True)
    assert flow.conversion_certificate.certified
