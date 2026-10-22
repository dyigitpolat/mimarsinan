"""A convolution whose core lowering needs capability class G (``groups > 1``) is the exact host op of its module.

The owner's decision 4: grouped and depthwise convolutions run on the host as their module's own call, their
reason a stated boundary ("grouped convolution on the host until N5"), never a refusal. Nothing folds into them (a
following BatchNorm is its own exact host affine op, an activation its own host op), and the converted model and the
MVM identity value program equal the native model. On the base every such program was refused by
``RepresentabilityError``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.torch_mapping.converter import check_representability
from operand_completeness.harness import (
    CONTRACTS,
    PACKAGINGS,
    assert_converted_exact,
    assert_value_program_exact,
    native,
)


class GroupedBlock(nn.Module):
    """conv, grouped conv, BatchNorm, ReLU, depthwise conv, pointwise conv, head."""

    def __init__(self, groups=2):
        super().__init__()
        self.stem, self.act0 = nn.Conv2d(3, 8, 3, padding=1), nn.ReLU()
        self.grouped = nn.Conv2d(8, 8, 3, padding=1, groups=groups)
        self.bn, self.act1 = nn.BatchNorm2d(8), nn.ReLU()
        self.depthwise = nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False)
        self.pointwise, self.act2 = nn.Conv2d(8, 4, 1), nn.ReLU()
        self.head = nn.Linear(4 * 6 * 6, 10)

    def forward(self, x):
        y = self.act1(self.bn(self.grouped(self.act0(self.stem(x)))))
        y = self.act2(self.pointwise(self.depthwise(y)))
        return self.head(torch.flatten(y, 1))


class DepthwiseConv1d(nn.Module):
    """A Mamba-style causal depthwise Conv1d over time (groups = channels)."""

    def __init__(self):
        super().__init__()
        self.inp, self.act = nn.Linear(6, 8), nn.ReLU()
        self.dw = nn.Conv1d(8, 8, 3, padding=2, groups=8)
        self.head = nn.Linear(8 * 5, 10)

    def forward(self, x):
        y = self.act(self.inp(x)).transpose(1, 2)
        return self.head(torch.flatten(self.dw(y)[..., :5], 1))


class ReflectGrouped(nn.Module):
    """A grouped conv with reflect padding: as the host op of its module, its padding is the module's own."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1, groups=2, padding_mode="reflect")
        self.head = nn.Linear(4 * 6 * 6, 10)

    def forward(self, x):
        return self.head(torch.flatten(self.conv(x), 1))


CASES = {
    "grouped_and_depthwise": (GroupedBlock, (3, 6, 6)),
    "depthwise_conv1d": (DepthwiseConv1d, (5, 6)),
    "reflect_grouped": (ReflectGrouped, (4, 6, 6)),
}


def _host_convs(flow):
    return [m for m in flow.get_mapper_repr().execution_order()
            if isinstance(m, ComputeOpMapper) and isinstance(m.module, (nn.Conv1d, nn.Conv2d))]


@pytest.mark.parametrize("key", sorted(CASES))
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_grouped_convolution_is_the_exact_host_op_of_its_module(key, contract):
    factory, shape = CASES[key]
    flow = assert_converted_exact(native(factory), shape, contract)
    grouped = [m for m in _host_convs(flow) if m.module.groups > 1]
    assert grouped and all(type(m.module) in (nn.Conv1d, nn.Conv2d) for m in grouped)


@pytest.mark.parametrize("key", sorted(CASES))
def test_a_grouped_convolution_value_program_is_exact(key):
    factory, shape = CASES[key]
    assert_value_program_exact(native(factory), shape)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_the_report_states_the_host_placement_and_folds_nothing_into_it(contract):
    report = check_representability(native(GroupedBlock), (3, 6, 6), packaging=PACKAGINGS[contract])
    assert report.is_representable
    stated = {op.node_name: op.reason for op in report.supported_ops if op.reason}
    assert set(stated) == {"grouped", "depthwise"}
    assert all("capability class G" in r and "grouped convolution on the host until N5" in r for r in stated.values())
    assert not any(target.endswith((":grouped", ":depthwise")) for target in report.absorption_plan.values())


def test_the_capability_rule_names_class_g_for_groups_only():
    from mimarsinan.torch_mapping.representability_analyzer import core_lowering_requirement

    assert core_lowering_requirement(nn.Conv2d(4, 4, 3)) is None
    assert core_lowering_requirement(nn.Linear(4, 4)) is None
    assert core_lowering_requirement(nn.Conv2d(4, 4, 3, groups=2)).capability == "G"
    assert core_lowering_requirement(nn.Conv1d(4, 4, 3, groups=4)).capability == "G"
