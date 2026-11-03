"""[U-LMP, U-BRG review m-3] A reader whose producers' grids the drive twin could not determine is determined through the
solvers' union, or refused by name; it never keeps the raw 1.0 gauge silently.

On the drive route (a declared readout law on value-domain cores, the bias on an always-on row) every core-mapped
perceptron's row carries the top code value of the grid its input lies on (``mapping/support/bias_drive.py``). Three
reader forms fell back to 1.0 with a log line (the defect the gauge fixed): a reader of a consumer-grouped producer
(N2's ``readout_ranges: per_consumer_group``), a reader mixing host and core inputs, and a host-fed reader without an
armed grid. The first is determined here as the law solver determines it: a static reader reads every group of its
producer (``consumer_group_partition``), the solver joins those groups' keys in its union and the reader's grid spans
the widest; groups of different levels joined by one reader are refused by name (the solver refuses them too). The
others are refused by name at Weight Quantization entry (the solver refuses them at Soft Core Mapping), for every
perceptron whose bias rides a row.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.export.law_grids import assign_core_grids
from mimarsinan.mapping.export.operand_laws import output_groups_of
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    calibrate_boundary_scales,
    install_boundary_quantizers,
)
from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import install_readout_quantizers
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons
from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron
from unit.mapping.test_bias_row_drive_program import _exact, _inputs, _perceptrons_by_index, _program
from unit.transformations.test_bias_row_drive import BITS, _value_config, _wq

GROUP_RANGES = (2.0, 3.5)


class _Chain(nn.Module):
    """``r`` and ``s`` read core ``p`` directly (two consumers keep ``p`` its own perceptron)."""

    def __init__(self):
        super().__init__()
        self.p, self.r, self.s = nn.Linear(16, 6), nn.Linear(6, 2), nn.Linear(6, 2)

    def forward(self, x):
        y = self.p(x)
        return torch.cat([self.r(y), self.s(y)], dim=1)


class _Mixed(nn.Module):
    """``m`` reads ``cat(a(x), sigmoid(x))``: one core output and one host value."""

    def __init__(self):
        super().__init__()
        self.a, self.m = nn.Linear(16, 6), nn.Linear(22, 4)

    def forward(self, x):
        return self.m(torch.cat([self.a(x), torch.sigmoid(x)], dim=1))


def _flow(module: nn.Module, *, levels=(127, 127)):
    """Converted, fused, entry grids and readout quantizers installed and calibrated; the first perceptron's readout
    split into two consumer groups (channels 0-2 and 3-5) when ``levels`` is given."""
    torch.manual_seed(0)
    flow = convert_torch_model(module.eval(), (16,), 4, device="cpu", packaging=MVM_PACKAGING).eval()
    perceptrons = list(flow.get_perceptrons())
    for perceptron in perceptrons:
        fuse_into_perceptron(perceptron, device="cpu")
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for perceptron in perceptrons:
            perceptron.layer.bias.copy_((torch.rand(perceptron.layer.bias.shape, generator=gen) * 2 - 1) * 0.6)
    entries = list(segment_entry_perceptrons(flow.get_mapper_repr()))
    grids = install_boundary_quantizers(entries, BITS)
    calibrate_boundary_scales(flow, entries, grids, [torch.randn(32, 16, generator=gen) * 3.0])
    readouts = install_readout_quantizers(perceptrons, BITS)
    for quantizer in readouts:
        quantizer.calibrate(4.0)
    if levels is not None:
        readouts[0].set_groups(np.array([0, 0, 0, 1, 1, 1]), tuple(levels), channel_axis=-1)
        readouts[0].calibrate_groups(GROUP_RANGES)
    flow.get_mapper_repr().assign_perceptron_indices()
    return flow


def _stamp(holder) -> float:
    return float(holder.__dict__.get("bias_row_drive", 1.0))


def test_a_reader_of_a_consumer_grouped_producer_is_driven_at_the_union_of_its_groups(monkeypatch):
    """The reader ``r`` reads both groups of ``p``: the solver joins them and assigns ``r`` the wider group's grid
    (range 3.5 on 127 levels), so ``r``'s row is driven at that grid's top code value, equal to the solver's
    ``input_step x input_levels``; the program is exact. Red before: ``r`` kept 1.0 and was printed as undetermined."""
    flow = _flow(_Chain())
    out = io.StringIO()
    with redirect_stdout(out):
        flow, ir = _program(monkeypatch, flow=flow)
    by_index = _perceptrons_by_index(flow)
    index = {str(p.name): i for i, p in by_index.items()}
    reader = by_index[index["r"]]
    assert _stamp(reader) == 127 * (GROUP_RANGES[1] / 127)
    assert "not determinable" not in out.getvalue()
    ranges = {int(i): float(p.scaler.range) for i, p in by_index.items()}
    grids, _ = assign_core_grids(ir, ranges, BITS, groups=output_groups_of(flow.get_mapper_repr()))
    cores = [n for n in ir.nodes if isinstance(n, NeuralCore) and n.perceptron_index == index["r"]]
    assert cores and all(_stamp(c) == grids[int(c.id)].input_step * grids[int(c.id)].input_levels for c in cores)
    _exact(flow, ir, _inputs())


def test_a_reader_joining_groups_of_different_levels_is_refused_by_name(monkeypatch):
    """Groups of 127 and 63 levels read by one reader have no one DAC grid (the solver refuses the union): the drive is
    refused by name at Weight Quantization entry, naming the reader. Red before: ``r`` kept 1.0 silently."""
    flow = _flow(_Chain(), levels=(127, 63))
    with pytest.raises(ToolchainRefusal) as caught:
        _wq(monkeypatch, flow, _value_config())
    assert type(caught.value).__name__ == "BiasRowDriveRefusal"
    assert "perceptron 'r'" in str(caught.value) and "levels" in str(caught.value), str(caught.value)


def test_a_reader_mixing_host_and_core_inputs_is_refused_by_name(monkeypatch):
    """``m`` reads a core output and a host value: no one grid (the solver refuses it at Soft Core Mapping); its drive
    is refused by name at Weight Quantization entry. Red before: ``m`` kept 1.0, printed as undetermined."""
    flow = _flow(_Mixed(), levels=None)
    with pytest.raises(ToolchainRefusal) as caught:
        _wq(monkeypatch, flow, _value_config())
    assert type(caught.value).__name__ == "BiasRowDriveRefusal"
    assert "perceptron 'm'" in str(caught.value) and "host" in str(caught.value), str(caught.value)


def test_a_reader_without_a_row_bias_keeps_no_drive_and_no_refusal(monkeypatch):
    """Guard: a reader whose bias is zero carries no row value, so nothing rides its row and no drive is owed."""
    flow = _flow(_Mixed(), levels=None)
    with torch.no_grad():
        flow.get_perceptrons()[1].layer.bias.zero_()
    _wq(monkeypatch, flow, _value_config())
    assert _stamp(flow.get_perceptrons()[1]) == 1.0
