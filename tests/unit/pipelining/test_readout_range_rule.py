"""[U-BRANGE] F-4, one range rule for entry and readout ranges: Readout Law ranges every readout grid (per perceptron,
per consumer group, a bank's payload, a lane's end) by the boundary's rule, read from the same source: the least squared
error of the site's own grid over the calibration batches, searched from the 0.999 quantile to the calibration
maximum. Every error here is read directly from the grid (``range_rule_reference``)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.lanes.model_lanes import measure_lane_ranges
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.models.nn.activations.readout_quantizer import ReadoutQuantizer
from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import (
    calibrate_readout_quantizers,
    install_readout_quantizers,
)
from mimarsinan.torch_mapping.converter import convert_torch_model
from unit.pipelining.range_rule_reference import RELATIVE, grid_error, lane_grid, least_error, signed_grid

LAW = {"adc_bits": 40, "range": "calibrated", "mult_bits": 24}


def _flow():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
    return convert_torch_model(model, (16,), 8, device="cpu", packaging=MVM_PACKAGING).eval()


def _batches():
    generator = torch.Generator().manual_seed(4)
    return [torch.randn(512, 16, generator=generator) for _ in range(2)]


class TestPerPerceptronReadouts:
    def test_each_readout_range_is_the_least_error_of_its_own_grid(self):
        flow, batches = _flow(), _batches()
        perceptrons = list(flow.get_perceptrons())
        quantizers = install_readout_quantizers(perceptrons, 8)
        seen = {id(q): [] for q in quantizers}
        handles = [q.register_forward_pre_hook(lambda m, args, q=q: seen[id(q)].append(args[0].detach().clone()))
                   for q in quantizers]
        with torch.no_grad():
            for x in batches:
                flow(x)
        for handle in handles:
            handle.remove()
        calibrate_readout_quantizers(flow, perceptrons, quantizers, batches,
                                     platform={"activation_bits": 8, "readout_law": LAW})
        for quantizer in quantizers:
            magnitudes = [y.abs() for y in seen[id(quantizer)]]
            chosen = float(quantizer.range)
            assert grid_error(magnitudes, chosen, signed_grid(8)) <= least_error(magnitudes, signed_grid(8)) * (
                1 + RELATIVE)


class TestLaneReadouts:
    @pytest.mark.parametrize("unsigned", [True, False])
    def test_a_lane_range_is_the_least_error_of_its_lane_grid(self, unsigned):
        quantizer = ReadoutQuantizer(8)
        quantizer.set_lane(unsigned=unsigned)
        generator = torch.Generator().manual_seed(6)
        batches = [torch.randn(256, 64, generator=generator) for _ in range(2)]
        (chosen,) = measure_lane_ranges(quantizer, [quantizer], batches)
        magnitudes = [torch.relu(b) if unsigned else b.abs() for b in batches]
        snap = lane_grid(8) if unsigned else signed_grid(8)
        assert grid_error(magnitudes, chosen, snap) <= least_error(magnitudes, snap) * (1 + RELATIVE)


def test_entry_and_readout_ranges_read_one_rule_from_one_source():
    import mimarsinan.mapping.lanes.model_lanes as lanes
    import mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step as entry
    import mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step as readout
    from mimarsinan.mapping.support import range_rule

    assert entry.least_squares_ranges is readout.least_squares_ranges is range_rule.least_squares_ranges
    assert lanes.least_squares_ranges is range_rule.least_squares_ranges


# --- [U-BRANGE2 stage 1b] the sweep's survivors at the readout sites' widths and the fixed range ----------------------


def test_a_readout_s_range_is_the_least_error_of_the_grid_of_its_own_width():
    """Sweep SW04: each readout's site reads its quantizer's own width. At 4 bits every range is the least error of the
    signed 4-bit grid, and the fixture holds a readout whose 8-bit least lies elsewhere (so 8 bits read in its place
    fails here)."""
    flow, batches = _flow(), _batches()
    perceptrons = list(flow.get_perceptrons())
    quantizers = install_readout_quantizers(perceptrons, 4)
    seen = {id(q): [] for q in quantizers}
    handles = [q.register_forward_pre_hook(lambda m, args, q=q: seen[id(q)].append(args[0].detach().clone()))
               for q in quantizers]
    with torch.no_grad():
        for x in batches:
            flow(x)
    for handle in handles:
        handle.remove()
    calibrate_readout_quantizers(flow, perceptrons, quantizers, batches,
                                 platform={"activation_bits": 4, "readout_law": LAW})
    told = False
    for quantizer in quantizers:
        magnitudes = [y.abs() for y in seen[id(quantizer)]]
        chosen = float(quantizer.range)
        assert grid_error(magnitudes, chosen, signed_grid(4)) <= least_error(magnitudes, signed_grid(4)) * (1 + RELATIVE)
        told = told or grid_error(magnitudes, chosen, signed_grid(8)) > least_error(magnitudes, signed_grid(8)) * (
            1 + RELATIVE)
    assert told, "fixture: a readout whose 4-bit range is not its 8-bit least"


def test_a_declared_fixed_range_is_installed_and_no_readout_site_is_searched(monkeypatch):
    """Sweep SW05 and SW09: under a declared ``range_value`` every readout takes it and no readout is a range site; on
    an operand program the re-snapped banks' payload grids are still calibrated by the rule (their only sites), and the
    readers' unions, which name readouts, are read for the payload sites alone."""
    import contextlib
    import io

    import mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step as readout_step
    from mimarsinan.mapping.mappers.operand import operand_perceptrons
    from mimarsinan.mapping.support.operand_lowering import lower_contractions
    from unit.chip_simulation.value_run.operand_programs import attention_block, operand_platform

    platform = operand_platform(payload_bits=4, grouped=False)
    platform = {**platform, "readout_law": {**platform["readout_law"], "range_value": 0.75}}
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(attention_block(), (3, 8, 8), 5, device="cpu", packaging=MVM_PACKAGING).eval()
    lower_contractions(flow.get_mapper_repr(), platform)
    banks = [op for op in operand_perceptrons(flow.get_mapper_repr()) if op.payload_quantizer is not None]
    assert banks, "fixture: a re-snapped bank"
    captured = []
    rule = readout_step.least_squares_pooled

    def recording(model, sites, unions, batches, **kwargs):
        captured.append((list(sites), [list(u) for u in unions]))
        return rule(model, sites, unions, batches, **kwargs)

    monkeypatch.setattr(readout_step, "least_squares_pooled", recording)
    perceptrons = list(flow.get_perceptrons()) + banks
    quantizers = readout_step.install_readout_quantizers(perceptrons, 8)
    batches = [torch.randn(4, 3, 8, 8, generator=torch.Generator().manual_seed(s)) for s in (5, 6)]
    ranges = readout_step.calibrate_readout_quantizers(flow, perceptrons, quantizers, batches, platform=platform)
    assert ranges == [0.75] * len(quantizers) and all(float(q.range) == 0.75 for q in quantizers)
    (sites, _unions), = captured
    payload = {id(op.payload_quantizer) for op in banks}
    assert {id(s.module) for s in sites} == payload, "a readout was searched under a declared fixed range"
    assert all(float(op.payload_quantizer.grid.scale) > 0.0 for op in banks)
