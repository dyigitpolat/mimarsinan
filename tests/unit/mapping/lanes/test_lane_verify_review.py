"""The lane fix round verification's majors as certificates (11_wave2/lanes/verify_report.md): NF = SCM = HCM = int64
golden at atol 0, or a refusal by name.

N1 the offset fold's refit runs per channel, so the position cores of one padded convolution keep one law per channel
(a narrow ADC on a symmetric law, the identity output law), lanes on and off; N2 and N3 a pruned lane program beside
an offset lane grid is refused exactly where the grid's own rounding and clamp leave value 0 off its codes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.mapping.export.law_rules import core_bias_and_data_weights
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
from mimarsinan.models.nn.activations.integer_law import (
    accumulator_bounds,
    law_magnitude_bound,
    saturation_reach,
)
from unit.chip_simulation.value_run.integer_programs import assert_exact
from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform, real_inputs, write_state

#: V1 / V2: a declared offset entry grid on the network input (values k 0.2 - 1.4, codes 0..15) and R2's signed one.
NARROW_ENTRY = {"scale": 3.0, "offset": 1.4, "rounding": "floor", "codes": [0, 15]}
IDENTITY_ENTRY = {"scale": 6.0, "offset": 3.0, "rounding": "floor", "codes": [-15, 15]}
LANES = {"lanes": ("L", "E", "P"), "no_lanes": None}
#: The e2e grid declarations (WI-5b fractional weight scale, WI-5c bias at the law's resolution) beside the default.
GRIDS = {"default": {}, "fractional_law_bias": {"weight_scale_rule": "fractional", "bias_resolution": "law"}}


def _certify(program, x, **geometry):
    with torch.no_grad():
        scm = program.identity()(x)
        assert_exact(program.nf()(x), scm)
        mapping = program.packed_mapping(**geometry)
        hcm = ValueHybridCoreFlow(mapping, dtype=torch.float64)(x)
    assert_exact(hcm, scm)
    codes, steps = run_golden(mapping, x.numpy())
    assert np.array_equal(codes * steps, hcm.reshape(codes.shape).numpy())
    return codes


def _build_with_entry(model, shape, *, num_classes, classes, law, entry, tmp_path, seed=0, **extra):
    platform = lane_platform(classes, activation_bits=5, readout_law=law, **extra)

    def declare(flow, entries, grids, perceptrons, quantizers, lanes):
        fx = {id(p): n for n, p in perceptron_fx_names(flow.get_mapper_repr()).items()}
        platform["quantizer_state"] = write_state(Path(tmp_path) / "state.json", {fx[id(entries[0])]: {"entry": entry}})

    return build_lane_program(model, shape, num_classes=num_classes, platform=platform, hardware_bias=True, seed=seed,
                              state_hook=declare)


def _first_cores(program):
    return [n for n in program.ir.nodes if isinstance(n, NeuralCore) and n.perceptron_index == 0]


def _bounds(core, program):
    _bias, weights = core_bias_and_data_weights(core, program.ir, real_bias=True)
    low, high = core.boundary_grid.code_range
    return accumulator_bounds(weights, max(abs(int(low)), abs(int(high))))


def _channel_values(cores, field):
    from mimarsinan.mapping.export.law_stamp import core_channels

    values = {}
    for core in cores:
        for channel, value in zip(core_channels(core), getattr(core.readout_law, field)):
            values.setdefault(int(channel), set()).add(int(value))
    return values


def _assert_positions_differ(cores):
    """The program exercises N1: some channel's position cores carry different folded biases."""
    assert len(cores) > 1 and any(len(v) > 1 for v in _channel_values(cores, "bias").values())


class TestN1RefitPerChannel:
    @pytest.mark.parametrize("grids", list(GRIDS), ids=list(GRIDS))
    @pytest.mark.parametrize("lanes", list(LANES), ids=list(LANES))
    @pytest.mark.parametrize("adc_bits", [6, 8])
    def test_v2_a_narrow_adc_beside_an_offset_entry_keeps_one_law_per_channel(self, tmp_path, adc_bits, lanes, grids):
        """A padded conv reads the network input through an offset entry grid; a host MaxPool reads its lane (no
        clamp, a symmetric law) and the ADC is narrower than the accumulator. Every position column's ADC span covers
        its folded reach, every channel carries one ADC step, and the program certifies at atol 0."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.MaxPool2d(2), nn.ReLU(), nn.Flatten(),
                              nn.Linear(8 * 16, 10))
        program = _build_with_entry(model, (3, 8, 8), num_classes=10, classes=LANES[lanes], tmp_path=tmp_path,
                                    law={"adc_bits": adc_bits, "range": "calibrated", "mult_bits": 16},
                                    entry=NARROW_ENTRY, **GRIDS[grids])
        cores = _first_cores(program)
        _assert_positions_differ(cores)
        assert all(len(v) == 1 for v in _channel_values(cores, "adc_shift").values()), "one ADC step per channel"
        narrow = 0
        for core in cores:
            law = core.readout_law
            assert int(law.lo[0]) == -int(law.hi[0]), "the conv's lane law is symmetric"
            for j, bound in enumerate(_bounds(core, program)):
                reach = min(int(bound), saturation_reach(int(law.m0[j]), int(law.bias[j]), int(law.shift[j]),
                                                         int(law.hi[j])))
                assert int(law.adc_levels[j]) << int(law.adc_shift[j]) >= reach, (core.name, j)
                narrow += int(law.adc_levels[j]) < int(bound)
        assert narrow > 0, "the ADC is narrower than the accumulator"
        _certify(program, real_inputs(program, 32), axons=1024, neurons=256)

    @pytest.mark.parametrize("lanes", list(LANES), ids=list(LANES))
    @pytest.mark.parametrize("adc_bits", [40, 8])
    def test_v1_v2_the_identity_output_of_a_padded_conv_beside_an_offset_entry_certifies(self, tmp_path, adc_bits,
                                                                                        lanes):
        """R2's offset grid read by a padded conv under the identity output law: one no-clamp bound per channel at
        or above every position column's folded |z| bound, and no output reaches it."""
        torch.manual_seed(1)
        conv = nn.Conv2d(3, 4, 3, padding=1)
        with torch.no_grad():
            conv.weight.abs_()
        law = {"adc_bits": adc_bits, "range": "calibrated", "mult_bits": 16, "output_law": "identity"}
        program = _build_with_entry(nn.Sequential(conv, nn.Flatten()), (3, 8, 8), num_classes=4 * 64,
                                    classes=LANES[lanes], law=law, entry=IDENTITY_ENTRY, tmp_path=tmp_path)
        cores = _first_cores(program)
        _assert_positions_differ(cores)
        assert all(len(v) == 1 for v in _channel_values(cores, "hi").values()), "one identity bound per channel"
        for core in cores:
            law = core.readout_law
            z = law_magnitude_bound(_bounds(core, program), law.m0, law.bias, law.adc_shift)
            assert int(law.hi.min()) >= z and np.array_equal(law.lo, -law.hi), core.name
        codes = _certify(program, real_inputs(program, 32), axons=1024, neurons=256)
        assert int(np.abs(codes).max()) < int(min(int(c.readout_law.hi.min()) for c in cores))

    def test_the_channel_refit_gives_every_carrier_the_largest_step_and_bound(self):
        """Two position cores of one channel whose folded biases differ (-20 and -4): a 7-level ADC spans their
        unsaturated reaches (27 and 11) at exponents 2 and 1, so both carry 2; identity cores take the larger |z|
        bound."""
        from mimarsinan.mapping.export.entry_offsets import FoldRefit, refit_folded_laws
        from mimarsinan.mapping.ir.core_law import CoreLaw

        def law(bias, hi=7):
            one = np.ones(1)
            return CoreLaw(input_step=one, m0=np.array([1]), shift=np.array([0]), bias=np.array([bias]),
                           lo=np.array([-hi]), hi=np.array([hi]), output_step=one, adc_shift=np.array([0]),
                           adc_levels=np.array([7]))

        border, inner = law(-4), law(-20)
        out = refit_folded_laws({
            1: FoldRefit("border", border, (100,), identity=False, reach_hi=7, channels=(("p", 0),)),
            2: FoldRefit("inner", inner, (100,), identity=False, reach_hi=7, channels=(("p", 0),)),
            3: FoldRefit("other", law(-4), (100,), identity=False, reach_hi=7, channels=(("p", 1),)),
        })
        assert out[1].adc_shift.tolist() == out[2].adc_shift.tolist() == [2]
        assert out[3].adc_shift.tolist() == [1], "another channel keeps its own finest step"
        identity = refit_folded_laws({
            1: FoldRefit("border", law(-4, hi=1), (100,), identity=True, reach_hi=None, channels=(("q", 0),)),
            2: FoldRefit("inner", law(-20, hi=1), (100,), identity=True, reach_hi=None, channels=(("q", 0),)),
        })
        assert identity[1].hi.tolist() == identity[2].hi.tolist() == [1 * (100 + 1) + 20]
        assert identity[1].lo.tolist() == [-121]


class TestN1NamedStampRefusals:
    def test_the_plain_stamp_refuses_a_split_channel_law_by_name_and_field(self, tmp_path):
        import dataclasses

        from mimarsinan.common.refusal import ToolchainRefusal
        from mimarsinan.mapping.export.law_stamp import ReadoutLawChannelError, _channel_law

        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.MaxPool2d(2), nn.ReLU(), nn.Flatten(),
                              nn.Linear(8 * 16, 10))
        program = _build_with_entry(model, (3, 8, 8), num_classes=10, classes=None, tmp_path=tmp_path,
                                    law={"adc_bits": 6, "range": "calibrated", "mult_bits": 16}, entry=NARROW_ENTRY)
        cores = _first_cores(program)
        laws = {int(c.id): c.readout_law for c in cores}
        last = cores[-1]
        laws[int(last.id)] = dataclasses.replace(last.readout_law, adc_shift=last.readout_law.adc_shift + 1)
        with pytest.raises(ReadoutLawChannelError, match="adc_shift") as raised:
            _channel_law(cores, laws, 8, program.ir)
        assert isinstance(raised.value, ToolchainRefusal)

    def test_the_lane_stamp_names_the_field_that_differs(self, tmp_path):
        import dataclasses

        from mimarsinan.mapping.export.law_stamp import core_channels
        from mimarsinan.mapping.lanes.law.stamp import LaneStampError, _channel_arrays

        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.MaxPool2d(2), nn.ReLU(), nn.Flatten(),
                              nn.Linear(8 * 16, 10))
        program = _build_with_entry(model, (3, 8, 8), num_classes=10, classes=("L", "E", "P"), tmp_path=tmp_path,
                                    law={"adc_bits": 6, "range": "calibrated", "mult_bits": 16}, entry=NARROW_ENTRY)
        cores = _first_cores(program)
        laws = {int(c.id): c.readout_law for c in cores}
        last = cores[-1]
        laws[int(last.id)] = dataclasses.replace(last.readout_law, adc_shift=last.readout_law.adc_shift + 1)
        with pytest.raises(LaneStampError, match="adc_shift") as raised:
            _channel_arrays(cores, laws, 8, program.ir, operand=False, column_channels=core_channels)
        assert "fused operand" not in str(raised.value), "no operand differs"


def _pruned_lane_model():
    torch.manual_seed(3)
    return nn.Sequential(nn.Conv2d(1, 6, 3, padding=1), nn.MaxPool2d(2), nn.ReLU(), nn.Conv2d(6, 8, 3), nn.ReLU(),
                         nn.Flatten(), nn.Linear(8 * 2 * 2, 4)).eval()


def _pruned_lane_program(tmp_path, offset, *, lane_range=3.875, prune=True):
    """V3b: the conv's lane (read by a host MaxPool) on a declared offset grid of 5 bits (codes 0..31)."""
    platform = lane_platform(("L", "E", "P"), activation_bits=5)

    def declare(flow, entries, grids, perceptrons, quantizers, lanes):
        platform["quantizer_state"] = write_state(Path(tmp_path) / "state.json",
                                                  {"_0": {"readout_range": lane_range, "readout_offset": offset}})

    return build_lane_program(_pruned_lane_model(), (1, 8, 8), num_classes=4, platform=platform,
                              prune={0: [1, 3, 5]} if prune else None, hardware_bias=True, state_hook=declare)


class TestN2N3PrunedOffsetLane:
    @pytest.mark.parametrize("steps", [-2, 34, 2.5], ids=["below_codes", "above_codes", "non_integral"])
    def test_v3b_a_pruned_program_beside_an_offset_lane_whose_zero_is_no_code_is_refused(self, tmp_path, steps):
        from mimarsinan.mapping.lanes.law.lane_grids import PrunedOffsetGridError

        with pytest.raises(PrunedOffsetGridError, match="value 0 is no code"):
            _pruned_lane_program(tmp_path, steps * 0.125)

    @pytest.mark.parametrize("prune", [True, False], ids=["pruned", "unpruned"])
    def test_v3b_an_offset_lane_whose_zero_is_a_code_certifies(self, tmp_path, prune):
        program = _pruned_lane_program(tmp_path, 2 * 0.125, prune=prune)
        _certify(program, real_inputs(program, 64), axons=256, neurons=64)

    def test_v3b_without_pruning_an_offset_outside_the_codes_certifies(self, tmp_path):
        program = _pruned_lane_program(tmp_path, -2 * 0.125, prune=False)
        _certify(program, real_inputs(program, 64), axons=256, neurons=64)

    @pytest.mark.parametrize("lane_range, bits, offset, exact", [
        (7.0, 3, 2.0, True), (7.0, 3, 7.0, True), (7.0, 3, 0.25, False), (7.0, 3, -2.0, False), (7.0, 3, 9.0, False),
        (3.1, 5, 0.2, True), (3.1, 5, 3.4, False), (3.1, 5, -0.2, False), (float(np.float32(3.1)), 5, 0.2, False),
    ])
    def test_value_zero_is_decided_by_the_lanes_own_rounding_and_clamp(self, lane_range, bits, offset, exact):
        """Unsigned grids (every offset lane's): value 0 is a code exactly where the lane's own snap of 0 is 0, so
        only whole steps inside the codes 0..2^b - 1 pass; a range stored as float32 (a quantizer buffer) has its own
        step, on which 0.2 is no whole number of steps."""
        from types import SimpleNamespace

        from mimarsinan.mapping.lanes.law.lane_grids import LaneGrid, PrunedOffsetGridError, refuse_pruned_offset_lanes
        from mimarsinan.models.nn.activations.lane_quantization import quantize_to_lane_grid

        pruned = [SimpleNamespace(pre_pruning_row_mask=[False, True])]
        grid = LaneGrid(lane_range, offset, True)
        refuse_pruned_offset_lanes([SimpleNamespace()], {3: grid}, bits)  # not pruned: nothing to refuse
        snapped = float(quantize_to_lane_grid(torch.zeros(1, dtype=torch.float64), lane_range, bits, offset)[0])
        assert (snapped == 0.0) == exact, (lane_range, offset, snapped)
        if exact:
            refuse_pruned_offset_lanes(pruned, {3: grid}, bits)
        else:
            with pytest.raises(PrunedOffsetGridError, match="value 0 is no code"):
                refuse_pruned_offset_lanes(pruned, {3: grid}, bits)
