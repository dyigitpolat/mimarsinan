"""N7 (11_wave2/lanes/verify2_report.md section 3): a pruned program beside an offset grid whose value 0 is a whole-step
code k. Pruning drops the rows that read a pruned channel; the model still reads that channel's code k. The integer
image of value 0 is exactly 0 (``M_o = k M0``, so ``W (M0 k - M_o) = 0`` in z), the accumulator bounds cover the k term
the model's accumulator keeps, and a reader whose ADC rounds that term is refused by name. Certificates: NF = SCM =
HCM = int64 golden at atol 0 over 2,048 inputs (verify2's W1 arms), or the refusal."""

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
from mimarsinan.mapping.ir.core_law import CoreLaw
from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
from mimarsinan.models.nn.activations.integer_law import accumulator_bounds
from unit.chip_simulation.value_run.integer_programs import assert_exact
from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform, real_inputs, write_state

RANGE, BITS = 3.875, 5  # unsigned codes 0..31, step 0.125 (exact in binary)
STEP = RANGE / (2 ** BITS - 1)
PRUNED = [1, 3, 5]  # conv1's output channels zeroed before mapping
LANES = ("L", "E", "P")
#: verify2's W1 arms: conv1's offset lane read on chip by conv2 or by the output linear, or conv1, host MaxPool, conv2
#: through a declared whole-step offset entry grid (lanes on, and lanes off: the plain law solver).
ARMS = {"lane_direct": LANES, "linear_reader": LANES, "entry_host": LANES, "entry_host_plain": None}
CASES = [(3, 2), (3, 5), (4, 2), (4, 5)]  # (seed, k): offsets of k whole steps, inside the codes
#: The entry grid of an entry_host arm: the lane chip's DAC reads unsigned codes 0..31; without lanes the signed
#: 5-bit DAC reads up to 15, so the plain arm's grid holds codes 0..15 on the same step.
ENTRY = {"entry_host": {"scale": RANGE, "codes": [0, 31]}, "entry_host_plain": {"scale": 15 * STEP, "codes": [0, 15]}}


def _input_levels(arm: str) -> int:
    return ENTRY[arm]["codes"][1] if arm in ENTRY else 2 ** BITS - 1


def _model(arm: str, seed: int, *, zero_readers: bool = False) -> nn.Module:
    torch.manual_seed(seed)
    if arm == "lane_direct":
        model = nn.Sequential(nn.Conv2d(1, 6, 3, padding=1), nn.Conv2d(6, 8, 3), nn.ReLU(), nn.Flatten(),
                              nn.Linear(8 * 6 * 6, 4))
    elif arm == "linear_reader":
        model = nn.Sequential(nn.Conv2d(1, 6, 3, padding=1), nn.Flatten(), nn.Linear(6 * 64, 4))
    else:
        model = nn.Sequential(nn.Conv2d(1, 6, 3, padding=1), nn.MaxPool2d(2), nn.Conv2d(6, 8, 3), nn.ReLU(),
                              nn.Flatten(), nn.Linear(8 * 2 * 2, 4))
    if zero_readers:  # the reader's weights on the pruned channels are 0 (a coupled mask): the dropped rows add 0
        reader = model[1 if arm == "lane_direct" else 2]
        with torch.no_grad():
            if arm == "linear_reader":
                for c in PRUNED:
                    reader.weight[:, c * 64:(c + 1) * 64] = 0.0
            else:
                reader.weight[:, PRUNED] = 0.0
    return model.eval()


def _build(tmp_path, arm: str, seed: int, k: int, *, mult_bits: int, prune: bool, adc_bits: int = 40,
           zero_readers: bool = False):
    law = {"adc_bits": adc_bits, "range": "calibrated", "mult_bits": mult_bits}
    platform = lane_platform(ARMS[arm], activation_bits=BITS, readout_law=law)

    def declare(flow, entries, grids, perceptrons, quantizers, lanes):
        if arm.startswith("entry_host"):
            fx = {id(p): n for n, p in perceptron_fx_names(flow.get_mapper_repr()).items()}
            body = {fx[id(entries[1])]: {"entry": {**ENTRY[arm], "offset": k * STEP, "rounding": "nearest"}}}
        else:
            body = {"_0": {"readout_range": RANGE, "readout_offset": k * STEP}}
        platform["quantizer_state"] = write_state(Path(tmp_path) / "state.json", body)

    return build_lane_program(_model(arm, seed, zero_readers=zero_readers), (1, 8, 8), num_classes=4,
                              platform=platform, prune={0: PRUNED} if prune else None, hardware_bias=True, seed=seed,
                              weight_bits=4, state_hook=declare)


def _readers(program):
    """The cores of the perceptron reading the offset grid (perceptron 1), with their offset."""
    return [n for n in program.ir.nodes if isinstance(n, NeuralCore) and n.perceptron_index == 1
            and float(n.readout_law.input_offset[0] if hasattr(n.readout_law, "input_offset")
                      else n.boundary_grid.entry_offset()) != 0.0]


def _certify(program, n: int = 2048) -> None:
    """NF = SCM = HCM = int64 golden at atol 0 over ``n`` inputs."""
    with torch.no_grad():
        mapping = program.packed_mapping(axons=512, neurons=64)
        hcm_flow = ValueHybridCoreFlow(mapping, dtype=torch.float64)
        for chunk in range(n // 512):
            x = real_inputs(program, 512, seed=100 + chunk)
            scm = program.identity()(x)
            assert_exact(program.nf()(x), scm)
            hcm = hcm_flow(x)
            assert_exact(hcm, scm)
            codes, steps = run_golden(mapping, x.numpy())
            assert np.array_equal(codes * steps, hcm.reshape(codes.shape).numpy())


class TestWholeStepOffsetMultiplier:
    @pytest.mark.parametrize("offset, step, code", [
        (0.25, 0.125, 2), (0.625, 0.125, 5), (-0.25, 0.125, -2), (0.0, 0.125, 0), (0.3, 0.125, None),
        (2.5 * 0.125, 0.125, None), (0.2, 3.1 / 31, 2), (0.2, float(np.float32(3.1)) / 31, None),
    ])
    def test_the_code_of_value_zero_is_k_exactly_where_the_offset_is_k_whole_steps(self, offset, step, code):
        """``k step == offset`` in fp64 (the grids' own test of value 0: ``code * step - offset``); a range stored in
        float32 has its own step, on which 0.2 is no whole number of steps."""
        from mimarsinan.mapping.export.entry_offsets import whole_step_code

        assert whole_step_code(offset, step) == code

    def test_a_whole_step_offset_multiplier_is_k_m0_and_any_other_keeps_its_rounding(self):
        from mimarsinan.mapping.export.entry_offsets import offset_multipliers

        def law(step):
            return CoreLaw(input_step=[step] * 3, m0=[37, 41, 1023], shift=[9, 9, 12], bias=[0, 5, -3], lo=[0] * 3,
                           hi=[31] * 3, output_step=[0.11, 0.07, 0.013], adc_shift=[0] * 3, adc_levels=[4095] * 3)

        whole = offset_multipliers(law(0.125), 1.7, 5 * 0.125)
        assert whole.tolist() == [5 * 37, 5 * 41, 5 * 1023]
        rounded = offset_multipliers(law(0.125), 1.7, 0.3)
        expected = np.round(np.ldexp(0.3 / (1.7 * np.array([0.11, 0.07, 0.013])), [9, 9, 12]))
        assert rounded.tolist() == expected.astype(np.int64).tolist()


class TestW1ArmsCertify:
    @pytest.mark.parametrize("seed, k", CASES, ids=[f"seed{s}_k{k}" for s, k in CASES])
    @pytest.mark.parametrize("prune", [True, False], ids=["pruned", "unpruned"])
    @pytest.mark.parametrize("mult_bits", [6, 8, 12])
    @pytest.mark.parametrize("arm", list(ARMS))
    def test_a_pruned_program_beside_a_whole_step_offset_grid_certifies_over_2048_inputs(
            self, tmp_path, arm, mult_bits, prune, seed, k):
        """verify2's W1: before the fix up to 704 of 2,048 inputs differed on the R-edge (NF against SCM) while the
        C-edge and the golden agreed. Every reader column's integer image of value 0 is now 0."""
        from mimarsinan.mapping.export.entry_offsets import offset_multipliers

        program = _build(tmp_path, arm, seed, k, mult_bits=mult_bits, prune=prune)
        _certify(program)
        readers = _readers(program)
        assert readers, "the arm reads an offset grid"
        for core in readers:
            law = core.readout_law
            residue = np.asarray(law.m0, dtype=np.int64) * k - offset_multipliers(law, float(core.threshold), k * STEP)
            assert not residue.any(), (core.name, sorted(set(residue.tolist())))


class TestNarrowADCReaders:
    @pytest.mark.parametrize("arm", list(ARMS))
    def test_a_narrow_adc_reading_dropped_rows_beside_an_offset_grid_is_refused_by_name(self, tmp_path, arm):
        """The model's accumulator keeps the dropped rows' ``k W``, the chip's does not; an ADC of step 2^t > 1
        rounds the two apart whatever M_o is (before: 2,033 of 2,048 inputs differed on lane_direct, adc_bits 8)."""
        from mimarsinan.mapping.export.entry_offsets import PrunedOffsetReaderError

        with pytest.raises(PrunedOffsetReaderError, match="pruning dropped rows"):
            _build(tmp_path, arm, 3, 2, mult_bits=12, prune=True, adc_bits=8)

    @pytest.mark.parametrize("arm", list(ARMS))
    def test_unpruned_narrow_adc_programs_beside_an_offset_grid_certify(self, tmp_path, arm):
        _certify(_build(tmp_path, arm, 3, 2, mult_bits=12, prune=False, adc_bits=8))

    @pytest.mark.parametrize("arm", list(ARMS))
    def test_dropped_rows_whose_weights_are_zero_leave_a_narrow_adc_exact(self, tmp_path, arm):
        """A coupled mask zeroes the reader's weights on the pruned channels: the dropped rows add 0 in both
        programs, so the narrow ADC is exact and nothing is refused."""
        _certify(_build(tmp_path, arm, 3, 2, mult_bits=12, prune=True, adc_bits=8, zero_readers=True))


class TestDroppedRowReach:
    def test_compaction_records_the_dropped_rows_weight_per_column(self, tmp_path):
        """linear_reader's output linear owns its matrix: compaction drops the 192 rows reading the pruned channels
        and records their per-column ``Σ|W|``, which the model's accumulator still holds at code k."""
        from mimarsinan.mapping.export.entry_offsets import dropped_row_magnitude

        unpruned = _build(tmp_path, "linear_reader", 3, 5, mult_bits=12, prune=False)
        pruned = _build(tmp_path, "linear_reader", 3, 5, mult_bits=12, prune=True)
        (full,), (kept,) = _readers(unpruned), _readers(pruned)
        rows = full.get_input_count() - kept.get_input_count()
        assert rows == 3 * 64
        magnitude, counted = dropped_row_magnitude(kept, pruned.ir)
        assert not counted, "an owned matrix's dropped rows are not in its bounds"
        _bias, before = core_bias_and_data_weights(full, unpruned.ir)
        _bias, after = core_bias_and_data_weights(kept, pruned.ir)
        expected = np.abs(before).sum(axis=0) - np.abs(after).sum(axis=0)
        assert magnitude.tolist() == expected.astype(np.int64).tolist() and magnitude.any()

    @pytest.mark.parametrize("arm", ["lane_direct", "linear_reader", "entry_host", "entry_host_plain"])
    def test_every_pruned_reader_adc_spans_the_accumulators_the_model_reaches(self, tmp_path, arm):
        """``|acc| <= L Σ_kept |W| + k Σ_dropped |W|`` on the model; the law's ADC reads every such accumulator
        unchanged (step 1, levels at least that bound)."""
        from mimarsinan.mapping.export.entry_offsets import dropped_row_magnitude

        program = _build(tmp_path, arm, 4, 5, mult_bits=8, prune=True)
        for core in _readers(program):
            law = core.readout_law
            _bias, weights = core_bias_and_data_weights(core, program.ir)
            bounds = np.asarray(accumulator_bounds(weights, _input_levels(arm)), dtype=np.int64)
            magnitude, counted = dropped_row_magnitude(core, program.ir)
            reach = bounds + (0 if counted else 5 * magnitude)
            assert (np.asarray(law.adc_shift) == 0).all(), core.name
            assert (np.asarray(law.adc_levels, dtype=np.int64) >= reach).all(), core.name

    def test_the_record_follows_a_later_column_compaction_and_adds_later_drops(self):
        from types import SimpleNamespace

        from mimarsinan.mapping.pruning.pruning_apply import PRUNED_ROW_MAGNITUDE, record_dropped_rows

        node = SimpleNamespace()
        record_dropped_rows(node, np.zeros((3, 4)), [])
        assert not hasattr(node, PRUNED_ROW_MAGNITUDE), "nothing recorded where nothing was dropped"
        record_dropped_rows(node, np.array([[1, -2, 3, 0], [4, 5, -6, 7], [0, 0, 0, 0]]), [0, 2])
        assert getattr(node, PRUNED_ROW_MAGNITUDE).tolist() == [1, 2, 3, 0]
        record_dropped_rows(node, np.array([[-1, 1], [9, 9]]), [0], keep_cols=[1, 3])
        assert getattr(node, PRUNED_ROW_MAGNITUDE).tolist() == [2 + 1, 0 + 1]
