"""[E6] The lanes' N7 predicate under W15 (``N-identity``, rule ``dropped_offset_row``): every row pruning drops reads
integer 0. The model still reads a dropped row at the code ``k`` of value 0 on its reader's input grid; the identity
program has no such row. The two differ in ``z`` by ``(M0 k - M_o) Σ W`` over the dropped rows, so the predicate holds,
on every reader of an offset grid with dropped weight, that value 0 is a whole-step code (``k`` exists), that its
integer image ``M0 k - M_o`` is 0 on every column with dropped weight (``M_o`` by the law's own offset multiplier,
``entry_offsets.offset_multipliers``), and that the ADC passes the dropped term unchanged (identity step, levels at
least the model's reach). Vehicles: verify2's W1 arms (``test_pruned_offset_readers``)."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest
import torch

from mimarsinan.certification.program_check.report import FAILED
from mimarsinan.certification.program_check.unit_forms import pruned_rows
from mimarsinan.mapping.export.entry_offsets import (
    core_entry_offset, dropped_row_magnitude, offset_multipliers, whole_step_code,
)
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.models.nn.activations.integer_law import round_half_away

from unit.certification.program_check import programs as P
from unit.certification.program_check import unit_vehicles as UV

ARMS = ("lane_direct", "linear_reader", "entry_host", "entry_host_plain")
CASES = [(3, 2, 8), (4, 5, 8), (3, 5, 6)]  # (seed, k, mult_bits)


def _reader_offset(node: NeuralCore) -> float:
    law = node.readout_law
    return float(law.input_offset[0]) if hasattr(law, "input_offset") else core_entry_offset(node)


def _pre_n7_multipliers(law, theta, offset):
    """The rule before N7's fix: ``M_o = rnd(2^s o / (θ Δout))`` on every column, whole steps or not."""
    scaled = np.ldexp(float(offset) / (float(theta) * np.asarray(law.output_step, dtype=np.float64)), law.shift)
    return round_half_away(torch.as_tensor(scaled)).numpy().astype(np.int64)


def _residue_columns(ir, rule) -> int:
    """Columns of offset readers with dropped weight whose image of value 0, ``M0 k - M_o`` under ``rule``, is not 0."""
    count = 0
    for node in ir.get_neural_cores():
        if getattr(node, "readout_law", None) is None or _reader_offset(node) == 0.0:
            continue
        magnitude, _counted = dropped_row_magnitude(node, ir)
        law, offset = node.readout_law, _reader_offset(node)
        k = whole_step_code(offset, float(np.asarray(law.input_step).reshape(-1)[0]))
        if k is None:  # value 0 is no code of the grid: every dropped column's term is nonzero
            count += int((magnitude != 0).sum())
            continue
        residue = np.asarray(law.m0, dtype=np.int64) * int(k) - rule(law, float(node.threshold), offset)
        count += int(((magnitude != 0) & (residue != 0)).sum())
    return count


@pytest.mark.parametrize("seed,k,mult_bits", CASES)
@pytest.mark.parametrize("arm", ARMS)
def test_a_pruned_reader_beside_a_whole_step_offset_grid_reads_integer_zero(arm, seed, k, mult_bits):
    program, ir, decl = UV.pruned(arm, True, mult_bits, seed, k)
    report = P.run(program, ir, decl)
    identity = report.results["N-identity"]
    assert "dropped_offset_row" not in {v.rule for v in identity.violations}, [v.text() for v in identity.violations]
    assert identity.status != FAILED, identity.note
    assert report.counts["dropped_offset_readers"] > 0
    assert report.established("theorem1"), report.theorem1


@pytest.mark.parametrize("arm", ARMS)
def test_an_unpruned_reader_has_no_dropped_row(arm):
    program, ir, decl = UV.pruned(arm, False)
    report = P.run(program, ir, decl)
    assert report.counts["dropped_offset_readers"] == 0
    assert report.results["N-identity"].status != FAILED


@pytest.mark.parametrize("seed,k,mult_bits", CASES)
@pytest.mark.parametrize("arm", ARMS)
def test_the_pre_n7_offset_multiplier_is_named_where_its_image_of_zero_is_not_zero(arm, seed, k, mult_bits,
                                                                                 monkeypatch):
    """The rule before N7's fix, read by the predicate: the check fails ``dropped_offset_row`` exactly on the arms whose
    dropped columns' image of value 0 is nonzero under that rule."""
    program, ir, decl = UV.pruned(arm, True, mult_bits, seed, k)
    expected = _residue_columns(ir, _pre_n7_multipliers)
    assert _residue_columns(ir, offset_multipliers) == 0
    monkeypatch.setattr(pruned_rows, "offset_multipliers", _pre_n7_multipliers)
    report = P.run(program, ir, decl)
    rules = {v.rule for v in report.results["N-identity"].violations}
    assert ("dropped_offset_row" in rules) == (expected > 0), (expected, rules)
    if expected:
        assert report.proposition2.startswith("failed") and "N-identity" in report.proposition2


def test_the_pre_n7_rule_bites_on_some_arm():
    assert any(_residue_columns(UV.pruned(arm, True, m, s, k)[1], _pre_n7_multipliers)
               for arm in ARMS for s, k, m in CASES)


def _with_reader_law(arm: str, edit):
    """The pruned arm with one offset reader's law edited (``edit(law, dropped columns)``) in its IR, and the program
    rebuilt from that IR, so the deployment and its identity program agree and only the network side departs."""
    source = UV.pruned_program(arm, True)
    ir = copy.deepcopy(source.ir)
    for node in ir.get_neural_cores():
        if getattr(node, "readout_law", None) is None or _reader_offset(node) == 0.0:
            continue
        magnitude, _counted = dropped_row_magnitude(node, ir)
        if magnitude.any():
            node.readout_law = edit(node.readout_law, magnitude != 0)
            break
    else:
        raise AssertionError("the arm holds no offset reader with dropped weight")
    program = type(source)(source.model, ir, source.input_shape).packed_mapping(axons=512, neurons=64)
    return program, ir, UV.value_decl([{"max_axons": 512, "max_neurons": 64, "count": 999}])


@pytest.mark.parametrize("arm", ["lane_direct", "entry_host_plain"])
def test_a_reader_whose_adc_rounds_the_dropped_term(arm):
    def rounding(law, dropped):
        return dataclasses.replace(law, adc_shift=np.where(dropped, 1, np.asarray(law.adc_shift)))

    program, ir, decl = _with_reader_law(arm, rounding)
    report = P.run(program, ir, decl)
    assert "dropped_offset_row" in {v.rule for v in report.results["N-identity"].violations}
    assert report.results["N-identity"].status == FAILED
    assert report.established("theorem1"), report.theorem1


@pytest.mark.parametrize("arm", ["lane_direct", "entry_host_plain"])
def test_a_reader_whose_value_zero_is_no_code_of_its_grid(arm):
    def off_grid(law, _dropped):
        return dataclasses.replace(law, input_step=np.asarray(law.input_step, dtype=np.float64) * 0.75)

    program, ir, decl = _with_reader_law(arm, off_grid)
    report = P.run(program, ir, decl)
    assert "dropped_offset_row" in {v.rule for v in report.results["N-identity"].violations}
