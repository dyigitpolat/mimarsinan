"""WI-5e: ``LaneLaw`` carries its lane clauses through every column and row operation, and leaves ``CoreLaw``'s
bytes, digests and pickles as they were."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimarsinan.mapping.ir.core_law import CoreLaw, merge_core_law
from mimarsinan.mapping.ir.lane_law import LANE_FIELDS, LaneLaw, LaneLawRowError
from mimarsinan.models.nn.activations.integer_law import apply_law, apply_law_int64


def _core(columns=4) -> CoreLaw:
    return CoreLaw(input_step=np.full(columns, 0.25), m0=np.arange(1, columns + 1), shift=np.full(columns, 3),
                   bias=np.arange(columns) - 1, lo=np.full(columns, 0), hi=np.full(columns, 15),
                   output_step=np.full(columns, 0.5), adc_shift=np.zeros(columns), adc_levels=np.full(columns, 99))


def _lane(columns=4) -> LaneLaw:
    return LaneLaw.of(_core(columns), input_offset=0.75, output_offset=0.0, operand_row=np.arange(columns) + 10,
                      operand_m=np.full(columns, 7), operand_step=np.full(columns, 0.125),
                      operand_offset=np.full(columns, 1.5))


class TestCoreLawUnchanged:
    def test_the_pickle_reduces_to_the_core_builder(self):
        law = _core()
        builder, _args = law.__reduce__()
        assert builder.__name__ == "_core_law_from_buffer"
        again = pickle.loads(pickle.dumps(law))
        assert type(again) is CoreLaw and again.digest() == law.digest()

    def test_column_operations_keep_the_type(self):
        law = _core()
        assert type(law.take_columns([1, 3])) is CoreLaw
        assert type(CoreLaw.blank(3).splice(1, law.take_columns([0]))) is CoreLaw


class TestLaneLaw:
    def test_the_lane_fields_follow_the_core_fields(self):
        assert [f for f in LANE_FIELDS] == ["input_offset", "output_offset", "operand_row", "operand_m",
                                            "operand_step", "operand_offset"]
        law = _lane()
        assert law.has_operands and float(law.input_offset[0]) == 0.75

    def test_blank_has_no_operand(self):
        blank = LaneLaw.blank(3)
        assert np.array_equal(blank.operand_row, [-1, -1, -1]) and not blank.has_operands

    def test_take_splice_and_pickle_keep_every_clause(self):
        law = _lane()
        part = law.take_columns([2, 0])
        assert type(part) is LaneLaw and part.operand_row.tolist() == [12, 10]
        assert part is law.take_columns([2, 0]), "one object per distinct selection"
        spliced = LaneLaw.blank(6).splice(2, law)
        assert spliced.operand_row.tolist() == [-1, -1, 10, 11, 12, 13]
        again = pickle.loads(pickle.dumps(law))
        assert type(again) is LaneLaw and again.digest() == law.digest()
        assert again.digest() != LaneLaw.of(_core()).digest(), "the lane clauses enter the digest"

    def test_shift_rows_moves_the_operand_rows_only(self):
        law = _lane().take_columns([0, 1])
        law = LaneLaw.blank(2).splice(0, law).with_operand_rows([10, -1])
        moved = law.shift_rows(5)
        assert moved.operand_row.tolist() == [15, -1] and moved.digest() != law.digest()
        assert law.shift_rows(0) is law

    def test_keep_rows_remaps_or_refuses(self):
        law = _lane(2)
        assert law.keep_rows([0, 10, 11]).operand_row.tolist() == [1, 2]
        with pytest.raises(LaneLawRowError, match="drops the fused operand rows"):
            law.keep_rows([0, 10])

    def test_a_hard_core_takes_the_soft_core_law_at_its_axon_offset(self):
        hard = SimpleNamespace(neurons_per_core=6, readout_law=None)
        merge_core_law(hard, SimpleNamespace(id=1, readout_law=_lane(2)), axon_offset=0, neuron_offset=0)
        merge_core_law(hard, SimpleNamespace(id=2, readout_law=_lane(2)), axon_offset=40, neuron_offset=2)
        assert type(hard.readout_law) is LaneLaw
        assert hard.readout_law.operand_row.tolist() == [10, 11, 50, 51, -1, -1]


class TestOperandTerm:
    def test_the_torch_and_int64_forms_agree_with_the_operand(self):
        generator = np.random.default_rng(0)
        acc = generator.integers(-500, 500, (64, 4))
        q = generator.integers(0, 16, (64, 4))
        kwargs = dict(m0=np.array([3, 5, 7, 9]), bias=np.array([-4, 0, 4, 9]), shift=np.array([2, 3, 4, 5]),
                      lo=np.zeros(4), hi=np.full(4, 15))
        m1 = np.array([11, 2, 1, 4])
        exact = apply_law_int64(acc, **kwargs, operands=((q, m1),))
        torch_form = apply_law(torch.as_tensor(acc, dtype=torch.float64), **kwargs,
                               operands=((torch.as_tensor(q, dtype=torch.float64), m1),))
        assert np.array_equal(torch_form.numpy().astype(np.int64), exact)
        assert not np.array_equal(exact, apply_law_int64(acc, **kwargs)), "the operand moves outputs"
