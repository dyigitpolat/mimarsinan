"""[ODIN4] Row-pair expansion: signed logical matrix -> physical rows with per-row signs.

The expansion CONSUMES `mapping/platform/event_order` (plan Sec.2.3 rows 4 and 2) —
it never re-derives the pair or the bias tail — and it allocates fresh arrays,
because `HardCore.get_core_matrix()` hands back a SHARED memoized grid.
"""

import numpy as np
import pytest

from mimarsinan.mapping.export.odin.expansion import (
    RowPairExpansionError,
    expand_row_pairs,
)
from mimarsinan.mapping.export.odin.feasibility import (
    KEY_WEIGHT_MAGNITUDE_RANGE,
    OdinFeasibilityError,
)
from mimarsinan.mapping.platform.event_order import (
    bias_tail_slots,
    excitatory_row,
    inhibitory_row,
)


def _matrix(rows):
    return np.array(rows, dtype=np.int64)


class TestTheExpansionUsesTheCanonicalPairContract:
    def test_slot_a_occupies_rows_2a_and_2a_plus_one(self):
        expansion = expand_row_pairs(
            _matrix([[3, -2], [-1, 4]]), n_bias_rows=0, weight_bits=4
        )
        assert [r.row for r in expansion.rows] == [0, 1, 2, 3]
        for slot in (0, 1):
            assert expansion.rows[excitatory_row(slot)].slot == slot
            assert expansion.rows[inhibitory_row(slot)].slot == slot

    def test_the_excitatory_row_carries_the_positive_part_only(self):
        expansion = expand_row_pairs(
            _matrix([[3, -2], [-1, 4]]), n_bias_rows=0, weight_bits=4
        )
        assert list(expansion.rows[0].magnitudes) == [3, 0]
        assert list(expansion.rows[2].magnitudes) == [0, 4]

    def test_the_inhibitory_row_carries_the_negated_negative_part_only(self):
        expansion = expand_row_pairs(
            _matrix([[3, -2], [-1, 4]]), n_bias_rows=0, weight_bits=4
        )
        assert list(expansion.rows[1].magnitudes) == [0, 2]
        assert list(expansion.rows[3].magnitudes) == [1, 0]

    def test_the_per_row_sign_is_the_syn_sign_bit(self):
        expansion = expand_row_pairs(
            _matrix([[3, -2], [-1, 4]]), n_bias_rows=0, weight_bits=4
        )
        assert [r.inhibitory for r in expansion.rows] == [False, True, False, True]
        assert expansion.syn_sign_bits == (False, True, False, True)

    def test_at_most_one_member_of_a_pair_is_nonzero_per_neuron(self):
        rng = np.random.default_rng(0)
        matrix = rng.integers(-7, 8, size=(9, 5))
        expansion = expand_row_pairs(matrix, n_bias_rows=0, weight_bits=4)
        for slot in range(9):
            exc = expansion.rows[excitatory_row(slot)].magnitudes
            inh = expansion.rows[inhibitory_row(slot)].magnitudes
            assert not np.any((exc != 0) & (inh != 0))

    def test_the_signed_fold_is_recoverable_from_the_pair(self):
        rng = np.random.default_rng(1)
        matrix = rng.integers(-7, 8, size=(6, 4))
        expansion = expand_row_pairs(matrix, n_bias_rows=0, weight_bits=4)
        recovered = np.stack([
            expansion.rows[excitatory_row(a)].magnitudes.astype(np.int64)
            - expansion.rows[inhibitory_row(a)].magnitudes.astype(np.int64)
            for a in range(6)
        ])
        assert np.array_equal(recovered, matrix)


class TestBiasTailRows:
    def test_the_bias_rows_are_the_tail_slots(self):
        expansion = expand_row_pairs(
            _matrix([[1, 0], [0, 2], [3, -3]]), n_bias_rows=1, weight_bits=4
        )
        assert expansion.bias_slots == tuple(bias_tail_slots(3, 1))
        assert expansion.bias_slots == (2,)

    def test_the_bias_pair_rows_are_flagged_as_always_on(self):
        expansion = expand_row_pairs(
            _matrix([[1, 0], [0, 2], [3, -3]]), n_bias_rows=1, weight_bits=4
        )
        assert [r.always_on for r in expansion.rows] == [
            False, False, False, False, True, True
        ]

    def test_more_bias_rows_than_slots_is_refused(self):
        with pytest.raises(ValueError, match="bias rows must fit"):
            expand_row_pairs(_matrix([[1.0]]), n_bias_rows=2, weight_bits=4)


class TestAllZeroRowsEmitNothing:
    def test_a_row_with_no_magnitude_is_marked_emit_nothing(self):
        expansion = expand_row_pairs(
            _matrix([[3, 2], [-1, -4]]), n_bias_rows=0, weight_bits=4
        )
        assert [r.emits for r in expansion.rows] == [True, False, False, True]

    def test_the_emitting_rows_are_exactly_the_nonzero_ones(self):
        expansion = expand_row_pairs(
            _matrix([[3, 2], [-1, -4]]), n_bias_rows=0, weight_bits=4
        )
        assert expansion.emitting_rows_for_slot(0) == (0,)
        assert expansion.emitting_rows_for_slot(1) == (3,)

    def test_a_dead_slot_emits_nothing_at_all(self):
        expansion = expand_row_pairs(
            _matrix([[0, 0], [1, 1]]), n_bias_rows=0, weight_bits=4
        )
        assert expansion.emitting_rows_for_slot(0) == ()
        assert expansion.emitting_rows_for_slot(1) == (2,)


class TestTheSharedGridIsNeverMutated:
    def test_the_source_matrix_is_untouched(self):
        matrix = _matrix([[3, -2], [-1, 4]])
        before = matrix.copy()
        expand_row_pairs(matrix, n_bias_rows=0, weight_bits=4)
        assert np.array_equal(matrix, before)

    def test_every_emitted_row_owns_its_array(self):
        matrix = _matrix([[3, -2], [-1, 4]])
        expansion = expand_row_pairs(matrix, n_bias_rows=0, weight_bits=4)
        for row in expansion.rows:
            assert row.magnitudes.base is None or row.magnitudes.base is not matrix
            row.magnitudes[0] = 7
        assert np.array_equal(matrix, _matrix([[3, -2], [-1, 4]]))

    def test_a_read_only_input_grid_is_accepted(self):
        matrix = _matrix([[3, -2], [-1, 4]])
        matrix.setflags(write=False)
        expansion = expand_row_pairs(matrix, n_bias_rows=0, weight_bits=4)
        assert list(expansion.rows[0].magnitudes) == [3, 0]


class TestTheExpansionRefusesWhatTheHardwareCannotHold:
    def test_a_non_integral_weight_is_refused(self):
        with pytest.raises(RowPairExpansionError, match="integral"):
            expand_row_pairs(
                np.array([[1.5, 0.0]]), n_bias_rows=0, weight_bits=4
            )

    def test_a_magnitude_outside_the_symmetric_range_defers_to_the_keyed_gate(self):
        # ONE home for the range question: the expansion consults the feasibility
        # gate rather than growing a second, differently-worded refusal.
        with pytest.raises(OdinFeasibilityError, match="-8") as excinfo:
            expand_row_pairs(_matrix([[-8, 0]]), n_bias_rows=0, weight_bits=4)
        assert excinfo.value.key == KEY_WEIGHT_MAGNITUDE_RANGE

    def test_a_one_dimensional_matrix_is_refused(self):
        with pytest.raises(RowPairExpansionError, match="2-D"):
            expand_row_pairs(_matrix([1, 2, 3]), n_bias_rows=0, weight_bits=4)
