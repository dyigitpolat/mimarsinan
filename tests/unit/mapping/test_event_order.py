"""The canonical event order: one home for slot order, bias tail, row pairs, adjacency."""

import pytest

from mimarsinan.mapping.platform.event_order import (
    bias_tail_slots,
    canonical_slot_order,
    drain_events,
    excitatory_row,
    inhibitory_row,
    is_inhibitory_row,
    logical_slot_of_row,
    normalize_event_counts,
)


class TestSlotOrder:
    def test_order_is_ascending_and_complete(self):
        assert list(canonical_slot_order(4)) == [0, 1, 2, 3]

    def test_zero_slots_is_empty(self):
        assert list(canonical_slot_order(0)) == []

    def test_negative_slot_count_refuses(self):
        with pytest.raises(ValueError, match="slot"):
            canonical_slot_order(-1)


class TestBiasTail:
    def test_bias_rows_occupy_the_tail(self):
        assert list(bias_tail_slots(6, 2)) == [4, 5]

    def test_no_bias_rows_is_empty(self):
        assert list(bias_tail_slots(6, 0)) == []

    def test_more_bias_rows_than_slots_refuses(self):
        with pytest.raises(ValueError, match="bias"):
            bias_tail_slots(2, 3)


class TestRowPairs:
    def test_pair_layout_is_even_excitatory_odd_inhibitory(self):
        assert excitatory_row(3) == 6
        assert inhibitory_row(3) == 7
        assert not is_inhibitory_row(6)
        assert is_inhibitory_row(7)

    def test_logical_of_physical_inverts_both_pair_members(self):
        for slot in range(8):
            assert logical_slot_of_row(excitatory_row(slot)) == slot
            assert logical_slot_of_row(inhibitory_row(slot)) == slot

    def test_negative_indices_refuse(self):
        with pytest.raises(ValueError, match="slot"):
            excitatory_row(-1)
        with pytest.raises(ValueError, match="row"):
            logical_slot_of_row(-2)


class TestNormalizeAndDrain:
    def test_counts_accumulate_per_slot(self):
        counts = normalize_event_counts([(0, 1), (2, 3), (0, 2)], n_slots=4)
        assert counts == [3, 0, 3, 0]

    def test_out_of_range_slot_refuses_by_name(self):
        with pytest.raises(ValueError, match="slot 4"):
            normalize_event_counts([(4, 1)], n_slots=4)

    def test_negative_multiplicity_refuses(self):
        with pytest.raises(ValueError, match="multiplicity"):
            normalize_event_counts([(1, -1)], n_slots=4)

    def test_drain_is_ascending_with_one_entry_per_active_slot(self):
        drained = drain_events([3, 0, 3, 0])
        assert drained == [(0, 3), (2, 3)]
        slots = [s for s, _ in drained]
        assert slots == sorted(slots)
        assert len(slots) == len(set(slots))

    def test_interleaved_and_adjacent_streams_normalize_identically(self):
        # Adjacency is count-changing at the fold (theta=5, w=[+3,-3], e=[2,1]:
        # adjacent order fires once, sweep order fires zero times), so the wire
        # contract is: normalize to per-slot counts, then drain ascending with
        # each slot's multiplicity adjacent. Both stream shapes must land on the
        # same drain.
        sweep_interleaved = [(0, 1), (1, 1), (0, 1)]
        adjacent = [(0, 2), (1, 1)]
        n = 2
        assert normalize_event_counts(sweep_interleaved, n_slots=n) == \
            normalize_event_counts(adjacent, n_slots=n)
        assert drain_events(normalize_event_counts(sweep_interleaved, n_slots=n)) == \
            [(0, 2), (1, 1)]

    def test_drain_normalize_round_trip(self):
        counts = [0, 2, 1, 0, 5]
        assert normalize_event_counts(drain_events(counts), n_slots=5) == counts
