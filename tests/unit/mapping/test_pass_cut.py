"""The true graph cut of a segment split into passes: what must cross, and for how long."""

import pytest

from mimarsinan.mapping.support.schedule.pass_cut import (
    COLLAPSE,
    VERBATIM,
    CutNode,
    PassCut,
    raster_bytes,
)


def _chain():
    """A -> B -> C -> D, one core per latency group, widths 8/4/16/2."""
    return [
        CutNode(core_id=0, latency=0, sources=(), out_width=8),
        CutNode(core_id=1, latency=1, sources=(0,), out_width=4),
        CutNode(core_id=2, latency=2, sources=(1,), out_width=16),
        CutNode(core_id=3, latency=3, sources=(2,), out_width=2),
    ]


class TestTheCut:
    def test_one_pass_carries_nothing(self):
        cut = PassCut.over(_chain(), [[0, 1, 2, 3]])
        assert cut.carried == ()
        assert cut.pass_count == 1

    def test_a_cut_carries_exactly_the_edges_that_cross_it(self):
        cut = PassCut.over(_chain(), [[0, 1], [2, 3]])
        assert [w.producer for w in cut.carried] == [1]
        assert cut.carried[0].width == 4

    def test_the_carried_wire_records_where_it_is_produced_and_last_read(self):
        cut = PassCut.over(_chain(), [[0], [1], [2], [3]])
        by_producer = {w.producer: w for w in cut.carried}
        assert by_producer[0].produced_in == 0
        assert by_producer[0].last_consumed_in == 1

    def test_a_wire_read_two_passes_later_stays_live_in_between(self):
        """A -> D skip: the wire must survive the passes that do not read it."""
        nodes = [
            CutNode(core_id=0, latency=0, sources=(), out_width=8),
            CutNode(core_id=1, latency=1, sources=(0,), out_width=4),
            CutNode(core_id=2, latency=2, sources=(1,), out_width=16),
            CutNode(core_id=3, latency=3, sources=(2, 0), out_width=2),
        ]
        cut = PassCut.over(nodes, [[0], [1], [2], [3]])
        skip = next(w for w in cut.carried if w.producer == 0)
        assert skip.last_consumed_in == 3
        assert skip.live_boundaries == (0, 1, 2)

    def test_a_producer_read_by_several_later_passes_is_carried_once(self):
        nodes = [
            CutNode(core_id=0, latency=0, sources=(), out_width=8),
            CutNode(core_id=1, latency=1, sources=(0,), out_width=4),
            CutNode(core_id=2, latency=1, sources=(0,), out_width=4),
        ]
        cut = PassCut.over(nodes, [[0], [1], [2]])
        assert len([w for w in cut.carried if w.producer == 0]) == 1

    def test_a_wire_inside_one_pass_is_never_carried(self):
        cut = PassCut.over(_chain(), [[0, 1, 2], [3]])
        assert [w.producer for w in cut.carried] == [2]


class TestASplitLatencyGroup:
    """The owner's 'incomplete latency groups': halving a group is legal because
    its cores are mutually independent, but the group's own INPUTS must then be
    carried into the pass holding the second half."""

    def _split_group(self):
        # A -> {B1, B2} -> C, with the B group halved across two passes.
        return [
            CutNode(core_id=0, latency=0, sources=(), out_width=8),
            CutNode(core_id=1, latency=1, sources=(0,), out_width=4),
            CutNode(core_id=2, latency=1, sources=(0,), out_width=4),
            CutNode(core_id=3, latency=2, sources=(1, 2), out_width=2),
        ]

    def test_the_groups_input_is_carried_into_the_second_half(self):
        cut = PassCut.over(self._split_group(), [[0, 1], [2], [3]])
        assert 0 in {w.producer for w in cut.carried}, (
            "B2 still needs A's spikes; carrying only the last group would "
            "starve the second half of a split group")

    def test_both_halves_are_carried_to_their_shared_consumer(self):
        cut = PassCut.over(self._split_group(), [[0, 1], [2], [3]])
        assert {1, 2} <= {w.producer for w in cut.carried}

    def test_the_split_halves_do_not_carry_to_each_other(self):
        """Same latency group = mutually independent, so no edge crosses."""
        cut = PassCut.over(self._split_group(), [[0, 1], [2], [3]])
        assert all(
            not (w.producer == 1 and w.last_consumed_in == 1) for w in cut.carried
        )


class TestRefusals:
    def test_a_core_placed_after_its_consumer_fails_loud(self):
        """A pass assignment must be monotone over the DAG, or it is not a
        schedule at all — the consumer would run before its input exists."""
        with pytest.raises(ValueError, match="monotone"):
            PassCut.over(_chain(), [[1], [0], [2], [3]])

    def test_a_core_assigned_twice_fails_loud(self):
        with pytest.raises(ValueError, match="exactly once"):
            PassCut.over(_chain(), [[0, 1], [1, 2, 3]])

    def test_a_core_assigned_to_no_pass_fails_loud(self):
        with pytest.raises(ValueError, match="exactly once"):
            PassCut.over(_chain(), [[0, 1], [2]])

    def test_an_unknown_core_id_fails_loud(self):
        with pytest.raises(ValueError, match="not a core"):
            PassCut.over(_chain(), [[0, 1], [2, 3, 99]])

    def test_an_empty_pass_fails_loud(self):
        with pytest.raises(ValueError, match="empty"):
            PassCut.over(_chain(), [[0, 1], [], [2, 3]])


class TestTheCensus:
    def test_a_raster_costs_one_bit_per_neuron_per_timestep(self):
        assert raster_bytes(16, 32) == 16 * 4
        assert raster_bytes(1, 1) == 1, "a partial byte still occupies one"

    def test_carried_bytes_sum_every_carried_wire(self):
        cut = PassCut.over(_chain(), [[0], [1], [2], [3]])
        assert cut.carried_bytes(32) == raster_bytes(8 + 4 + 16, 32)

    def test_peak_live_is_the_worst_boundary_not_the_total(self):
        """Wires whose live ranges do not overlap share the buffer."""
        cut = PassCut.over(_chain(), [[0], [1], [2], [3]])
        assert cut.peak_live_bytes(32) == raster_bytes(16, 32)
        assert cut.peak_live_bytes(32) < cut.carried_bytes(32)

    def test_an_overlapping_live_range_adds_to_the_peak(self):
        nodes = [
            CutNode(core_id=0, latency=0, sources=(), out_width=8),
            CutNode(core_id=1, latency=1, sources=(0,), out_width=4),
            CutNode(core_id=2, latency=2, sources=(1, 0), out_width=2),
        ]
        cut = PassCut.over(nodes, [[0], [1], [2]])
        # At boundary 1 both A (skipping to C) and B are live.
        assert cut.peak_live_bytes(32) == raster_bytes(8 + 4, 32)

    def test_one_pass_costs_nothing(self):
        cut = PassCut.over(_chain(), [[0, 1, 2, 3]])
        assert cut.carried_bytes(32) == 0
        assert cut.peak_live_bytes(32) == 0

    def test_cut_width_is_reported_per_boundary(self):
        cut = PassCut.over(_chain(), [[0], [1], [2], [3]])
        assert [cut.cut_width(b) for b in range(3)] == [8, 4, 16]


class TestTheTransferDiscipline:
    def test_streamed_carries_the_raster_verbatim(self):
        from mimarsinan.mapping.support.schedule.pass_cut import transfer_for

        assert transfer_for(streamed=True) == VERBATIM

    def test_a_windowed_discipline_collapses_to_counts(self):
        from mimarsinan.mapping.support.schedule.pass_cut import transfer_for

        assert transfer_for(streamed=False) == COLLAPSE


class TestTheContractSelectsTheDiscipline:
    """The semantics decides; the schedule package owns the vocabulary."""

    def _contract(self, streamed: bool):
        import dataclasses

        from mimarsinan.chip_simulation.deployment_contract import (
            SpikingDeploymentContract,
        )

        fields = {f.name for f in dataclasses.fields(SpikingDeploymentContract)}
        assert "lif_streamed" in fields, fields
        blank = SpikingDeploymentContract.__new__(SpikingDeploymentContract)
        object.__setattr__(blank, "lif_streamed", streamed)
        return blank

    def test_streamed_lif_asks_for_a_verbatim_pass_boundary(self):
        assert self._contract(True).pass_boundary_transfer() == VERBATIM

    def test_a_windowed_run_keeps_the_collapse_boundary(self):
        assert self._contract(False).pass_boundary_transfer() == COLLAPSE


class TestSkewIsRefusedNotSilentlyRemoved:
    """A carried wire re-enters as a SEGMENT INPUT, aligned per consuming core;
    inside a pass the same wire is a live handoff with a fixed one-cycle delay
    and no realignment. They agree only for consumers at producer_latency + 1,
    so a cut across a skewed edge would quietly compute something else."""

    def _skewed(self):
        # A(lat 0) feeds both B(lat 1) and D(lat 3) — the deep edge is skewed.
        return [
            CutNode(core_id=0, latency=0, sources=(), out_width=8),
            CutNode(core_id=1, latency=1, sources=(0,), out_width=4),
            CutNode(core_id=2, latency=2, sources=(1,), out_width=4),
            CutNode(core_id=3, latency=3, sources=(2, 0), out_width=2),
        ]

    def test_a_skew_free_cut_is_exact(self):
        cut = PassCut.over(_chain(), [[0, 1], [2, 3]])
        assert cut.skewed_carries == ()
        cut.require_exact("seg0")

    def test_a_skewed_carried_wire_is_reported(self):
        cut = PassCut.over(self._skewed(), [[0], [1, 2, 3]])
        assert [w.producer for w in cut.skewed_carries] == [0]

    def test_the_refusal_names_the_cores_and_the_remedy(self):
        cut = PassCut.over(self._skewed(), [[0], [1, 2, 3]])
        with pytest.raises(ValueError, match="lif_depth_balancing_relays"):
            cut.require_exact("seg0")
        with pytest.raises(ValueError, match="core 0"):
            cut.require_exact("seg0")

    def test_a_skewed_edge_INSIDE_one_pass_is_not_a_carry_and_not_refused(self):
        """The fused skew is the model's own semantics; only CUTTING it lies."""
        cut = PassCut.over(self._skewed(), [[0, 1, 2, 3]])
        assert cut.skewed_carries == ()
        cut.require_exact("seg0")

    def test_a_wire_with_one_consumer_at_the_next_latency_is_skew_free(self):
        cut = PassCut.over(_chain(), [[0], [1], [2], [3]])
        assert all(w.is_skew_free for w in cut.carried)


class _Slice:
    def __init__(self, node_id):
        self.node_id = node_id


class _Stage:
    def __init__(self, segment, pass_index, inputs, outputs, kind="neural"):
        self.kind = kind
        self.schedule_segment_index = segment
        self.schedule_pass_index = pass_index
        self.input_map = [_Slice(n) for n in inputs]
        self.output_map = [_Slice(n) for n in outputs]


class TestWhichOutputsAreCarried:
    def test_a_wire_read_by_a_later_pass_of_the_same_segment_is_carried(self):
        from mimarsinan.mapping.support.schedule.pass_cut import (
            carried_outputs_by_stage,
        )

        stages = [_Stage(0, 0, [-2], [7]), _Stage(0, 1, [7], [9])]
        assert carried_outputs_by_stage(stages) == {0: (7,)}

    def test_a_wire_read_by_a_later_SEGMENT_collapses_and_is_not_carried(self):
        """That crossing is a host boundary — counts are the contract there."""
        from mimarsinan.mapping.support.schedule.pass_cut import (
            carried_outputs_by_stage,
        )

        stages = [_Stage(0, 0, [-2], [7]), _Stage(1, 0, [7], [9])]
        assert carried_outputs_by_stage(stages) == {}

    def test_a_single_pass_segment_carries_nothing(self):
        from mimarsinan.mapping.support.schedule.pass_cut import (
            carried_outputs_by_stage,
        )

        assert carried_outputs_by_stage([_Stage(0, 0, [-2], [7])]) == {}
