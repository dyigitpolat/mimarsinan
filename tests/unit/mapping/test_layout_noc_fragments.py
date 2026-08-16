"""N2 — shape-only NoC fragments: the wire census off the layout walk and
per-pass softcore placements off the packer.

The census counts DISTINCT producer cells per (producer, consumer) softcore
pair — the message-generation basis (one message per firing source neuron per
destination core) — plus per-consumer network-input and always-on cells. The
packer exposes which hardcore each softcore landed on, with split/coalescing
fragments carrying their origin index, so the NoC estimator can place traffic
on the same mesh the trace analysis measures.
"""

import numpy as np

from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_source_view import LayoutSourceView
from mimarsinan.mapping.layout.layout_source_view_ops import concat_source_views
from mimarsinan.mapping.layout.layout_types import (
    LayoutHardCoreType,
    LayoutSoftCoreSpec,
)
from mimarsinan.mapping.noc.wire_census import LayoutWireCensus, census_of_walk
from mimarsinan.mapping.noc import collect_noc_fragments


def _input_view(size):
    return LayoutSourceView.from_producer(producer_node_id=-2, shape=(size,))


def _mapping(**kwargs):
    kwargs.setdefault("collect_wire_census", True)
    return LayoutIRMapping(max_axons=None, max_neurons=None, **kwargs)


class TestWireCensus:
    def test_an_uncollected_census_refuses_loud(self):
        m = _mapping(collect_wire_census=False)
        m.add_neural_core(input_sources=_input_view(2), weights=np.zeros((1, 2)))
        try:
            census_of_walk(m)
        except ValueError as exc:
            assert "collect_wire_census" in str(exc)
        else:
            raise AssertionError("uncollected census answered silently")

    def test_network_input_cells_are_input_wires(self):
        m = _mapping()
        m.add_neural_core(input_sources=_input_view(4), weights=np.zeros((3, 4)))
        census = census_of_walk(m)
        assert census.input_wires == (4,)
        assert census.pair_wires == {}
        assert census.on_wires == (0,)

    def test_producer_consumer_wires_count_distinct_neurons(self):
        m = _mapping()
        a = m.add_neural_core(input_sources=_input_view(4), weights=np.zeros((5, 4)))
        m.add_neural_core(input_sources=a[0:3], weights=np.zeros((2, 3)))
        assert census_of_walk(m).pair_wires == {(0, 1): 3}

    def test_duplicate_cells_count_once(self):
        """A source neuron wired to two axons of ONE consumer core still sends
        one message per spike — messages are per (neuron, destination core)."""
        m = _mapping()
        a = m.add_neural_core(input_sources=_input_view(2), weights=np.zeros((4, 2)))
        dup = concat_source_views([a[0:2], a[0:2]])
        m.add_neural_core(input_sources=dup, weights=np.zeros((1, 4)))
        assert census_of_walk(m).pair_wires == {(0, 1): 2}

    def test_two_consumers_of_one_producer_are_two_pairs(self):
        m = _mapping()
        a = m.add_neural_core(input_sources=_input_view(2), weights=np.zeros((4, 2)))
        m.add_neural_core(input_sources=a[0:2], weights=np.zeros((1, 2)))
        m.add_neural_core(input_sources=a[2:4], weights=np.zeros((1, 2)))
        assert census_of_walk(m).pair_wires == {(0, 1): 2, (0, 2): 2}

    def test_compute_op_boundary_cells_are_input_wires(self):
        """A host op breaks the on-chip pair: its consumer re-enters as
        segment input, never as mesh traffic from the producer core."""
        m = _mapping()
        a = m.add_neural_core(input_sources=_input_view(2), weights=np.zeros((4, 2)))
        host = m.add_compute_op(a, op_type="softmax")
        m.add_neural_core(input_sources=host, weights=np.zeros((2, 4)))
        census = census_of_walk(m)
        assert census.pair_wires == {}
        assert census.input_wires == (2, 4)

    def test_bias_axon_is_an_on_wire(self):
        m = _mapping()
        m.add_neural_core(
            input_sources=_input_view(3), weights=np.zeros((2, 3)),
            biases=np.zeros(2),
        )
        assert census_of_walk(m).on_wires == (1,)

    def test_hardware_bias_adds_no_on_wire(self):
        m = _mapping(hardware_bias=True)
        m.add_neural_core(
            input_sources=_input_view(3), weights=np.zeros((2, 3)),
            biases=np.zeros(2),
        )
        assert census_of_walk(m).on_wires == (0,)

    def test_shared_bank_cores_census_like_owned_ones(self):
        m = _mapping()
        a = m.add_neural_core(input_sources=_input_view(2), weights=np.zeros((4, 2)))
        bank = m.register_weight_bank(np.zeros((3, 4)))
        m.add_shared_neural_core(
            input_sources=a[0:4], weight_bank_id=bank, has_bias=False,
        )
        assert census_of_walk(m).pair_wires == {(0, 1): 4}


def _spec(ax, ne, name=None):
    return LayoutSoftCoreSpec(input_count=ax, output_count=ne, name=name)


def _types(ax=8, ne=8, count=4):
    return [LayoutHardCoreType(max_axons=ax, max_neurons=ne, count=count)]


class TestPackerPlacements:
    def test_placements_cover_every_softcore_deterministically(self):
        specs = [_spec(8, 8), _spec(8, 8), _spec(4, 4)]
        r1 = pack_layout(softcores=specs, core_types=_types(),
                         collect_placements=True)
        r2 = pack_layout(softcores=specs, core_types=_types(),
                         collect_placements=True)
        assert r1.feasible
        assert r1.placements == r2.placements
        assert sorted(origin for origin, _ in r1.placements) == [0, 1, 2]
        assert all(0 <= hc < r1.cores_used for _, hc in r1.placements)

    def test_placements_are_absent_unless_collected(self):
        r = pack_layout(softcores=[_spec(4, 4)], core_types=_types())
        assert r.placements is None

    def test_split_fragments_carry_their_origin(self):
        r = pack_layout(
            softcores=[_spec(4, 12)], core_types=_types(count=2),
            allow_neuron_splitting=True, collect_placements=True,
        )
        assert r.feasible
        assert [origin for origin, _ in r.placements] == [0, 0]
        assert len({hc for _, hc in r.placements}) == 2

    def test_coalescing_fragments_carry_their_origin(self):
        r = pack_layout(
            softcores=[_spec(12, 4)], core_types=_types(count=2),
            allow_coalescing=True, collect_placements=True,
        )
        assert r.feasible
        assert [origin for origin, _ in r.placements] == [0, 0]


class TestNocFragmentsCollection:
    def _census(self, n):
        return LayoutWireCensus(
            pair_wires={}, input_wires=(0,) * n, on_wires=(0,) * n,
        )

    def test_single_pass_collects_one_placement_set(self):
        specs = [_spec(8, 8, "a"), _spec(8, 8, "b")]
        frags = collect_noc_fragments(
            softcores=specs, core_types=_types(),
            census=self._census(2),
            allow_scheduling=False, allow_neuron_splitting=False,
            allow_coalescing=False, schedule_policy="pool",
            max_schedule_passes=8,
        )
        assert len(frags.pass_placements) == 1
        assert sorted(o for o, _ in frags.pass_placements[0]) == [0, 1]
        assert frags.census.input_wires == (0, 0)

    def test_scheduled_passes_partition_the_softcores(self):
        """A 4-softcore program on a 2-core chip schedules into passes; every
        softcore appears exactly once, placed within its own pass."""
        specs = [_spec(8, 8, f"s{i}") for i in range(4)]
        frags = collect_noc_fragments(
            softcores=specs, core_types=_types(count=2),
            census=self._census(4),
            allow_scheduling=True, allow_neuron_splitting=False,
            allow_coalescing=False, schedule_policy="pool",
            max_schedule_passes=8,
        )
        assert len(frags.pass_placements) == 2
        all_origins = sorted(
            o for placements in frags.pass_placements for o, _ in placements
        )
        assert all_origins == [0, 1, 2, 3]
        for placements in frags.pass_placements:
            assert all(0 <= hc < 2 for _, hc in placements)
