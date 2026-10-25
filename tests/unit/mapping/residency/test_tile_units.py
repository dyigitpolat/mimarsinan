"""[W03] Tile-granular residency units: a layer splits into the tile groups it packs into alone, so the floor plans
wherever a pass can hold a tile group (U1's capacity split cuts a layer too), and a fit plans in closed form."""

from __future__ import annotations

import random

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.packing.residency_tiles import tile_groups, unit_tile_count
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.mapping.support.schedule.pass_cut import CutNode, PassCut
from mimarsinan.mapping.support.schedule.residency_front import (
    FRONTS,
    Q_SPAN,
    plan_steps,
    q_fit,
)
from mimarsinan.mapping.support.schedule.residency_units import residency_steps, residency_units

CORE = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=4)]


def _spec(i, layer, axons=16, neurons=8, segment=0, latency=None):
    return LayoutSoftCoreSpec(input_count=axons, output_count=neurons, perceptron_index=layer, segment_id=segment,
                              latency_tag=layer if latency is None else latency, name=f"sc{i}")


def _layered(sizes, **shape):
    """``sizes[L]`` independent specs in layer L, each reading every spec of layer L - 1 (one segment)."""
    specs, nodes, previous = [], [], []
    for layer, size in enumerate(sizes):
        mine = []
        for _ in range(size):
            i = len(specs)
            specs.append(_spec(i, layer, **shape))
            nodes.append(CutNode(core_id=i, latency=layer, sources=tuple(previous), out_width=specs[i].output_count))
            mine.append(i)
        previous = mine
    return specs, nodes, [0] * len(specs)


def _units(specs, nodes, segments, **kw):
    return residency_units(specs, nodes, CORE, segments=segments, **kw)


class TestTileGroups:
    def test_specs_that_share_a_packed_core_form_one_group(self):
        specs = [_spec(i, 0, axons=8, neurons=4) for i in range(3)]
        # two 8x4 softcores share a 16x8 core diagonally (used 16x8), the third uses 8x4 of a core of its own
        groups = tile_groups(specs, CORE, allow_coalescing=False, allow_splitting=False)
        assert sorted((len(members), cores) for members, cores in groups) == [(1, ((32, 1),)), (2, ((128, 1),))]
        assert sorted(i for members, _ in groups for i in members) == [0, 1, 2]

    def test_a_spec_that_fills_a_core_is_a_group_of_its_own(self):
        specs = [_spec(i, 0) for i in range(3)]
        assert tile_groups(specs, CORE, allow_coalescing=False, allow_splitting=False) == tuple(
            ((i,), ((128, 1),)) for i in range(3))

    def test_a_bond_joins_specs_that_share_no_core(self):
        specs = [_spec(i, 0) for i in range(3)]
        groups = tile_groups(specs, CORE, allow_coalescing=False, allow_splitting=False, bonds=((2, 0),))
        assert groups == (((0, 2), ((128, 1), (128, 1))), ((1,), ((128, 1),)))

    def test_a_neuron_split_spec_is_one_group_over_its_cores(self):
        groups = tile_groups([_spec(0, 0, neurons=24)], CORE, allow_coalescing=False, allow_splitting=True)
        assert groups == (((0,), ((128, 1),) * 3),)

    def test_a_spec_wider_than_a_core_is_one_group_of_its_tiles(self):
        """Fused under declared coalescing (axons summed over three cores): one group, three tiles."""
        groups = tile_groups([_spec(0, 0, axons=48)], CORE, allow_coalescing=True, allow_splitting=False)
        assert len(groups) == 1 and groups[0][0] == (0,) and sum(t for _, t in groups[0][1]) == 3

    def test_a_spec_wider_than_a_core_refuses_by_name_without_declared_coalescing(self):
        """[integration of W03 with pack's W04 class 2] The layout packer fuses cores only under declared coalescing,
        as the hard-core packer does, so without it a 48-axon spec on 16-axon cores packs nowhere: a refusal by name
        (the capacity refusal both planes name), never a fused instance no build can place."""
        with pytest.raises(ResidencyPlanError, match="CoreCapacityRefusal") as caught:
            tile_groups([_spec(0, 0, axons=48)], CORE, allow_coalescing=False, allow_splitting=False)
        assert isinstance(caught.value, ToolchainRefusal)

    def test_the_groups_partition_the_layer_packed_alone(self):
        specs = [_spec(i, 0, axons=8, neurons=4 if i % 3 else 8) for i in range(10)]
        groups = tile_groups(specs, CORE, allow_coalescing=False, allow_splitting=False)
        assert sorted(i for members, _ in groups for i in members) == list(range(10))
        assert [members[0] for members, _ in groups] == sorted(members[0] for members, _ in groups)
        assert sum(t for _, cores in groups for _, t in cores) == unit_tile_count(
            specs, CORE, allow_coalescing=False, allow_splitting=False)


class TestTileUnits:
    def test_each_layer_splits_into_its_tile_groups_in_program_order(self):
        specs, nodes, seg = _layered([2, 1, 3])
        units = _units(specs, nodes, seg)
        assert units.members == tuple((i,) for i in range(6))
        assert units.layers == (0, 0, 1, 2, 2, 2) and units.tiles == (1,) * 6 and units.forced == ()

    def test_the_live_set_at_every_unit_cut_is_the_cut_width(self):
        specs, nodes, seg = _layered([2, 1, 3])
        units = _units(specs, nodes, seg)
        # after spec 0: its output is read by spec 2; after spec 1: both; then spec 2 is read until spec 5
        assert units.live == (8, 16, 8, 8, 8, 0)
        cut = PassCut.over(nodes, [list(m) for m in units.members])
        assert units.live[:-1] == tuple(cut.cut_width(b) for b in range(len(units.members) - 1))

    def test_per_layer_the_unit_tiles_are_the_layer_packed_alone(self):
        specs, nodes, seg = _layered([5, 3], axons=8, neurons=4)
        units = _units(specs, nodes, seg)
        for layer in (0, 1):
            members = [i for u, m in enumerate(units.members) if units.layers[u] == layer for i in m]
            tiles = sum(t for u, t in enumerate(units.tiles) if units.layers[u] == layer)
            assert tiles == unit_tile_count([specs[i] for i in members], CORE, allow_coalescing=False,
                                            allow_splitting=False)
        assert len(units.members) > 2                                     # finer than one unit per layer

    def test_an_intra_layer_dependency_keeps_producer_and_consumer_in_one_unit(self):
        # spec 1 has no layer: it joins layer 1, which it feeds (spec 2 reads it)
        specs = [_spec(0, 0), _spec(1, None, latency=1), _spec(2, 1, latency=2)]
        nodes = [CutNode(0, 0, (), 8), CutNode(1, 1, (0,), 8), CutNode(2, 2, (1,), 8)]
        assert _units(specs, nodes, [0, 0, 0]).members == ((0,), (1, 2))

    def test_a_declared_bond_keeps_its_specs_in_one_unit(self):
        specs, nodes, seg = _layered([3, 1])
        units = _units(specs, nodes, seg, bonds=((0, 2),))
        assert units.members == ((0, 2), (1,), (3,)) and units.tiles == (2, 1, 1)

    def test_the_floor_plans_where_a_layer_exceeds_the_budget(self):
        specs, nodes, seg = _layered([3, 1])
        units = _units(specs, nodes, seg)
        assert units.tiles == (1, 1, 1, 1) and units.layers == (0, 0, 0, 1)   # a 3-tile layer in 1-tile units
        plan = residency_steps(units.tiles, units.live, 2, forced=units.forced)
        assert plan.steps and plan.F == 4 and plan.steps[0][0] < 2

    def test_a_tile_group_above_the_budget_refuses_by_name(self):
        specs = [_spec(0, 0, axons=48), _spec(1, 1)]
        nodes = [CutNode(0, 0, (), 8), CutNode(1, 1, (0,), 8)]
        units = _units(specs, nodes, [0, 0], allow_coalescing=True)  # the wide spec fuses under declared coalescing
        assert units.tiles == (3, 1)
        with pytest.raises(ResidencyPlanError, match="tile group") as caught:
            residency_steps(units.tiles, units.live, 2, forced=units.forced)
        assert isinstance(caught.value, ToolchainRefusal)


def _random_fit(rng: random.Random):
    n = rng.randint(1, 12)
    tiles = [rng.randint(1, 6) for _ in range(n)]
    forced = sorted(rng.sample(range(1, n), rng.randint(0, n - 1))) if n > 1 else []
    live = [rng.randint(0, 9) for _ in range(n - 1)] + [0]
    return tiles, live, sum(tiles) + rng.randint(0, 4), forced


class TestTheFit:
    """At F <= C the step function is one step in closed form: r = 0, one phase per segment, the carry of the forced
    cuts; it equals what both fronts answer at r = 0 and at r = C, on seeded random programs."""

    @pytest.mark.parametrize("seed", range(40))
    def test_the_fit_is_one_resident_step_equal_to_both_fronts(self, seed):
        tiles, live, C, forced = _random_fit(random.Random(seed))
        plan = plan_steps(tiles, live, C, forced=forced)
        qf = q_fit(tiles, C, forced)
        carry = sum(2 * live[p - 1] for p in set(forced))
        assert plan.steps == ((0, carry, qf),) and plan.stateless == (carry, qf) and plan.q_fit == qf == len(
            set(forced)) + 1
        for front in FRONTS.values():
            assert front(tiles, live, C, 0, qf + Q_SPAN, forced) == (carry, qf)
            assert front(tiles, live, C, C, qf + Q_SPAN, forced) == (carry, qf)
