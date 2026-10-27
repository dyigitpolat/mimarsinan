"""[F2] The residency units in a program order: each layer's tile groups, the layers of a segment ranked by a
topological order of the layer graph (ties to the layer's first spec index, the layer order), the units in a
topological order of the unit graph (ties to the segment, the layer's rank and the layer-order position), and a
strongly connected set of units one unit, which one pass holds.

A lane program's fused add (WI-5e class E) makes the accumulator read its operand's producer, which the IR emits
after it, so layer order put a unit before one it reads and every step refused ("layer order is not a program
order"). Wherever layer order is a program order (every program that planned before), the units are the same tuples
in the same order.
"""

from __future__ import annotations

import random

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.packing.residency_tiles import tile_groups
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.mapping.support.schedule.pass_cut import CutNode, PassCut
from mimarsinan.mapping.support.schedule.residency_units import residency_steps, residency_units

CORE = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=4)]


def _spec(i, layer, segment=0, axons=16, neurons=8):
    return LayoutSoftCoreSpec(input_count=axons, output_count=neurons, perceptron_index=layer, segment_id=segment,
                              latency_tag=i, name=f"sc{i}")


def _program(layers, edges, shapes=None, segments=None):
    """One spec per entry of ``layers``; ``edges`` (producer, consumer) spec pairs, within a segment."""
    shapes = shapes or [(16, 8)] * len(layers)
    segments = segments or [0] * len(layers)
    specs = [_spec(i, layer, segments[i], *shapes[i]) for i, layer in enumerate(layers)]
    nodes = [CutNode(core_id=i, latency=i, sources=tuple(sorted(a for a, b in edges if b == i)),
                     out_width=shapes[i][1]) for i in range(len(layers))]
    return specs, nodes, segments


def _monotone(units, nodes):
    """Each segment's units are a program order (``PassCut.over`` refuses a unit that reads a later one)."""
    by_id = {n.core_id: n for n in nodes}
    for segment in sorted(set(units.segments)):
        mine = [m for m, s in zip(units.members, units.segments) if s == segment]
        PassCut.over([by_id[i] for m in mine for i in m], [list(m) for m in mine])


def _layer_order_units(specs, nodes, segments):
    """The units in layer order, as the planner formed them before F2: each layer's tile groups, the layers by
    (segment, first spec index)."""
    order = {}
    for index, spec in enumerate(specs):
        order.setdefault((segments[index], spec.perceptron_index), []).append(index)
    tied = [(int(n.core_id), int(s)) for n in nodes for s in n.sources]
    out = []
    for key in sorted(order, key=lambda k: (k[0], order[k][0])):
        local = {spec: i for i, spec in enumerate(order[key])}
        pairs = [(local[a], local[b]) for a, b in tied if a in local and b in local]
        for members, cores in tile_groups([specs[i] for i in order[key]], CORE, allow_coalescing=False,
                                          allow_splitting=False, bonds=pairs):
            out.append((tuple(order[key][m] for m in members), sum(t for _, t in cores)))
    return out


class TestUnitOrder:
    def test_a_fused_add_whose_accumulator_layer_comes_first_plans_in_dependency_order(self):
        # the lane residual shape: the accumulator's layer (2) holds the lower spec index and reads layer 1's output
        # through its operand rows; layer order put it before its producer
        specs, nodes, seg = _program([0, 2, 1], edges=[(0, 2), (0, 1), (2, 1)])
        units = residency_units(specs, nodes, CORE, segments=seg)
        assert units.members == ((0,), (2,), (1,)) and units.layers == (0, 1, 2)
        _monotone(units, nodes)

    def test_an_acyclic_layer_graph_keeps_each_layer_whole_in_its_topological_order(self):
        # layer 2 (specs 1 and 2, one tile group each) precedes layer 1 (spec 3) in layer order; spec 2 reads layer 1,
        # spec 1 only layer 0: the layer graph's order is 0, 1, 2 and layer 2's tile groups stay together after layer
        # 1 (a unit-by-unit order would run spec 1 before layer 1)
        specs, nodes, seg = _program([0, 2, 2, 1], edges=[(0, 1), (0, 3), (3, 2)])
        units = residency_units(specs, nodes, CORE, segments=seg)
        assert units.members == ((0,), (3,), (1,), (2,)) and units.layers == (0, 1, 2, 2)
        _monotone(units, nodes)

    def test_a_layer_cycle_over_acyclic_tile_groups_plans(self):
        # layer 0 holds specs 0 and 2 (one core each), spec 2 reads layer 1, which reads spec 0: the layer graph is
        # cyclic, the unit graph is not, so the units run 0, 1, 2 (a pass may hold part of a layer)
        specs, nodes, seg = _program([0, 1, 0], edges=[(0, 1), (1, 2)])
        units = residency_units(specs, nodes, CORE, segments=seg)
        assert units.members == ((0,), (1,), (2,)) and units.layers == (0, 1, 0)
        _monotone(units, nodes)

    def test_a_cycle_of_tile_groups_is_one_unit(self):
        # spec 1 has no layer (a pool-lane core): it joins the layer of its lowest-index consumer, spec 2 (layer 2),
        # and is tied to it (spec 2 reads it). Spec 3 (layer 1) reads spec 1 and spec 2 reads spec 3: the unit {1, 2}
        # and the unit {3} read each other, so no order of them is a program order; they are one unit (one pass)
        specs, nodes, seg = _program([0, None, 2, 1], edges=[(0, 1), (1, 2), (1, 3), (3, 2)])
        units = residency_units(specs, nodes, CORE, segments=seg)
        assert units.members == ((0,), (1, 2, 3)) and units.tiles == (1, 3) and units.layers == (0, 2)
        assert units.cores == (((128, 1),), ((128, 1), (128, 1), (128, 1)))
        _monotone(units, nodes)

    def test_a_merged_unit_above_the_budget_refuses_by_name(self):
        specs, nodes, seg = _program([0, None, 2, 1], edges=[(0, 1), (1, 2), (1, 3), (3, 2)])
        units = residency_units(specs, nodes, CORE, segments=seg)
        with pytest.raises(ResidencyPlanError, match="exceed the core budget") as caught:
            residency_steps(units.tiles, units.live, 2)
        assert isinstance(caught.value, ToolchainRefusal) and "units [1]" in str(caught.value)

    def test_live_sets_are_the_cut_widths_of_the_new_order(self):
        specs, nodes, seg = _program([0, 2, 1], edges=[(0, 2), (0, 1), (2, 1)])
        units = residency_units(specs, nodes, CORE, segments=seg)
        cut = PassCut.over(nodes, [list(m) for m in units.members])
        assert units.live[:-1] == tuple(cut.cut_width(b) for b in range(len(units.members) - 1))
        assert units.spans == tuple((w.width, w.produced_in, w.last_consumed_in) for w in cut.carried)

    def test_each_segment_is_ordered_on_its_own_and_its_cut_stays_forced(self):
        # segment 0 (specs 0 to 2) in layer order; segment 1 holds the fused-add shape: spec 3 (layer 4, first in
        # layer order) reads spec 4 (layer 3); a read across the segment cut is a host boundary, no cut-node source
        specs, nodes, seg = _program([0, 1, 2, 4, 3], edges=[(0, 1), (1, 2), (4, 3)], segments=[0, 0, 0, 1, 1])
        units = residency_units(specs, nodes, CORE, segments=seg)
        assert units.members == ((0,), (1,), (2,), (4,), (3,)) and units.forced == (3,)
        assert units.segments == (0, 0, 0, 1, 1) and units.layers == (0, 1, 2, 3, 4)
        _monotone(units, nodes)


class TestLayerOrderKept:
    """Order identity: wherever layer order is a program order the units are the same tuples in the same order."""

    @pytest.mark.parametrize("seed", range(40))
    def test_a_program_whose_layer_order_is_a_program_order_keeps_its_units(self, seed):
        rng = random.Random(seed)
        n = rng.randint(3, 14)
        layers = sorted(rng.randint(0, max(1, n // 2)) for _ in range(n))       # layers in spec order
        edges = [(a, b) for b in range(n) for a in range(b) if layers[a] < layers[b] and rng.random() < 0.3]
        edges += [(a, b) for b in range(n) for a in range(b) if layers[a] == layers[b] and rng.random() < 0.1]
        shapes = [rng.choice([(16, 8), (8, 4), (8, 8)]) for _ in range(n)]
        specs, nodes, seg = _program(layers, edges, shapes)
        units = residency_units(specs, nodes, CORE, segments=seg)
        assert [(m, t) for m, t in zip(units.members, units.tiles)] == _layer_order_units(specs, nodes, seg)
        _monotone(units, nodes)

    @pytest.mark.parametrize("seed", range(40))
    def test_any_acyclic_spec_graph_plans_a_program_order(self, seed):
        rng = random.Random(1000 + seed)
        n = rng.randint(3, 14)
        hidden = list(range(n))
        rng.shuffle(hidden)                                                     # a hidden topological order
        rank = {s: k for k, s in enumerate(hidden)}
        layers = [rng.randint(0, 4) for _ in range(n)]
        edges = [(a, b) for a in range(n) for b in range(n) if rank[a] < rank[b] and rng.random() < 0.25]
        specs, nodes, seg = _program(layers, edges)                             # 16x8 specs: one core each
        units = residency_units(specs, nodes, CORE, segments=seg)
        _monotone(units, nodes)
        assert sorted(i for m in units.members for i in m) == list(range(n))
        assert sum(units.tiles) == n
