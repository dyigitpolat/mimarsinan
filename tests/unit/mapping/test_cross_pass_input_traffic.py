"""[E5] A carried wire re-enters the chip, and that entry is traffic.

The estimator skipped a producer/consumer pair whose two ends landed in
different passes, correctly: that wire does not ride the mesh — it leaves to
the host at the producer's pass and comes back at the consumer's. But it was
skipped and then counted NOWHERE, so a scheduled program's re-injected carry
was invisible to every traffic axis. The more passes the schedule cut, the
more traffic went missing.

Owner decision: cross-pass wires ARE input traffic of the consuming pass. That
is the same class the estimator already models for network inputs — cells that
enter at their consumer's core rather than crossing the mesh — so they land in
the input path, at their consumer's core, with no hops.
"""

from __future__ import annotations

import pytest

from mimarsinan.chip_simulation.sanafe.noc_estimate import estimate_noc
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.noc.fragments import LayoutNocFragments
from mimarsinan.mapping.noc.wire_census import LayoutWireCensus

from unit.mapping.bank_clustered_vehicles import (
    hard_core_types,
    softcores_of,
    two_layer_dependency_graph,
)

#: Two cores of 8 neurons: layer 0's three tokens fill pass 0, layer 1's fill
#: pass 1, and each layer-1 token consumes its own layer-0 token across the cut.
NARROW_CHIP = [{"max_axons": 16, "max_neurons": 8, "count": 2}]
ACTIVITY, STEPS, WIRES = 0.5, 10, 4


def _fragments(pair_wires):
    graph = two_layer_dependency_graph(3, 4)
    softcores = softcores_of(graph)
    planned = collect_noc_fragments(
        softcores=softcores, core_types=hard_core_types(NARROW_CHIP),
        census=None, allow_scheduling=True, allow_neuron_splitting=False,
        allow_coalescing=False, max_schedule_passes=8,
    )
    census = LayoutWireCensus(
        pair_wires=dict(pair_wires),
        input_wires=tuple([0] * len(softcores)),
        on_wires=tuple([0] * len(softcores)),
    )
    return LayoutNocFragments(
        pass_placements=planned.pass_placements, census=census,
        pass_programs=planned.pass_programs,
    )


def _estimate(pair_wires):
    return estimate_noc(
        fragments=_fragments(pair_wires), cores_per_tile=2, mesh_height=2,
        activity_factor=ACTIVITY, timesteps=STEPS,
    )


class TestTheProgramReallyCutsTheWire:
    def test_the_two_layers_land_in_different_passes(self):
        """Without this the pins below would pass on a program that never
        carried anything."""
        fragments = _fragments({})
        passes = [sorted({softcore for softcore, _ in placements})
                  for placements in fragments.pass_placements]
        assert passes == [[0, 1, 2], [3, 4, 5]]


class TestACarriedWireIsCountedOnce:
    def test_it_enters_at_its_consumer_instead_of_vanishing(self):
        """Softcore 0 (pass 0) feeds softcore 3 (pass 1): host-carried, so it
        is the consumer's input traffic — not zero, and not a mesh pair."""
        estimate = _estimate({(0, 3): WIRES})
        expected = WIRES * ACTIVITY * STEPS
        assert estimate.input_path_packets == pytest.approx(expected)
        assert estimate.total_packets == pytest.approx(expected)

    def test_it_costs_no_hops(self):
        """It arrives over the host boundary at the consumer's own core, so
        there is no mesh distance to charge — charging one would double-count
        against the carry the DMA terms already price."""
        assert _estimate({(0, 3): WIRES}).total_hops == 0.0

    def test_it_is_charged_in_the_consuming_pass_only(self):
        """Once, not once per pass: the producing pass sends it to the host,
        which is the readout direction, priced as carry rather than mesh."""
        one_wire = _estimate({(0, 3): WIRES}).total_packets
        assert one_wire == pytest.approx(WIRES * ACTIVITY * STEPS)


class TestSamePassTrafficIsUnchanged:
    def test_a_wire_inside_one_pass_still_rides_the_mesh(self):
        """Softcores 0 and 1 share pass 0 — a real mesh pair, and the packets
        must stay pair traffic rather than moving to the input path."""
        estimate = _estimate({(0, 1): WIRES})
        assert estimate.input_path_packets == 0.0
        assert estimate.total_packets == pytest.approx(WIRES * ACTIVITY * STEPS)

    def test_a_pair_in_neither_pass_is_still_nothing(self):
        """A census entry naming softcores this program never placed cannot
        become traffic just because its producer is missing."""
        assert _estimate({(97, 98): WIRES}).total_packets == 0.0
