"""[H1] The measured synaptic-event census: arrivals ⊗ occupied columns.

``total_spikes`` counts EMISSIONS; energy is priced per synapse ARRIVAL — one
spike landing on one occupied cell of a consumer row. The record sealed no
arrival census, so ``energy_per_inference_mj`` refused on the record plane and
the candidate's modeled energy (``onchip_macs x timesteps x activity``) was
never checked against a deployment.

The census is a pure join of facts the runner already holds: the HCM's axon
source spans (which producer neurons / input wires feed which consumer rows),
per-core measured LIF emissions, and per-core measured boundary arrivals
(input + always-on trace groups). Every quantity is measured or structural —
nothing is assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.chip_simulation.synaptic_events import synaptic_event_census
from mimarsinan.mapping.support.spike_source_spans import SpikeSourceSpan


class _Core:
    """A packed hard core, duck-typed: spans + used geometry."""

    def __init__(self, spans, neurons_used, axons_used=4):
        self._spans = list(spans)
        self.axons_per_core = 8
        self.neurons_per_core = 8
        self.available_axons = self.axons_per_core - axons_used
        self.available_neurons = self.neurons_per_core - neurons_used
        self.unusable_space = 0

    def get_axon_source_spans(self):
        return self._spans


def _span(kind, src_core=-1, src_start=0, length=1, dst_start=0):
    return SpikeSourceSpan(kind=kind, src_core=src_core, src_start=src_start,
                      length=length, dst_start=dst_start)


class TestTheJoin:
    def test_inter_core_arrivals_multiply_the_consumers_columns(self):
        """Core 1 reads core 0's neurons [0, 2): emissions 3 and 5 land on a
        consumer with 4 used columns — (3 + 5) x 4 events, nothing else."""
        cores = [
            _Core([_span("input", src_start=0, length=2)], neurons_used=2),
            _Core([_span("core", src_core=0, src_start=0, length=2)],
                  neurons_used=4),
        ]
        events = synaptic_event_census(
            cores,
            emissions_of=lambda c: np.array([3, 5]) if c == 0 else None,
            boundary_arrivals_of=lambda c: 0,
        )
        assert events == (3 + 5) * 4

    def test_boundary_arrivals_multiply_their_own_cores_columns(self):
        """Input/always-on arrivals are already tallied PER CONSUMER by the
        trace groups — the census only weights them by that core's columns."""
        cores = [_Core([_span("input", length=3)], neurons_used=5)]
        events = synaptic_event_census(
            cores,
            emissions_of=lambda c: np.zeros(0),
            boundary_arrivals_of=lambda c: 7,
        )
        assert events == 7 * 5

    def test_a_producer_slice_counts_only_the_read_neurons(self):
        """The consumer reads neurons [1, 3) of a 3-neuron producer: the
        unread neuron's emissions are not its arrivals."""
        cores = [
            _Core([_span("input", length=1)], neurons_used=3),
            _Core([_span("core", src_core=0, src_start=1, length=2)],
                  neurons_used=2),
        ]
        events = synaptic_event_census(
            cores,
            emissions_of=lambda c: np.array([100, 4, 6]) if c == 0 else None,
            boundary_arrivals_of=lambda c: 0,
        )
        assert events == (4 + 6) * 2

    def test_off_rows_carry_nothing(self):
        cores = [_Core([_span("off", length=4)], neurons_used=4)]
        events = synaptic_event_census(
            cores, emissions_of=lambda c: np.zeros(0),
            boundary_arrivals_of=lambda c: 0,
        )
        assert events == 0

    def test_an_unused_core_contributes_no_arrivals(self):
        """Zero used columns means the arrivals land on nothing occupied."""
        cores = [_Core([_span("input", length=2)], neurons_used=0)]
        events = synaptic_event_census(
            cores, emissions_of=lambda c: np.zeros(0),
            boundary_arrivals_of=lambda c: 9,
        )
        assert events == 0


class TestHonestAbsence:
    def test_a_missing_emission_trace_refuses_rather_than_undercounts(self):
        """A consumer reads a producer whose trace was not parsed: the census
        cannot answer, and None is the answer — a partial count priced as a
        full one is exactly the silent-zero this program exists to kill."""
        cores = [
            _Core([_span("input", length=1)], neurons_used=2),
            _Core([_span("core", src_core=0, src_start=0, length=2)],
                  neurons_used=2),
        ]
        events = synaptic_event_census(
            cores, emissions_of=lambda c: None,
            boundary_arrivals_of=lambda c: 0,
        )
        assert events is None

    def test_a_short_emission_array_is_a_defect_not_a_clip(self):
        """The span reads past the producer's array: index books are broken,
        so fail loud instead of counting what happens to be there."""
        cores = [
            _Core([_span("core", src_core=0, src_start=0, length=3)],
                  neurons_used=1),
        ]
        with pytest.raises(ValueError, match="span"):
            synaptic_event_census(
                cores, emissions_of=lambda c: np.array([1, 2]),
                boundary_arrivals_of=lambda c: 0,
            )
