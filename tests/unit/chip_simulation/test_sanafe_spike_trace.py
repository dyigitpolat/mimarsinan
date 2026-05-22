"""Spike-trace parsing and message taxonomy helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mimarsinan.chip_simulation.sanafe.runner import (
    _build_spike_capture_warning,
    _hardcore_index_from_spike_group,
    _lif_and_input_spike_totals,
    _spike_event_group_and_index,
    _spike_trace_to_group_counts,
    _summarize_message_trace,
)


class _FakeNeuronAddress:
    def __init__(self, group_name: str, neuron_offset: int):
        self.group_name = group_name
        self.neuron_offset = neuron_offset

    def __repr__(self) -> str:
        return f"{self.group_name}.{self.neuron_offset}"


def test_spike_event_group_and_index_from_neuron_address():
    ev = _FakeNeuronAddress("core3", 7)
    assert _spike_event_group_and_index(ev) == ("core3", 7)


def test_spike_event_group_and_index_from_string():
    assert _spike_event_group_and_index("core12.4") == ("core12", 4)


def test_hardcore_index_from_spike_group():
    assert _hardcore_index_from_spike_group("core5") == 5
    assert _hardcore_index_from_spike_group("core5_in") == 5
    assert _hardcore_index_from_spike_group("core5_on") == 5
    assert _hardcore_index_from_spike_group("other") is None


def test_spike_trace_counts_neuron_address_objects():
    trace = [[_FakeNeuronAddress("core0_in", 0), _FakeNeuronAddress("core0", 1)]]
    counts, skipped = _spike_trace_to_group_counts(
        trace, group_sizes={"core0": 2, "core0_in": 1},
    )
    assert skipped == 0
    assert counts["core0_in"].tolist() == [1]
    assert counts["core0"].tolist() == [0, 1]
    lif, inp = _lif_and_input_spike_totals(counts)
    assert lif == 1
    assert inp == 1


def test_summarize_message_trace_taxonomy():
    mt = [[
        {
            "src_tile_id": 0, "dest_tile_id": 1,
            "src_neuron_group_id": "core0", "placeholder": False,
        },
        {
            "src_tile_id": 2, "dest_tile_id": 2,
            "src_neuron_group_id": "core23_in", "placeholder": False,
        },
    ]]
    s = _summarize_message_trace(mt)
    assert s["inter_tile_packets"] == 1
    assert s["intra_tile_packets"] == 1
    assert s["input_path_packets"] == 1


def test_spike_capture_warning_input_only():
    msg = _build_spike_capture_warning(
        chip_spike_count=1000,
        lif_spike_count=0,
        input_path_packets=666,
        spike_trace_parse_skipped=0,
    )
    assert msg is not None
    assert "input-path" in msg
