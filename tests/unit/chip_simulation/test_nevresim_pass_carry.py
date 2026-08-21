"""nevresim's verbatim pass carry: gather trains out, assemble trains in."""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.chip_simulation.simulation_runner.emit import _PreparedSegment
from mimarsinan.chip_simulation.simulation_runner.segment_run import (
    raster_from_spike_trains,
    window_counts_from_records,
)


class TestTheRasterGather:
    """The train records are already producer-local, so the gather is a row
    selection with NO time shift — a shift here would double-apply the latency
    the recorder already removed."""

    def test_core_sources_take_the_neurons_train(self):
        raster = raster_from_spike_trains(
            {0: [[0, 1, 0, 1], [0, 0, 1, 1]]},
            [("core", 0, 0), ("core", 0, 1)], 4, None,
        )
        assert raster[:, 0].tolist() == [0, 1, 0, 1]
        assert raster[:, 1].tolist() == [0, 0, 1, 1]

    def test_a_counted_train_carries_its_multiplicities_into_the_raster(self):
        """[ODIN P3] the carry seam's whole point: a per-event producer emits
        3 spikes in one cycle and the raster the next segment replays says 3,
        not 1."""
        raster = raster_from_spike_trains(
            {0: [[0, 3, 1, 0]]}, [("core", 0, 0)], 4, None,
        )
        assert raster[:, 0].tolist() == [0, 3, 1, 0]

    def test_always_on_sources_spike_every_cycle(self):
        raster = raster_from_spike_trains({}, [("on", 0, 0)], 3, None)
        assert raster[:, 0].tolist() == [1, 1, 1]

    def test_input_passthroughs_replay_the_segment_input_train(self):
        train = np.array([[1, 0], [0, 0], [0, 1]], dtype=np.float64)
        raster = raster_from_spike_trains(
            {}, [("input", 0, 1)], 3, train,
        )
        assert raster[:, 0].tolist() == [0, 0, 1]

    def test_an_input_passthrough_without_the_train_fails_loud(self):
        with pytest.raises(ValueError, match="input train"):
            raster_from_spike_trains({}, [("input", 0, 0)], 3, None)

    def test_a_missing_core_yields_silence_not_a_crash(self):
        """A core the trace never mentioned emitted nothing — zeros are the
        truth, and the window-count certificates would catch anything else."""
        raster = raster_from_spike_trains({}, [("core", 7, 0)], 3, None)
        assert raster[:, 0].tolist() == [0, 0, 0]


class TestSpikeTrainWindowCounts:
    """A SpikeTrain-mode segment's input IS a train: an input-kind output must
    count its spikes, not comb-encode the flattened train as if it were a
    value — which would read garbage."""

    def _prepared(self, mode):
        return _PreparedSegment(
            seg_idx=0, seg_dir=".", binary_path="x", output_size=1,
            input_size=2, record_mode=True,
            output_sources=[("input", 0, 1)], input_mode=mode,
        )

    def test_spike_train_input_counts_the_train(self):
        # (T=3, size=2) cycle-major: neuron 1 spikes at cycles 0 and 2.
        flat = np.array([0, 1, 0, 0, 0, 1], dtype=np.float64)
        counts = window_counts_from_records(
            self._prepared("SpikeTrain"), [{}], [(flat, np.zeros(1))], 3,
        )
        assert counts[0, 0] == 2.0

    def test_value_input_keeps_the_comb_count(self):
        counts = window_counts_from_records(
            self._prepared(None), [{}], [(np.array([0.0, 1.0]), np.zeros(1))], 3,
        )
        assert counts[0, 0] == 3.0
