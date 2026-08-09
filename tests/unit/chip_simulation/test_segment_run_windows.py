"""[nevresim parity] Window-count assembly: the runner's segment output is the
record build's [lat, lat+T) currency, re-read through the chip output wiring."""

import numpy as np

from mimarsinan.chip_simulation.simulation_runner.emit import _PreparedSegment
from mimarsinan.chip_simulation.simulation_runner.segment_run import (
    window_counts_from_records,
)

T = 4


def _prepared(output_sources):
    return _PreparedSegment(
        seg_idx=0, seg_dir="/dev/null", binary_path="/dev/null",
        output_size=len(output_sources), input_size=3,
        record_mode=True, output_sources=output_sources,
    )


class TestWindowAssembly:
    def test_core_input_on_off_kinds(self):
        prepared = _prepared([
            ("core", 0, 1),
            ("input", 0, 2),
            ("on", 0, 0),
            ("off", 0, 0),
        ])
        records = [{0: {"in": [0, 0, 0], "out": [7, 3, 1]}}]
        x = np.array([0.0, 0.0, 0.5])
        out = window_counts_from_records(prepared, records, [(x, 0)], T)
        assert out.tolist() == [[3.0, 2.0, 4.0, 0.0]]

    def test_input_passthrough_mirrors_comb_clamps(self):
        prepared = _prepared([("input", 0, 0), ("input", 0, 1), ("input", 0, 2)])
        records = [{0: {"in": [], "out": []}}]
        x = np.array([1.7, -0.3, 1.0])
        out = window_counts_from_records(prepared, records, [(x, 0)], T)
        assert out.tolist() == [[float(T), 0.0, float(T)]]

    def test_per_sample_rows(self):
        prepared = _prepared([("core", 1, 0)])
        records = [
            {0: {"in": [], "out": []}, 1: {"in": [], "out": [2]}},
            {0: {"in": [], "out": []}, 1: {"in": [], "out": [4]}},
        ]
        xs = [(np.zeros(3), 0), (np.zeros(3), 0)]
        out = window_counts_from_records(prepared, records, xs, T)
        assert out.tolist() == [[2.0], [4.0]]
