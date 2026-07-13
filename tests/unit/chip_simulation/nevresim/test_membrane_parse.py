"""Parsing of ``MEMB`` stderr lines from a ``NEVRESIM_EXPORT_MEMBRANE`` build."""

from __future__ import annotations

import pytest

from mimarsinan.chip_simulation.nevresim.execute_nevresim import (
    parse_membrane_records,
    parse_spike_records,
)


class TestParseMembraneRecords:
    def test_one_line_per_sample(self):
        stderr = "MEMB 0.5 -0.25 1\nMEMB 0 0 0.125\n"
        assert parse_membrane_records(stderr) == [
            [0.5, -0.25, 1.0],
            [0.0, 0.0, 0.125],
        ]

    def test_ignores_non_memb_lines(self):
        stderr = (
            "SPKREC 0 IN 1 OUT 2\n"
            "MEMB -0.75 0.5\n"
            "SPKREC_END\n"
            "noise\n"
        )
        assert parse_membrane_records(stderr) == [[-0.75, 0.5]]

    def test_empty_and_none_stderr(self):
        assert parse_membrane_records("") == []
        assert parse_membrane_records(None) == []

    def test_round_trip_precision_survives(self):
        value = 0.7300000000000001
        assert parse_membrane_records(f"MEMB {value!r}") == [[value]]

    def test_memb_lines_do_not_confuse_spike_record_parse(self):
        """The two stderr channels coexist: SPKREC parsing skips MEMB lines."""
        stderr = (
            "MEMB 0.5\n"
            "SPKREC 0 IN 1 OUT 2\n"
            "SPKREC_END\n"
        )
        assert parse_spike_records(stderr) == [{0: {"in": [1], "out": [2]}}]
        assert parse_membrane_records(stderr) == [[0.5]]


class TestRunBinaryRawMembraneContract:
    def test_record_spikes_and_export_membrane_are_exclusive(self):
        from mimarsinan.chip_simulation.nevresim.segment_execute import run_binary_raw

        with pytest.raises(ValueError, match="export_membrane"):
            run_binary_raw(
                binary_path="/nonexistent",
                work_dir="/nonexistent",
                input_loader=[],
                output_size=1,
                simulation_length=4,
                input_size=1,
                spike_generation_mode="Uniform",
                max_input_count=0,
                record_spikes=True,
                export_membrane=True,
            )
