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


class TestSpikeTrainParse:
    """SPKTRN is its own record line beside SPKREC — the count parser is untouched,
    and the trains arrive in producer-local time as per-neuron bitstrings."""

    def test_trains_parse_per_sample_per_core(self):
        from mimarsinan.chip_simulation.nevresim.execute_nevresim import (
            parse_spike_trains,
        )

        stderr = (
            "SPKTRN 0 0101 0011\nSPKTRN 1 1111\nSPKTRN_END\n"
            "SPKTRN 0 0000 1000\nSPKTRN 1 0001\nSPKTRN_END\n"
        )
        samples = parse_spike_trains(stderr)
        assert samples == [
            {0: ["0101", "0011"], 1: ["1111"]},
            {0: ["0000", "1000"], 1: ["0001"]},
        ]

    def test_spkrec_lines_are_not_trains_and_trains_are_not_counts(self):
        from mimarsinan.chip_simulation.nevresim.execute_nevresim import (
            parse_spike_records,
            parse_spike_trains,
        )

        stderr = (
            "SPKREC 0 IN 1 OUT 2\nSPKREC_END\n"
            "SPKTRN 0 0101\nSPKTRN_END\n"
        )
        assert parse_spike_records(stderr) == [{0: {"in": [1], "out": [2]}}]
        assert parse_spike_trains(stderr) == [{0: ["0101"]}]

    def test_a_ragged_core_fails_loud(self):
        import pytest

        from mimarsinan.chip_simulation.nevresim.execute_nevresim import (
            parse_spike_trains,
        )

        with pytest.raises(ValueError, match="ragged"):
            parse_spike_trains("SPKTRN 0 010 0111\nSPKTRN_END\n")

    def test_a_non_binary_symbol_fails_loud(self):
        import pytest

        from mimarsinan.chip_simulation.nevresim.execute_nevresim import (
            parse_spike_trains,
        )

        with pytest.raises(ValueError, match="non-binary"):
            parse_spike_trains("SPKTRN 0 0102\nSPKTRN_END\n")

    def test_the_recorder_header_pins_the_protocol(self):
        """The C++ side is compiled only in the integration tier, so the unit
        tier pins the contract textually: its own define, its own record line,
        and the producer-local time convention."""
        from pathlib import Path

        header = Path("nevresim/include/simulator/recording/"
                      "spike_train_recorder.hpp").read_text()
        assert "NEVRESIM_RECORD_SPIKE_TRAINS" in header
        assert '"SPKTRN "' in header and '"SPKTRN_END' in header
        assert "cycle - lat" in header, "producer-local time is the convention"
        execution = Path("nevresim/include/simulator/execution/"
                         "spiking_execution.hpp").read_text()
        assert execution.count("NEVRESIM_RECORD_SPIKE_TRAINS") >= 4
