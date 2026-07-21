"""[calculus §17/PR45] typed certificates over backend RunRecords (Edge B)."""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.certification.record_certificates import certify_run_records
from mimarsinan.chip_simulation.recording.records import (
    CoreSpikeCounts,
    RunRecord,
    SegmentSpikeRecord,
)


def _record(out0=(3, 0, 7), core_out=(3, 0, 7)) -> RunRecord:
    seg = SegmentSpikeRecord(
        stage_index=0, stage_name="neural_segment_final",
        schedule_segment_index=None, schedule_pass_index=None,
        seg_input_rates=np.zeros((1, 2)),
        seg_input_spike_count=np.array([2, 1]),
        seg_output_spike_count=np.asarray(out0),
        cores=[CoreSpikeCounts(
            core_index=0, n_in_used=2, n_out_used=3, core_latency=0,
            has_hardware_bias=False, n_always_on_axons=0,
            input_spike_count=np.array([2, 1]),
            output_spike_count=np.asarray(core_out),
        )],
    )
    return RunRecord(sample_index=0, T=4, segments={0: seg})


def test_identical_records_certify_exact():
    cert = certify_run_records(_record(), _record(), backend="sanafe")
    assert cert.passed and cert.exact_match_fraction == 1.0
    assert cert.neuron_windows_compared > 0


def test_one_count_flip_fails_exact_class_but_passes_counts_export():
    ref = _record()
    off = _record(out0=(3, 1, 7), core_out=(3, 1, 7))
    exact = certify_run_records(ref, off, backend="sanafe")
    assert not exact.passed and exact.max_abs_delta == 1.0
    export = certify_run_records(ref, off, backend="loihi")
    assert export.passed  # documented ±1 tolerance for counts-export


def test_missing_segment_fails_loud():
    ref = _record()
    actual = RunRecord(sample_index=0, T=4, segments={})
    with pytest.raises(KeyError):
        certify_run_records(ref, actual, backend="sanafe")
