"""format_first_diff: a self-contained first line and a per-layer remainder tally."""

from __future__ import annotations

import numpy as np

from mimarsinan.chip_simulation.recording.compare import (
    compare_records,
    format_first_diff,
)
from mimarsinan.chip_simulation.recording.records import (
    CoreSpikeCounts,
    RunRecord,
    SegmentSpikeRecord,
)


def _core(idx: int, inp: np.ndarray, out: np.ndarray) -> CoreSpikeCounts:
    return CoreSpikeCounts(
        core_index=idx, n_in_used=len(inp), n_out_used=len(out),
        core_latency=0, has_hardware_bias=False, n_always_on_axons=0,
        input_spike_count=inp, output_spike_count=out,
    )


def _record(seg_in: np.ndarray, core_in: np.ndarray, core_out: np.ndarray,
            seg_out: np.ndarray) -> RunRecord:
    seg = SegmentSpikeRecord(
        stage_index=1,
        stage_name="neural_segment_until:classifier_col0",
        schedule_segment_index=0,
        schedule_pass_index=0,
        seg_input_rates=(seg_in / 32.0).reshape(1, -1),
        seg_input_spike_count=seg_in,
        seg_output_spike_count=seg_out,
        cores=[_core(0, core_in, core_out)],
    )
    return RunRecord(sample_index=0, T=32, segments={1: seg})


def _diverged_records() -> tuple[RunRecord, RunRecord]:
    base = dict(
        seg_in=np.array([2, 3, 5, 6], dtype=np.int64),
        core_in=np.array([1, 1], dtype=np.int64),
        core_out=np.array([4, 4], dtype=np.int64),
        seg_out=np.array([9], dtype=np.int64),
    )
    ref = _record(**base)
    actual = _record(
        seg_in=np.array([2, 8, 5, 16], dtype=np.int64),
        core_in=np.array([1, 2], dtype=np.int64),
        core_out=np.array([4, 6], dtype=np.int64),
        seg_out=np.array([11], dtype=np.int64),
    )
    return ref, actual


def test_first_line_is_self_contained_summary() -> None:
    ref, actual = _diverged_records()
    diffs = compare_records(ref, actual)
    assert diffs and diffs[0].layer == "seg_input"

    msg = format_first_diff(diffs)
    first_line = msg.splitlines()[0]
    assert first_line.strip(), "first line must not be empty"
    assert "neural_segment_until:classifier_col0" in first_line
    assert "seg_input" in first_line
    assert "2/4" in first_line
    # both sums appear on line 1 (16 expected vs 31 actual)
    assert "16" in first_line and "31" in first_line
    # AssertionError(msg) renders the summary, not an empty string
    assert str(AssertionError(msg)).splitlines()[0].strip()


def test_remaining_diffs_render_as_per_layer_tally() -> None:
    ref, actual = _diverged_records()
    diffs = compare_records(ref, actual)
    assert len(diffs) == 4  # seg_input, core_input, core_output, seg_output

    msg = format_first_diff(diffs)
    assert "more diffs not shown" not in msg
    tally_lines = [l for l in msg.splitlines() if "remaining" in l]
    assert len(tally_lines) == 1
    tally = tally_lines[0]
    assert "core_input" in tally and "1" in tally
    assert "core_output" in tally
    assert "seg_output" in tally


def test_single_diff_has_no_tally() -> None:
    ref, actual = _diverged_records()
    diffs = compare_records(ref, actual)[:1]
    msg = format_first_diff(diffs)
    assert "remaining" not in msg
    assert "more diffs not shown" not in msg


def test_no_diffs_and_scalar_diffs_are_robust() -> None:
    assert format_first_diff([]) == "no diffs"

    ref, actual = _diverged_records()
    actual.T = 16  # T mismatch produces a 0-d expected/actual pair
    diffs = compare_records(ref, actual)
    assert diffs and diffs[0].layer == "T"
    msg = format_first_diff(diffs)
    assert msg.splitlines()[0].strip()
    assert "32" in msg and "16" in msg
