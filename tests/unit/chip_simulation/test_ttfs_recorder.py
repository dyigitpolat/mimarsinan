import numpy as np

from mimarsinan.chip_simulation.ttfs_recorder import (
    CoreTtfsActivations,
    SegmentTtfsRecord,
    TtfsRunRecord,
    compare_ttfs_records,
)


def test_compare_ttfs_records_detects_mismatch():
    ref = TtfsRunRecord(sample_index=0, simulation_length=4, spiking_mode="ttfs")
    ref.segments[0] = SegmentTtfsRecord(
        stage_index=0, stage_name="s0",
        schedule_segment_index=None, schedule_pass_index=None,
        seg_output=np.zeros(2),
        cores=[CoreTtfsActivations(0, 2, np.array([0.5, 0.25]))],
    )
    act = TtfsRunRecord(sample_index=0, simulation_length=4, spiking_mode="ttfs")
    act.segments[0] = SegmentTtfsRecord(
        stage_index=0, stage_name="s0",
        schedule_segment_index=None, schedule_pass_index=None,
        seg_output=np.zeros(2),
        cores=[CoreTtfsActivations(0, 2, np.array([0.5, 0.3]))],
    )
    diffs = compare_ttfs_records(ref, act)
    assert len(diffs) == 1
    assert diffs[0].neuron_index == 1
