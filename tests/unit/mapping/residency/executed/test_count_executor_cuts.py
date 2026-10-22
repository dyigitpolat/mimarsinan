"""The synchronized count executor (``lif_execution_synchronized``, the X3a screen's ``synchronized_lif``) across U1's
cuts: it reads each input slice's window count, so a pass's delivery delay (R2, a shift in time) does not apply to its
input, and it walks a pass's cores in dependency order (R6), because the pins (R1) can give a producer the latency of
the consumer that reads it."""

from __future__ import annotations

from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.pass_timing import replay_delays
from mimarsinan.models.spiking.hybrid.executors import runs_synchronized_counts

from . import runs as R
from . import vehicles as V


def _latency_order_breaks_a_read(stage) -> bool:
    """A core that reads a core of its own pass at the same stored latency and sits before it: the latency order
    would walk the reader first."""
    hcm = stage.hard_core_mapping
    ChipLatency(hcm).calculate()
    latency = [int(core.latency or 0) for core in hcm.cores]
    return any(int(span.src_core) > reader and latency[int(span.src_core)] == latency[reader]
               for reader, core in enumerate(hcm.cores) for span in core.get_axon_source_spans()
               if span.kind == "core")


def test_a_delayed_cut_runs_equal_to_the_identity_run_under_the_count_executor():
    ir = V.ir_of(V.branchy_repr("skew3"))
    u1 = V.build(ir, 2, {"step": "floor"})
    assert any(replay_delays(stage) for stage in V.neural_passes(u1))
    R.assert_runs_equal_to_the_identity_run(ir, 2, R.SYNCHRONIZED_LIF, mapping=u1)


def test_a_pinned_pass_whose_latency_order_is_no_dependency_order_runs_equal_under_the_count_executor():
    ir = V.ir_of(V.branchy_repr("late_skip"))
    u1 = V.build(ir, 3, None)
    assert any(_latency_order_breaks_a_read(stage) for stage in V.neural_passes(u1))
    R.assert_runs_equal_to_the_identity_run(ir, 3, R.SYNCHRONIZED_LIF, mapping=u1)


def test_the_count_executor_runs_a_synchronized_flow_unless_a_spike_record_is_asked_for():
    ir = V.ir_of(V.branchy_repr("skew3"))
    synchronized = R.flow(V.build(ir, 2, None), **R.SYNCHRONIZED_LIF)
    streamed = R.flow(V.build(ir, 2, None), **R.STREAMED_LIF)
    assert runs_synchronized_counts(synchronized, recording=False)
    assert not runs_synchronized_counts(synchronized, recording=True)
    assert not runs_synchronized_counts(streamed, recording=False)
