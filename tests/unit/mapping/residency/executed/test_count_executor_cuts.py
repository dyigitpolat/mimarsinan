"""Integrate then emit (``lif_execution_synchronized``, the X3a screen's ``synchronized_lif``: [U-LIF2 W9] its gated
cycle-level executor, and the count executor, its closed form) across U1's cuts: it reads each input slice's window
count, so a pass's delivery delay (R2, a shift in time) does not apply to its input, and it walks a pass's cores in
dependency order (R6), because the pins (R1) can give a producer the latency of the consumer that reads it."""

from __future__ import annotations

from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.pass_timing import replay_delays
from mimarsinan.models.spiking.hybrid.executors import integrates_then_emits, runs_closed_form

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


def test_a_synchronized_flow_reads_counts_and_runs_the_closed_form_only_when_asked_by_name():
    """[U-LIF2 W9] A flow that integrates then emits reads each input slice's window count, recording or not (its
    executor of record is the gated cycle-level executor); the closed form runs only when asked for by its name, never
    for a spike record."""
    ir = V.ir_of(V.branchy_repr("skew3"))
    synchronized = R.flow(V.build(ir, 2, None), **R.SYNCHRONIZED_LIF)
    streamed = R.flow(V.build(ir, 2, None), **R.STREAMED_LIF)
    assert integrates_then_emits(synchronized) and not integrates_then_emits(streamed)
    assert not runs_closed_form(synchronized, recording=False)
    synchronized.window_count_closed_form = True
    assert runs_closed_form(synchronized, recording=False)
    assert not runs_closed_form(synchronized, recording=True)
    streamed.window_count_closed_form = True
    assert not runs_closed_form(streamed, recording=False)
