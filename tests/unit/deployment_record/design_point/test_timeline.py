"""The pass timeline of an executed program with its write events per pass, and the period's write summary."""

from __future__ import annotations

import math

import pytest

from mimarsinan.deployment_record.design_point.placement import operator_of_nodes
from mimarsinan.deployment_record.design_point.timeline import (
    TimelineError, pass_timeline, program_writes, writes_summary,
)

from . import programs as P


def _timeline(count, **kwargs):
    program = P.residual_program()
    hybrid = P.built(count, **kwargs)
    platform = P.platform(count, **kwargs)
    writes = program_writes(hybrid, platform, weight_bits=platform["weight_bits"])
    return hybrid, writes, pass_timeline(hybrid, writes, operator_of_nodes(program.ir, P.perceptron_names(program)))


def test_entries_follow_the_stage_order_and_kinds():
    hybrid, _writes, timeline = _timeline(3)
    assert [e["position"] for e in timeline] == list(range(len(hybrid.stages)))
    assert [e["kind"] for e in timeline] == ["pass" if s.kind == "neural" else "host" for s in hybrid.stages]
    assert [e["pass"] for e in timeline if e["kind"] == "pass"] == list(range(sum(s.kind == "neural"
                                                                                  for s in hybrid.stages)))


def test_write_events_per_pass_are_the_write_units_over_the_lanes_and_sum_to_the_period():
    _hybrid, writes, timeline = _timeline(3)
    passes = [e for e in timeline if e["kind"] == "pass"]
    summary = writes_summary(writes)
    assert summary["written_passes"] >= 2 and summary["static_written_cells_per_period"] > 0
    assert sum(e["static_written_cells_per_period"] for e in passes) == summary["static_written_cells_per_period"]
    for e in passes:  # cell granularity, one unit per write and one port: one event per written cell
        assert e["write_units_per_period"] == e["static_written_cells_per_period"]
        assert e["write_events_per_period"] == math.ceil(e["write_units_per_period"] / 1)
        assert e["operand_written_cells_per_inference"] == 0 and e["operand_write_events_per_inference"] is None
    assert summary["write_events_per_inference"] == pytest.approx(sum(e["write_events_per_period"] for e in passes))


def test_the_batch_divides_the_static_writes_per_inference_only():
    _h, writes_1, _t = _timeline(3, k=1)
    _h, writes_16, _t = _timeline(3, k=16)
    one, sixteen = writes_summary(writes_1), writes_summary(writes_16)
    assert one["static_written_cells_per_period"] == sixteen["static_written_cells_per_period"]
    assert sixteen["write_events_per_inference"] == pytest.approx(one["write_events_per_inference"] / 16)
    assert sixteen["max_writes_per_cell_per_inference"] == pytest.approx(one["max_writes_per_cell_per_inference"] / 16)


def test_an_undeclared_batch_has_no_summary_by_name():
    program = P.residual_program()
    platform = {k: v for k, v in P.platform(3).items() if k != "inference_batch_K"}
    with pytest.raises(TimelineError):
        writes_summary(program_writes(P.built(3), platform, weight_bits=platform["weight_bits"]))
