"""SoftCoreMappingStep capacity gate: reject provably-infeasible IR EARLY.

The gate runs AFTER the IR is built and BEFORE HardCoreMapping. When the static
``estimate_cores_needed`` lower bound exceeds the declared core budget, the step
raises ``CapacityExceededError`` (naming the overflowing segment + counts) — the
diagnosable early failure that replaces the late greedy-packer crash. Feasible
configs pass untouched; the gate is opt-out via ``capacity_gate=False``.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import MockPipeline
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.verification.capacity import CapacityExceededError
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import (
    SoftCoreMappingStep,
)


def _src(specs):
    return np.array(
        [IRSource(node_id=nid, index=idx) for nid, idx in specs], dtype=object,
    )


def _core(node_id, name, in_count, out_count):
    return NeuralCore(
        id=node_id, name=name,
        input_sources=_src([(-2, i) for i in range(in_count)]),
        core_matrix=np.ones((in_count, out_count), dtype=np.float64),
        threshold=1.0, latency=0,
    )


def _feasible_ir():
    a = _core(0, "A", in_count=2, out_count=2)
    return IRGraph(nodes=[a], output_sources=_src([(0, 0)]))


def _infeasible_ir():
    huge = _core(0, "huge", in_count=80, out_count=80)
    return IRGraph(nodes=[huge], output_sources=_src([(0, 0)]))


_TINY_BUDGET = {
    "cores": [{"max_axons": 8, "max_neurons": 8, "count": 2, "has_bias": True}],
    "allow_coalescing": True,
}


def _step(config_overrides=None):
    cfg = {"capacity_gate": True}
    cfg.update(config_overrides or {})
    return SoftCoreMappingStep(MockPipeline(config=cfg))


def test_feasible_ir_passes_capacity_gate():
    step = _step()
    # must not raise; returns the estimate so callers can log it
    est = step._run_capacity_gate(_feasible_ir(), _TINY_BUDGET)
    assert est.feasible is True
    assert est.cores_needed <= est.cores_available


def test_infeasible_ir_raises_capacity_exceeded_naming_segment():
    step = _step()
    with pytest.raises(CapacityExceededError) as exc:
        step._run_capacity_gate(_infeasible_ir(), _TINY_BUDGET)
    # huge: frags=ceil(80/8)=10, groups=10 → 100 cores on a 2-core budget
    assert exc.value.cores_needed == 100
    assert exc.value.cores_available == 2
    assert exc.value.overflowing_segment == "neural_segment_final"
    assert "100" in str(exc.value)
    assert "neural_segment_final" in str(exc.value)


def test_gate_disabled_admits_infeasible_ir():
    step = _step({"capacity_gate": False})
    # disabled gate must NOT raise even on a provably-infeasible IR
    est = step._run_capacity_gate(_infeasible_ir(), _TINY_BUDGET)
    assert est is None


def test_gate_default_on_when_key_absent():
    """No ``capacity_gate`` key → gate defaults ON (the diagnostic is the point)."""
    step = SoftCoreMappingStep(MockPipeline(config={}))
    with pytest.raises(CapacityExceededError):
        step._run_capacity_gate(_infeasible_ir(), _TINY_BUDGET)


def test_overflowing_segment_named_across_barrier():
    """Multi-segment IR: the gate names the first segment whose cumulative bound
    overflows, not just the whole graph."""
    a = _core(0, "small", 2, 2)
    barrier = ComputeOp(
        id=1, name="pool", input_sources=_src([(0, 0)]),
        op_type="identity", input_shape=(2,), output_shape=(2,),
    )
    huge = _core(2, "huge", in_count=80, out_count=80)
    huge.input_sources = _src([(1, i) for i in range(80)])
    graph = IRGraph(nodes=[a, barrier, huge], output_sources=_src([(2, 0)]))
    step = _step()
    with pytest.raises(CapacityExceededError) as exc:
        step._run_capacity_gate(graph, _TINY_BUDGET)
    assert exc.value.overflowing_segment == "neural_segment_final"


def _two_segment_sum_over_peak_ir():
    """Two equal (2,3) segments split by a barrier: SUM=8, PEAK=4 on an 8x8 grid."""
    nodes = [_core(i, f"A{i}", 2, 3) for i in range(10)]
    barrier = ComputeOp(
        id=10, name="pool", input_sources=_src([(9, 0)]),
        op_type="identity", input_shape=(3,), output_shape=(3,),
    )
    nodes.append(barrier)
    for j in range(10):
        c = _core(11 + j, f"B{j}", 2, 3)
        c.input_sources = _src([(10, i % 3) for i in range(2)])
        nodes.append(c)
    return IRGraph(nodes=nodes, output_sources=_src([(20, 0)]))


_SCHEDULED_BUDGET = {
    "cores": [{"max_axons": 8, "max_neurons": 8, "count": 5, "has_bias": True}],
    "allow_coalescing": True,
    "allow_scheduling": True,
}

_UNSCHEDULED_BUDGET = dict(_SCHEDULED_BUDGET, allow_scheduling=False)


def test_scheduled_peak_fits_does_not_raise():
    """SUM (8) > budget (5) but PEAK (4) ≤ budget: scheduling-aware gate ADMITS."""
    step = _step()
    graph = _two_segment_sum_over_peak_ir()
    est = step._run_capacity_gate(graph, _SCHEDULED_BUDGET)
    assert est.scheduled is True
    assert est.feasible is True
    assert est.peak_phase_cores == 4
    assert est.phase_count == 2


def test_unscheduled_same_ir_still_raises():
    """The SAME IR/budget WITHOUT scheduling: SUM (8) > 5 → still rejected."""
    step = _step()
    graph = _two_segment_sum_over_peak_ir()
    with pytest.raises(CapacityExceededError):
        step._run_capacity_gate(graph, _UNSCHEDULED_BUDGET)


def test_scheduled_atomic_overflow_still_raises():
    """A single bundle bigger than the whole budget cannot split across phases —
    the scheduling-aware gate STILL raises."""
    step = _step()
    # huge (80,80) on 8x8 → atomic unit 100 > 50-core budget, even scheduled.
    budget = {
        "cores": [{"max_axons": 8, "max_neurons": 8, "count": 50, "has_bias": True}],
        "allow_coalescing": True,
        "allow_scheduling": True,
    }
    with pytest.raises(CapacityExceededError):
        step._run_capacity_gate(_infeasible_ir(), budget)


class TestPackerDivergenceBandWarning:
    """Feasible-but-within-band unscheduled estimates warn loud (W2 Q2a)."""

    def _band_ir(self):
        # Two 8-axon cores on a 2-core budget: needed == budget == 2, and
        # 2 * 1.45 > 2 puts the verdict inside the divergence band.
        a = _core(0, "A", in_count=8, out_count=8)
        b = _core(1, "B", in_count=8, out_count=8)
        b.input_sources = _src([(0, i) for i in range(8)])
        return IRGraph(nodes=[a, b], output_sources=_src([(1, 0)]))

    def test_within_band_estimate_warns(self):
        step = _step()
        with pytest.warns(UserWarning, match="divergence band"):
            est = step._run_capacity_gate(self._band_ir(), _TINY_BUDGET)
        assert est.feasible is True

    def test_comfortable_estimate_does_not_warn(self):
        import warnings as _warnings

        step = _step()
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            est = step._run_capacity_gate(_feasible_ir(), _TINY_BUDGET)
        assert est.feasible is True
