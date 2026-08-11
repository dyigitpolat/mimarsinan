"""The latency decomposition must add commensurable quantities.

``compute_sim_time_s`` is a PER-SAMPLE census (sample 0); host-op walls are
accumulated over every invocation of the run. Summing them raw overstates the
per-sample latency by the sample factor and inverts into a wrong throughput.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from unit.deployment_record.record_fixtures import make_full_record

from mimarsinan.deployment_record.cost import DeploymentCostModel


def _with_host_walls(record, *, wall_s_total, invocations):
    stages = tuple(
        replace(s, wall_s_total=wall_s_total, invocations=invocations)
        if hasattr(s, "op_type") else s
        for s in record.schedule.stages
    )
    schedule = replace(record.schedule, stages=stages)
    latency = replace(
        record.timing.latency,
        host_ops_s=wall_s_total,
        host_ops_s_per_pass=(
            None if invocations in (None, 0) else wall_s_total / invocations
        ),
    )
    timing = replace(record.timing, latency=latency)
    return replace(record, schedule=schedule, timing=timing)


def test_host_op_term_is_normalized_per_pass_not_the_raw_total():
    # 40 invocations of a 0.8 s total = 0.02 s per program traversal; the
    # per-sample decomposition must use 0.02, never 0.8.
    record = _with_host_walls(make_full_record(), wall_s_total=0.8, invocations=40)
    terms = {t.name: t for t in DeploymentCostModel().latency(record)}
    assert terms["host_ops_s"].value == pytest.approx(0.02)
    compute = terms["compute_s"].value
    assert terms["total_s"].value == pytest.approx(
        compute + 0.02 + terms["programming_s"].value
        + terms["core_init_s"].value + terms["sync_s"].value
    )


def test_unnormalizable_host_walls_fail_loud():
    # A measured total with no invocation count cannot be put on a per-sample
    # axis; guessing a divisor would be a proxy presented as a measurement.
    record = _with_host_walls(make_full_record(), wall_s_total=0.8, invocations=None)
    with pytest.raises(ValueError, match="per pass|invocation"):
        DeploymentCostModel().latency(record)


def test_throughput_inverts_the_normalized_latency():
    record = _with_host_walls(make_full_record(), wall_s_total=0.8, invocations=40)
    model = DeploymentCostModel()
    total = {t.name: t for t in model.latency(record)}["total_s"]
    tp = {t.name: t for t in model.throughput(record)}["samples_per_s"]
    assert tp.value == pytest.approx(1.0 / total.value)


def test_the_basis_states_the_per_sample_normalization():
    record = _with_host_walls(make_full_record(), wall_s_total=0.8, invocations=40)
    total = {t.name: t for t in DeploymentCostModel().latency(record)}["total_s"]
    assert "per" in (total.band.basis or "").lower()
    assert "sample" in (total.band.basis or "").lower() or "pass" in (total.band.basis or "").lower()
