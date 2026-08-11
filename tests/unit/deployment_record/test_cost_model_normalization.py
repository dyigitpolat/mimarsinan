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


def _all_terms(record):
    model = DeploymentCostModel()
    latency = model.latency(record)
    return (
        list(model.energy(record)) + list(latency)
        + list(model.area(record)) + list(model._throughput(latency))
    )


def test_every_terms_value_is_its_band_nominal():
    # A consumer reading .value expects the nominal corner; a term whose value
    # silently reports a band edge would skew every downstream comparison.
    record = _with_host_walls(make_full_record(), wall_s_total=0.8, invocations=40)
    for term in _all_terms(record):
        if term.band is not None:
            assert term.value == pytest.approx(term.band.nominal), term.name


def test_the_term_set_per_group_is_exact():
    # Pins the published surface: a spurious or vanished term is a contract
    # change, not a silently tolerated difference.
    record = _with_host_walls(make_full_record(), wall_s_total=0.8, invocations=40)
    model = DeploymentCostModel()
    latency = model.latency(record)
    assert {t.name for t in latency} == {
        "programming_s", "core_init_s", "compute_s", "host_ops_s", "sync_s", "total_s",
    }
    assert {t.name for t in model._throughput(latency)} == {"samples_per_s"}
    assert {t.name for t in model.energy(record)} == {
        "measured_sanafe_total_mj", "measured_total_mj", "modeled_core_init_mj",
        "modeled_programming_mj", "modeled_sync_mj", "total_mj",
    }
    assert {t.name for t in model.area(record)} == {
        "cores_used", "cell_occupancy", "cell_waste_fraction",
        "fragmentation_pct", "unusable_space_cells", "unused_area_cells",
    }


def test_owner_signed_coefficients_match_the_schema_doc():
    # docs/deployment_record_schema.md section 7 is the owner-review surface;
    # code drifting from the blessed band must fail, not pass silently.
    from mimarsinan.deployment_record.cost.coefficients import (
        BYTES_PER_CONNECTIVITY_ENTRY,
        PROGRAMMING_BANDWIDTH_BYTES_PER_S,
        SYNC_BARRIER_S,
    )
    assert (BYTES_PER_CONNECTIVITY_ENTRY.low,
            BYTES_PER_CONNECTIVITY_ENTRY.nominal,
            BYTES_PER_CONNECTIVITY_ENTRY.high) == (4.0, 8.0, 16.0)
    assert PROGRAMMING_BANDWIDTH_BYTES_PER_S.nominal == pytest.approx(12.8e9)
    assert (PROGRAMMING_BANDWIDTH_BYTES_PER_S.low,
            PROGRAMMING_BANDWIDTH_BYTES_PER_S.high) == pytest.approx((1e9, 256e9))
    assert SYNC_BARRIER_S.nominal == pytest.approx(8.0e-8)
    assert (SYNC_BARRIER_S.low, SYNC_BARRIER_S.high) == pytest.approx((6.4e-8, 6.4e-7))
